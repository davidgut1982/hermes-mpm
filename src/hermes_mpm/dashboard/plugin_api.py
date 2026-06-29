"""MPM Runs dashboard plugin — backend API routes (read-only).

Why: The hermes-mpm run history lives in ``<hermes_home>/mpm_runs.db`` and is
otherwise only visible through the ``hermes mpm runs`` CLI. This router surfaces
the same data to the dashboard so an operator can watch live/finished subagent
runs in a browser. It is deliberately READ-ONLY and MAINTENANCE-FREE: the
dashboard process is one of several that load this plugin, and a maintenance
write from a non-gateway process is exactly what historically corrupted run
state (see ``runs_db.sweep_orphaned``'s docstring). Therefore this module owns
its OWN write-incapable reader connection (``_reader_connect``) and reimplements
the two read queries as plain SELECTs — it never imports ``runs_db._connect`` /
``query_runs`` (which carry the ``journal_mode`` WAL pragma — a WRITE on a
non-WAL filesystem) and never calls ``init_db``, ``sweep_orphaned``,
``record_*``, ``purge_old`` or any write/DDL path. The only thing it borrows
from ``runs_db`` is pure path resolution (``_db_path``) and the status-name
constants. If the DB does not exist yet, reads degrade to an empty result
rather than creating/sweeping it.

What: A FastAPI ``APIRouter`` mounted at ``/api/plugins/mpm-runs/`` exposing
``GET /runs`` (filtered, newest-first) and ``GET /runs/stats`` (status→count).

Test: ``pytest src/hermes_mpm/tests/test_dashboard_plugin_api.py`` — asserts the
routes return filtered/aggregated results, that the reader issues NO journal_mode
pragma and does not mutate the DB FILE (main-file mtime/size unchanged,
``_apply_wal_with_fallback`` never called; ``query_only=ON`` blocks both SQL
content writes and the journal_mode header rewrite — it does not stop SQLite from
touching the gateway-owned ``-wal``/``-shm`` sidecars on a WAL read), that a live
WAL-mode DB reads correctly, that a malformed DB degrades to empty (no 500), and
that importing this module / calling its routes triggers NO sweep / init / write.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Optional

from fastapi import APIRouter, Query

from hermes_mpm import runs_db

log = logging.getLogger(__name__)

router = APIRouter()

# Connection-level busy wait for the reader — borrowed from runs_db so a read
# overlapping a gateway write burst waits in C rather than raising "locked".
_READER_BUSY_TIMEOUT_MS = runs_db._BUSY_TIMEOUT_MS


def _reader_connect() -> sqlite3.Connection:
    """Open a WRITE-INCAPABLE connection to the run DB for read-only routes.

    Why: ``runs_db._connect`` runs ``_apply_wal_with_fallback``, which can issue
    ``PRAGMA journal_mode=WAL|DELETE`` — a WRITE to the DB header/sidecars — from
    this non-gateway dashboard process. On a non-WAL filesystem that is a real
    on-disk mutation from a process that must stay strictly read-only (the same
    cross-process-write class of bug that historically corrupted run state).
    This reader instead refuses writes at the connection level and NEVER touches
    journal_mode, so it reads the gateway's live WAL DB without ever writing.
    What: Opens a plain ``sqlite3.connect`` (no parent ``mkdir``: callers gate on
    ``_db_exists`` so the file is never created), sets ``busy_timeout`` so reads
    wait out write locks, sets ``query_only = ON`` (a per-connection flag — NOT a
    disk write — that makes the connection refuse every INSERT/UPDATE/DDL), and a
    ``Row`` factory for dict access. It issues NO ``journal_mode`` pragma and
    calls NONE of ``runs_db``'s write/connect helpers.
    Test: ``test_reader_does_not_write_db`` (mtime/size unchanged, no
    ``_apply_wal_with_fallback`` call) + ``test_reader_reads_live_wal_db``.
    """
    conn = sqlite3.connect(
        str(runs_db._db_path()),
        check_same_thread=False,
        timeout=_READER_BUSY_TIMEOUT_MS / 1000.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={_READER_BUSY_TIMEOUT_MS}")
    # Connection-level read-only guard: refuses any write/DDL on this connection.
    # This is a session flag held in memory, NOT a write to the database file,
    # and (unlike mode=ro URIs) it reads a live WAL DB without -shm complications.
    conn.execute("PRAGMA query_only = ON")
    return conn


# Status vocabulary mirrors runs_db; used to give /runs/stats a stable key set
# even for statuses that currently have zero rows.
_KNOWN_STATUSES = (
    runs_db.STATUS_RUNNING,
    runs_db.STATUS_DONE,
    runs_db.STATUS_FAILED,
    runs_db.STATUS_CRASHED,
    runs_db.STATUS_TIMED_OUT,
)


def _db_exists() -> bool:
    """True iff the run DB file already exists — the read gate for both routes.

    Why: ``runs_db._connect`` (and ``query_runs`` through it) calls
    ``path.parent.mkdir(...)`` + ``sqlite3.connect(path)``, which CREATES the
    file plus ``-wal``/``-shm`` sidecars as a side effect. If the dashboard
    polled before the gateway ever created ``mpm_runs.db``, it would create that
    file from a non-gateway process — possibly under a different uid, breaking
    the gateway's later tracking writes — the exact cross-process hazard this
    plugin promises to avoid. Gating on existence keeps the routes strictly
    read-only: no file is ever created by the dashboard process.
    What: Returns whether ``runs_db._db_path()`` points at an existing file.
    Test: ``test_routes_degrade_when_db_missing`` asserts the file stays absent
    after both routes are called against a missing DB.
    """
    try:
        return runs_db._db_path().exists()
    except Exception:
        return False


def _query_runs(
    status: Optional[str],
    session: Optional[str],
    since: Optional[int],
    limit: int,
) -> list[dict[str, Any]]:
    """Local read-only reimplementation of ``runs_db.query_runs`` (no WAL pragma).

    Why: ``runs_db.query_runs`` reaches the DB via ``_connect`` →
    ``_apply_wal_with_fallback``, which can WRITE a ``journal_mode`` pragma from
    this read-only process. Reimplementing the SELECT over ``_reader_connect``
    (``query_only=ON``, no journal_mode) gives the dashboard the same rows with a
    genuinely write-free connection.
    What: Builds the same WHERE filters as ``query_runs`` — ``status``,
    ``parent_session_id`` (the ``session`` param), ``started_at >= since`` —
    ``ORDER BY started_at DESC LIMIT ?``, and returns plain dicts.
    Test: ``test_get_runs_*`` (parity with the old query_runs results) +
    ``test_reader_reads_live_wal_db``.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if session:
        clauses.append("parent_session_id = ?")
        params.append(session)
    if since is not None:
        clauses.append("started_at >= ?")
        params.append(int(since))
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = "SELECT * FROM subagent_runs" + where + " ORDER BY started_at DESC LIMIT ?"
    params.append(int(limit))

    conn = _reader_connect()
    try:
        return [dict(r) for r in conn.execute(sql, tuple(params))]
    finally:
        conn.close()


@router.get("/runs")
def get_runs(
    status: Optional[str] = Query(None),
    session: Optional[str] = Query(None),
    since: Optional[int] = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """Return runs matching the filters, newest first, plus the server clock.

    Why: Backs the dashboard runs table. ``now`` is returned so the frontend can
    compute live ages/durations against the SERVER clock (avoids client-clock
    skew making a fresh run look hours old). Read-only: delegates to the local
    write-free ``_query_runs`` (over ``_reader_connect`` with ``query_only=ON``)
    — NOT ``runs_db.query_runs``, which would route through the journal_mode WAL
    write this module exists to avoid. No init, no sweep, no write.
    What: ``GET /runs?status=&session=&since=&limit=`` →
    ``{"runs": [...], "now": <epoch_seconds>}``. A missing DB yields ``{"runs":
    [], "now": ...}`` WITHOUT creating the file (the gateway hasn't created it
    yet — the dashboard must not).
    Test: ``test_get_runs_returns_query_runs`` + ``test_get_runs_passes_*`` +
    ``test_routes_degrade_when_db_missing``.
    """
    if not _db_exists():
        # No DB file yet — return empty WITHOUT opening a connection, which would
        # create the file + WAL sidecars from a non-gateway process.
        return {"runs": [], "now": int(time.time())}
    try:
        rows = _query_runs(status=status, session=session, since=since, limit=limit)
    except sqlite3.Error:
        # Table not created yet (file exists but gateway hasn't run init), or a
        # malformed/corrupt DB — degrade to empty rather than a 500.
        rows = []
    return {"runs": rows, "now": int(time.time())}


@router.get("/runs/stats")
def get_runs_stats() -> dict[str, Any]:
    """Return a status→count aggregate over all runs (read-only).

    Why: Powers the status summary chips above the table without pulling every
    row to the client. Uses a single ``GROUP BY status`` aggregate over the
    local write-incapable ``_reader_connect`` — NO journal_mode pragma, NO init,
    NO sweep, NO write.
    What: ``GET /runs/stats`` → ``{"stats": {<status>: <count>, ...}, "total":
    <int>}``. Every known status is present (zero-filled) so the UI chip set is
    stable; an unexpected status value still appears if the DB holds one. A
    missing DB — or a malformed/corrupt one — yields all-zero counts WITHOUT
    creating the file and WITHOUT a 500.
    Test: ``test_get_runs_stats_aggregates`` +
    ``test_routes_do_not_trigger_sweep_or_write`` +
    ``test_routes_degrade_when_db_missing`` + ``test_routes_degrade_on_malformed_db``.
    """
    counts: dict[str, int] = {s: 0 for s in _KNOWN_STATUSES}
    if not _db_exists():
        # No DB file yet — zero-filled WITHOUT opening a connection (no file create).
        return {"stats": counts, "total": 0}
    try:
        conn = _reader_connect()
    except sqlite3.Error:
        return {"stats": counts, "total": 0}
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM subagent_runs GROUP BY status"
        ).fetchall()
        for row in rows:
            counts[row["status"]] = int(row["n"])
    except sqlite3.Error:
        # Table not created yet, or a malformed/corrupt DB — zero-filled set.
        return {"stats": counts, "total": 0}
    finally:
        conn.close()
    return {"stats": counts, "total": sum(counts.values())}

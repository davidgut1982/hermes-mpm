"""MPM Runs dashboard plugin — backend API routes (read-only).

Why: The hermes-mpm run history lives in ``<hermes_home>/mpm_runs.db`` and is
otherwise only visible through the ``hermes mpm runs`` CLI. This router surfaces
the same data to the dashboard so an operator can watch live/finished subagent
runs in a browser. It is deliberately READ-ONLY and MAINTENANCE-FREE: the
dashboard process is one of several that load this plugin, and a maintenance
write from a non-gateway process is exactly what historically corrupted run
state (see ``runs_db.sweep_orphaned``'s docstring). Therefore this module
imports ONLY ``query_runs`` (pure read) and ``_connect`` (read connection) — it
never calls ``init_db``, ``sweep_orphaned``, ``record_*``, ``purge_old`` or any
write/DDL path. If the DB does not exist yet, reads degrade to an empty result
rather than creating/sweeping it.

What: A FastAPI ``APIRouter`` mounted at ``/api/plugins/mpm-runs/`` exposing
``GET /runs`` (filtered, newest-first) and ``GET /runs/stats`` (status→count).

Test: ``pytest src/hermes_mpm/tests/test_dashboard_plugin_api.py`` — asserts the
routes return ``query_runs`` results with filters, the stats aggregate, and that
importing this module / calling its routes triggers NO sweep / init / write.
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
    skew making a fresh run look hours old). Read-only: delegates straight to
    ``runs_db.query_runs`` (no init, no sweep, no write).
    What: ``GET /runs?status=&session=&since=&limit=`` →
    ``{"runs": [...], "now": <epoch_seconds>}``. A missing DB yields ``{"runs":
    [], "now": ...}`` WITHOUT creating the file (the gateway hasn't created it
    yet — the dashboard must not).
    Test: ``test_get_runs_returns_query_runs`` + ``test_get_runs_passes_*`` +
    ``test_routes_degrade_when_db_missing``.
    """
    if not _db_exists():
        # No DB file yet — return empty WITHOUT calling query_runs/_connect,
        # which would create the file + WAL sidecars from a non-gateway process.
        return {"runs": [], "now": int(time.time())}
    try:
        rows = runs_db.query_runs(
            status=status,
            session=session,
            since=since,
            limit=limit,
        )
    except sqlite3.OperationalError:
        # Table not created yet (file exists but gateway hasn't run init) — empty.
        rows = []
    return {"runs": rows, "now": int(time.time())}


@router.get("/runs/stats")
def get_runs_stats() -> dict[str, Any]:
    """Return a status→count aggregate over all runs (read-only).

    Why: Powers the status summary chips above the table without pulling every
    row to the client. Uses a single ``GROUP BY status`` aggregate over a
    read connection (``runs_db._connect``) — NO init, NO sweep, NO write.
    What: ``GET /runs/stats`` → ``{"stats": {<status>: <count>, ...}, "total":
    <int>}``. Every known status is present (zero-filled) so the UI chip set is
    stable; an unexpected status value still appears if the DB holds one. A
    missing DB yields all-zero counts WITHOUT creating the file.
    Test: ``test_get_runs_stats_aggregates`` +
    ``test_routes_do_not_trigger_sweep_or_write`` +
    ``test_routes_degrade_when_db_missing``.
    """
    counts: dict[str, int] = {s: 0 for s in _KNOWN_STATUSES}
    if not _db_exists():
        # No DB file yet — zero-filled WITHOUT _connect() creating the file.
        return {"stats": counts, "total": 0}
    try:
        conn = runs_db._connect()
    except sqlite3.OperationalError:
        return {"stats": counts, "total": 0}
    try:
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM subagent_runs GROUP BY status"
            ).fetchall()
        except sqlite3.OperationalError:
            # Table not created yet — return the zero-filled known set.
            return {"stats": counts, "total": 0}
        for row in rows:
            counts[row["status"]] = int(row["n"])
    finally:
        conn.close()
    return {"stats": counts, "total": sum(counts.values())}

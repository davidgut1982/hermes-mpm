"""Durable SQLite run-tracking + observability for hermes-mpm subagent runs.

Why: Delegations (sync + async/background) are fire-and-forget from the PM's
view — when the gateway restarts mid-run, or an async child finishes while the
parent has moved on, there is no durable record of what ran, how long, and how
it ended. This module is the spine of that observability: every subagent start
and stop is written to ``<hermes_home>/mpm_runs.db`` so ``hermes mpm runs`` can
answer "what ran / is running / crashed" across restarts. The durability fill is
``sweep_orphaned`` at startup: any run left ``running`` by a prior (dead) process
is marked ``crashed`` instead of lingering forever. A parallel ``turn_batches``
table records the per-turn assistant tool-call count for BOTH the main agent and
every subagent so ``hermes mpm parallelism`` can report a trustworthy batch rate.

What: A self-contained SQLite layer (no hermes_cli internals imported — the
connection pattern is *replicated* from kanban_db/hermes_state to avoid version
coupling) exposing ``init_db``, ``record_start``, ``record_end``,
``record_batch_telemetry``, ``sweep_orphaned``, ``query_runs``, ``purge_old``,
the delegation-correlation helpers (``find_running_by_delegation``,
``find_running_by_goal``, ``stamp_delegation_id``), and the turn-batch telemetry
functions (``record_turn_batch``, ``batch_stats``, ``record_run_turn``,
``purge_old_turn_batches``). DDL runs only in ``init_db`` (never on a hot path —
concurrent DDL is what corrupted the kanban DB historically). Writes use
``BEGIN IMMEDIATE`` with app-level retry-with-jitter on transient "database is
locked". The primary run table is named ``runs``.

Test: ``pytest src/hermes_mpm/tests/test_runs_db.py`` +
``test_turn_batches.py`` — covers create/close, orphan-sweep (pid-liveness
scoped), delegation correlation + ambiguity guard, filtered query, purge,
batch telemetry insert/aggregate/fold/purge, idempotent re-init, and concurrent
writers.
"""

from __future__ import annotations

import logging
import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("hermes_mpm.runs_db")

# SQLite busy timeout — 5s matches the engine's timeout. Concurrent writes
# (common: subagent_start + subagent_stop from different hooks in the same turn)
# will wait; if the busy is exhausted the write fails closed.
_BUSY_TIMEOUT_MS = 5000

# Run-DB filename — always lands under get_hermes_home() so CLI/gateway agree.
DB_FILENAME = "mpm_runs.db"

# --- shared status vocabulary ----------------------------------------------
# These are the terminal/running status strings the ``runs`` table stores. They
# are legitimate shared API: the hook layer (__init__.py) maps the engine's
# child_status onto them and the dashboard uses them to zero-fill its stat chips,
# so callers and the schema agree on ONE set of values. Kept as module-level
# constants (string subclass values) rather than magic strings.
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CRASHED = "crashed"
STATUS_TIMED_OUT = "timed_out"

# Marker written into ``error`` when the orphan sweep reaps a run left running by
# a prior dead process — surfaced by the CLI/dashboard so the crash is explained.
_ORPHAN_ERROR = "orphaned by restart"

# Table-creation DDL — idempotent, columns only (no indexes: a legacy table
# migrated by RENAME may lack the new columns, so indexes are created AFTER
# _ensure_columns fills them in — see init_db).
_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    parent_session_id TEXT,
    role TEXT,
    profile TEXT,
    goal TEXT,
    status TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    ended_at INTEGER,
    duration_ms INTEGER,
    summary TEXT,
    error TEXT,
    delegation_id TEXT,
    run_type TEXT,
    metadata TEXT,
    owner_pid INTEGER,
    batch_count INTEGER,
    max_batch_size INTEGER,
    turn_count INTEGER,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Per-turn assistant tool-call count (the parallelism signal), one row per LLM
-- turn that emitted >=1 tool call, for BOTH the main agent and every subagent.
-- The main agent has no ``runs`` row, so this separate store is what lets
-- ``batch_stats`` compute a GLOBAL batch rate (folding into ``runs`` alone would
-- lose main-agent turns). Keyed by api_request_id (first write wins).
CREATE TABLE IF NOT EXISTS turn_batches (
    api_request_id TEXT PRIMARY KEY,
    turn_id TEXT,
    session_id TEXT,
    model TEXT,
    tool_call_count INTEGER NOT NULL,
    ts INTEGER NOT NULL
);
"""

# Index DDL — created only AFTER columns are guaranteed present (post-migration),
# so an index on a newly-ALTER-added column (e.g. owner_pid on a migrated legacy
# table) never fails with "no such column".
_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_runs_status  ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_parent  ON runs(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_deleg   ON runs(delegation_id);
CREATE INDEX IF NOT EXISTS idx_runs_owner   ON runs(owner_pid);
CREATE INDEX IF NOT EXISTS idx_turn_batches_ts    ON turn_batches(ts);
CREATE INDEX IF NOT EXISTS idx_turn_batches_model ON turn_batches(model);
"""


def _db_path() -> Path:
    """Resolve the run-DB path under the active Hermes home.

    Why: Runs must land in the SAME home the engine uses, so the CLI and the
    hooks agree on one file even under multi-profile / sandbox layouts.
    What: Returns ``get_hermes_home() / mpm_runs.db``. Falls back to
    ``~/.hermes/mpm_runs.db`` if ``hermes_constants`` is unavailable (e.g. the
    plugin loaded outside a full core), so the module never hard-fails on import.
    Test: Patched to a tmp path in tests; the fallback branch is exercised by
    importing without hermes_constants.
    """
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / DB_FILENAME
    except Exception:  # no core on path — degrade to the platform default home
        return Path.home() / ".hermes" / DB_FILENAME


def _pid_alive(pid: int) -> bool:
    """Return whether ``pid`` is a live process on this host.

    Why: The orphan sweep must only reap runs owned by a DEAD prior process. A
    running row owned by a different but still-alive process is genuine in-flight
    work and must never be marked crashed — reaping it would corrupt live state.
    What: Uses ``os.kill(pid, 0)`` — no signal is sent; it only probes existence.
    Returns True if the process exists (or exists-but-not-permitted, ESRCH=False),
    False if no such process. Non-positive pids are treated as not-alive.
    Test: ``test_sweep_orphaned_skips_alive_other_owner`` (monkeypatched True) +
    the reap tests (monkeypatched False).
    """
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user — still alive
    except OSError:
        return False
    return True


def _apply_wal_with_fallback(conn: sqlite3.Connection) -> None:
    """Set journal_mode=WAL, falling back to DELETE on WAL-incompatible FS.

    Why: WAL gives concurrent readers + one writer (what the hooks need), but it
    fails on some network filesystems (e.g. ZFS with certain settings, NFS).
    Falling back to DELETE maintains durability at the cost of concurrency.
    What: Tries ``PRAGMA journal_mode=WAL``; on error sets ``DELETE``. Logs mode.
    Test: Covered by test_init_db_on_wal_incompatible_fs.
    """
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA journal_mode").fetchone()
        logger.debug("mpm-runs: journal_mode=WAL")
    except sqlite3.OperationalError:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA journal_mode").fetchone()
        logger.debug("mpm-runs: journal_mode=DELETE (WAL fallback)")


def _connect() -> sqlite3.Connection:
    """Open a run-DB connection with the standard lock-waiting PRAGMAs.

    Why: Every hook handler and the CLI need a consistently-configured
    connection (autocommit + busy wait) so explicit ``BEGIN IMMEDIATE`` controls
    write transactions and lock contention degrades to waiting, not errors.
    What: Opens with check_same_thread=False, isolation_level=None (autocommit),
    timeout/busy_timeout=5000ms, WAL (or DELETE fallback), Row factory. Creates
    the parent dir best-effort (retried) so a fresh home never fails the write.
    Test: Indirectly via every test; concurrency test stresses the lock waiting.
    """
    path = _db_path()
    for attempt in range(3):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            break
        except OSError as exc:
            if attempt == 2:
                logger.error("hermes-mpm: failed to create parent dir %s: %s", path.parent, exc)
            else:
                time.sleep(0.1 * (2 ** attempt))

    conn = sqlite3.connect(
        str(path),
        check_same_thread=False,
        isolation_level=None,
        timeout=_BUSY_TIMEOUT_MS / 1000.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    _apply_wal_with_fallback(conn)
    return conn


def _write(sql: str, params: tuple) -> int:
    """Run one write statement in a BEGIN IMMEDIATE txn with retry-with-jitter.

    Why: SQLite handles concurrent readers well, but writers contend. Without
    an explicit BEGIN IMMEDIATE the first write might conflict with another
    writer and get a "database is locked" error. The jitter avoids a thundering
    herd when multiple processes hit the same locked DB.
    What: Begins IMMEDIATE, executes, commits, returns the affected rowcount. On
    "database is locked" sleeps a random 0.01–0.1s and retries up to 3 times.
    Test: ``test_write_retries_on_locked_db`` + every write-path test.
    """
    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(sql, params)
        conn.commit()
        return cursor.rowcount
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc).lower():
            conn.close()
            for attempt in range(3):
                time.sleep(random.uniform(0.01, 0.1))
                try:
                    conn = _connect()
                    cursor = conn.cursor()
                    cursor.execute("BEGIN IMMEDIATE")
                    cursor.execute(sql, params)
                    conn.commit()
                    return cursor.rowcount
                except sqlite3.OperationalError:
                    if attempt == 2:
                        raise
                    continue
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def init_db() -> None:
    """Create the schema + indexes once. Safe to call repeatedly (idempotent).

    Why: DDL must run exactly at startup, never on a hot write path — concurrent
    CREATE/ALTER is what historically corrupted the kanban DB. Centralizing it
    here keeps record_start/record_end DDL-free. The ALTERs migrate DBs created
    before owner_pid / batch columns existed (a legacy ``runs`` table, or a
    legacy pre-rename ``subagent_runs`` table renamed by this migration).
    What: Renames a legacy ``subagent_runs`` table to ``runs`` if present, then
    executescript()s the IF NOT EXISTS schema, then ALTER-adds any missing
    columns on an EXISTING ``runs`` table — each guarded by a PRAGMA column check
    so re-runs are a clean no-op.
    Test: ``test_init_db_is_idempotent`` + ``test_init_db_adds_owner_pid_to_existing_db``
    + ``test_init_db_migrates_legacy_subagent_runs``.
    """
    conn = _connect()
    try:
        _migrate_legacy_table_name(conn)
        conn.executescript(_TABLE_SQL)
        _ensure_columns(conn)  # fill columns a migrated legacy table may lack
        conn.executescript(_INDEX_SQL)  # indexes only after columns exist
    finally:
        conn.close()
    logger.debug("mpm-runs: db ready (schema initialized; sweep deferred to first hook)")


def _migrate_legacy_table_name(conn: sqlite3.Connection) -> None:
    """Rename a legacy ``subagent_runs`` table to ``runs`` if it exists.

    Why: Earlier builds named the table ``subagent_runs``; the current schema
    uses ``runs``. A live DB on disk must migrate without losing rows.
    What: If ``subagent_runs`` exists AND ``runs`` does not, ``ALTER TABLE …
    RENAME``. If both exist (shouldn't happen), leave them — the new ``runs`` is
    authoritative. No-op on a fresh DB.
    Test: ``test_init_db_migrates_legacy_subagent_runs``.
    """
    names = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('runs','subagent_runs')"
        )
    }
    if "subagent_runs" in names and "runs" not in names:
        conn.execute("ALTER TABLE subagent_runs RENAME TO runs")
        logger.debug("mpm-runs: migrated legacy subagent_runs -> runs")


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """ALTER-add any columns missing from an EXISTING ``runs`` table.

    Why: A legacy on-disk table may predate owner_pid / batch telemetry columns;
    the sweep + telemetry paths require them. Each add is guarded so re-running
    init_db is a clean no-op.
    What: Reads PRAGMA table_info(runs); ADDs owner_pid, batch_count,
    max_batch_size, turn_count, summary, error, metadata, delegation_id if absent.
    Test: ``test_init_db_adds_owner_pid_to_existing_db`` +
    ``test_init_db_migrates_legacy_subagent_runs``.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    for col in (
        "owner_pid",
        "batch_count",
        "max_batch_size",
        "turn_count",
        "summary",
        "error",
        "metadata",
        "delegation_id",
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} INTEGER" if col in (
                "owner_pid",
                "batch_count",
                "max_batch_size",
                "turn_count",
            ) else f"ALTER TABLE runs ADD COLUMN {col} TEXT")
            logger.debug("mpm-runs: added %s column", col)


def record_start(
    run_id: str,
    parent_session_id: Optional[str],
    role: Optional[str],
    profile: Optional[str],
    goal: Optional[str],
    started_at: int,
    run_type: str,
    *,
    delegation_id: Optional[str] = None,
    owner_pid: Optional[int] = None,
) -> None:
    """Record a run start. Idempotent — reinserting the same run_id is a no-op.

    Why: Every subagent_start hook fires once per spawned child. We capture the
    initial state so the PM can track what's running and orphaned runs can be
    detected. ``owner_pid`` defaults to THIS process's pid so the sweep can tell
    its own in-flight rows from a dead prior process's.
    What: INSERT OR IGNORE the run row with status='running'. Accepts args
    positionally (tests) or by keyword (hooks). Uses _write for concurrency.
    Test: ``test_record_start_creates_running_row``,
    ``test_record_start_insert_or_ignore_is_idempotent``,
    ``test_record_start_stamps_owner_pid``.
    """
    if owner_pid is None:
        owner_pid = os.getpid()
    _write(
        """
        INSERT OR IGNORE INTO runs (
            run_id, parent_session_id, role, profile, goal,
            status, started_at, run_type, delegation_id, owner_pid
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            parent_session_id,
            role,
            profile,
            goal,
            STATUS_RUNNING,
            int(started_at),
            run_type,
            delegation_id,
            owner_pid,
        ),
    )


def record_end(
    run_id: str,
    *,
    status: str,
    ended_at: int,
    duration_ms: Optional[int] = None,
    summary: Optional[str] = None,
    error: Optional[str] = None,
    batch_count: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    turn_count: Optional[int] = None,
) -> None:
    """Update a running row to its terminal state. Idempotent.

    Why: subagent_stop (sync) and pre_llm_call (async complete) both need to
    close a run. Only a still-``running`` row is closed, so a duplicate close is a
    no-op. Optional batch telemetry may be folded in at close time.
    What: UPDATEs the row WHERE run_id matches AND status='running', setting the
    terminal status, ended_at, and any provided optional columns. Columns not
    provided are left unchanged (COALESCE) so a close never wipes telemetry a
    prior hook recorded.
    Test: ``test_record_end_closes_row``, ``test_record_end_records_error``.
    """
    _write(
        """
        UPDATE runs
           SET status = ?,
               ended_at = ?,
               duration_ms = COALESCE(?, duration_ms),
               summary = COALESCE(?, summary),
               error = COALESCE(?, error),
               batch_count = COALESCE(?, batch_count),
               max_batch_size = COALESCE(?, max_batch_size),
               turn_count = COALESCE(?, turn_count)
         WHERE run_id = ? AND status = ?
        """,
        (
            status,
            int(ended_at),
            duration_ms,
            summary,
            error,
            batch_count,
            max_batch_size,
            turn_count,
            run_id,
            STATUS_RUNNING,
        ),
    )


def record_batch_telemetry(
    run_id: str,
    *,
    batch_count: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    turn_count: Optional[int] = None,
) -> None:
    """Fold batch telemetry into a running run row (consolidated entry point).

    Why: A single, explicit entry point for stamping the per-run batch columns
    (batch_count / max_batch_size / turn_count) without going through the turn
    counting logic — used when a caller already has aggregate batch numbers.
    What: UPDATEs the RUNNING run's batch columns, COALESCE-preserving any column
    not supplied. No-op if the run doesn't exist or isn't running.
    Test: ``test_record_batch_telemetry_updates_row``.
    """
    _write(
        """
        UPDATE runs
           SET batch_count = COALESCE(?, batch_count),
               max_batch_size = COALESCE(?, max_batch_size),
               turn_count = COALESCE(?, turn_count)
         WHERE run_id = ? AND status = ?
        """,
        (batch_count, max_batch_size, turn_count, run_id, STATUS_RUNNING),
    )


def query_runs(
    *,
    status: Optional[str] = None,
    run_type: Optional[str] = None,
    session: Optional[str] = None,
    since: Optional[int] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Query runs with optional filters, newest-first.

    Why: The CLI (`hermes mpm runs`) and orchestration list/filter runs.
    What: Builds a dynamic WHERE from status / run_type / parent-session (the
    ``session`` param) / ``started_at >= since``, orders by started_at DESC, and
    limits. Returns a list of dicts (one per row).
    Test: ``test_query_runs_filters_and_orders``.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if run_type:
        clauses.append("run_type = ?")
        params.append(run_type)
    if session:
        clauses.append("parent_session_id = ?")
        params.append(session)
    if since is not None:
        clauses.append("started_at >= ?")
        params.append(int(since))
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = "SELECT * FROM runs" + where + " ORDER BY started_at DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))

    conn = _connect()
    try:
        return [dict(row) for row in conn.execute(sql, tuple(params))]
    finally:
        conn.close()


def sweep_orphaned(now: int, current_pid: Optional[int] = None) -> int:
    """Mark runs left 'running' by a prior DEAD process as 'crashed'.

    Why: If the gateway restarts mid-run, the child processes are killed but the
    DB rows still say 'running'. These orphans would linger forever. This sweep
    marks them 'crashed' so they don't pollute the UI — but ONLY when the owning
    pid is not the current process AND is not still alive, so a concurrent
    still-running sibling process's in-flight work is never reaped.
    What: Finds running rows whose owner_pid is NULL (legacy) OR (!= current_pid
    AND not alive), then UPDATEs each to 'crashed' with error=_ORPHAN_ERROR and
    ended_at=now. Returns the count reaped.
    Test: ``test_sweep_orphaned_marks_running_crashed``,
    ``test_sweep_orphaned_only_reaps_other_pid_rows``,
    ``test_sweep_orphaned_skips_alive_other_owner``.
    """
    if current_pid is None:
        current_pid = os.getpid()
    conn = _connect()
    try:
        candidates = conn.execute(
            "SELECT run_id, owner_pid FROM runs WHERE status = ?",
            (STATUS_RUNNING,),
        ).fetchall()
    finally:
        conn.close()

    reaped = 0
    for row in candidates:
        owner = row["owner_pid"]
        # A row owned by the current process is genuine in-flight work — skip.
        if owner is not None and owner == current_pid:
            continue
        # A row owned by a different but STILL-ALIVE process is also genuine — skip.
        if owner is not None and _pid_alive(owner):
            continue
        # NULL owner (legacy) or a dead prior owner → reap.
        updated = _write(
            "UPDATE runs SET status = ?, error = ?, ended_at = ? "
            "WHERE run_id = ? AND status = ?",
            (STATUS_CRASHED, _ORPHAN_ERROR, int(now), row["run_id"], STATUS_RUNNING),
        )
        reaped += updated
    return reaped


def find_running_by_delegation(delegation_id: str) -> Optional[str]:
    """Return the run_id of a running row carrying ``delegation_id``, if any.

    Why: The async-complete fallback needs to close the right run by its
    delegation_id when one was stamped.
    What: SELECTs the newest running run_id whose delegation_id matches.
    Test: Exercised via the async-complete hook tests in test_runs_hooks.py.
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT run_id FROM runs WHERE delegation_id = ? AND status = ? "
            "ORDER BY started_at DESC LIMIT 1",
            (delegation_id, STATUS_RUNNING),
        ).fetchone()
        return row["run_id"] if row else None
    finally:
        conn.close()


def stamp_delegation_id(goal: str, delegation_id: str, now: Optional[int] = None) -> bool:
    """Stamp ``delegation_id`` onto the newest un-stamped running row for ``goal``.

    Why: For a direct ``delegate_task(background=True)`` the subagent_start hook
    creates the run row (delegation_id NULL) BEFORE delegate_task returns; the
    synchronous post_tool_call then carries the delegation_id. Stamping it here
    lets the async-complete marker close the run by EXACT delegation_id instead of
    fragile goal-text matching. The ``delegation_id IS NULL`` filter disambiguates
    two sequential identical-goal async runs (the second stamp cannot overwrite
    the first — it falls through to the next still-null row).
    What: Finds the newest ``running`` row with this goal AND delegation_id NULL,
    UPDATEs its delegation_id (re-asserting the NULL guard so a concurrent stamp
    can't double-write). Returns True iff a row was stamped. ``now`` is accepted
    for signature symmetry (unused).
    Test: ``test_stamp_delegation_id_*`` in test_runs_db.py.
    """
    del now  # accepted for symmetry; this write mutates no timestamp
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT run_id FROM runs "
            "WHERE goal = ? AND status = ? AND delegation_id IS NULL "
            "ORDER BY started_at DESC LIMIT 1",
            (goal, STATUS_RUNNING),
        ).fetchone()
        run_id = row["run_id"] if row else None
    finally:
        conn.close()
    if run_id is None:
        return False
    updated = _write(
        "UPDATE runs SET delegation_id = ? WHERE run_id = ? AND delegation_id IS NULL",
        (delegation_id, run_id),
    )
    return updated > 0


def find_running_by_goal(goal: str) -> Optional[str]:
    """Return the run_id of the SOLE running row whose goal matches, else None.

    Why: A LAST-RESORT fallback — delegation_id correlation is primary. Goal text
    at completion can be truncated/reformatted, and two async runs can share a
    goal, so this must never GUESS: if more than one running row matches, return
    None and leave the ambiguous pair for the honest crash-sweep.
    What: Counts running rows with an exact goal match; returns the run_id only
    when EXACTLY one matches (ambiguity guard), else None.
    Test: ``test_find_running_by_goal_returns_none_when_ambiguous`` +
    ``test_find_running_by_goal_returns_single_match``.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT run_id FROM runs WHERE goal = ? AND status = ? ORDER BY started_at DESC",
            (goal, STATUS_RUNNING),
        ).fetchall()
        if len(rows) != 1:
            return None  # 0 = no match; >1 = ambiguous, do NOT guess
        return rows[0]["run_id"]
    finally:
        conn.close()


def purge_old(retention_days: int, now: Optional[int] = None) -> int:
    """Delete ended runs older than ``retention_days``. Returns rows deleted.

    Why: Keeps the DB bounded without an external cron — called at startup. Only
    ENDED rows are purged; a still-running row (ended_at NULL) is never deleted,
    even if its start is ancient, so in-flight work is never lost.
    What: DELETEs rows with ended_at IS NOT NULL AND ended_at < (now - retention).
    A non-positive retention is a no-op (retention disabled).
    Test: ``test_purge_old_deletes_only_old_ended_rows``.
    """
    if retention_days <= 0:
        return 0
    if now is None:
        now = int(time.time())
    cutoff = int(now) - retention_days * 86400
    return _write(
        "DELETE FROM runs WHERE ended_at IS NOT NULL AND ended_at < ?",
        (cutoff,),
    )


# --- batch telemetry: per-turn tool-call counts ----------------------------


def record_turn_batch(
    turn_id: Optional[str],
    api_request_id: str,
    session_id: Optional[str],
    model: Optional[str],
    tool_call_count: int,
    ts: int,
) -> int:
    """Record one LLM turn's assistant tool-call count (the batch signal).

    Why: The global, queryable parallelism signal — one row per turn that emitted
    >=1 tool call, for BOTH the main agent and every subagent. >1 means the
    assistant batched tool calls (parallelism working). Recording it durably lets
    ``batch_stats`` compute the batch rate by SELECT/GROUP BY. Callers record only
    turns with tool_call_count >= 1; this writes whatever it is given.
    What: INSERT OR IGNORE one row keyed by api_request_id (first write wins — the
    hook may fire more than once for a logically-single request). Returns the
    rowcount: 1 when newly inserted, 0 on a duplicate — so the caller can gate the
    per-run fold on a NEW turn and avoid double-counting turn_count.
    Test: ``test_record_turn_batch_inserts_row`` +
    ``test_record_turn_batch_idempotent_on_api_request_id`` +
    ``test_record_turn_batch_returns_rowcount``.
    """
    return _write(
        "INSERT OR IGNORE INTO turn_batches "
        "(api_request_id, turn_id, session_id, model, tool_call_count, ts) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (api_request_id, turn_id, session_id, model, int(tool_call_count), int(ts)),
    )


def batch_stats(
    since: Optional[int] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregate the turn-batch store into a parallelism scorecard.

    Why: Backs ``hermes mpm parallelism`` — turns the raw per-turn rows into the
    one number operators ask for: the batch rate (fraction of tool-turns that
    batched >1 call), overall and per model.
    What: SELECT/GROUP BY over turn_batches with optional ts>=since and model=
    filters. Returns ``{tool_turns, multi_tool_turns, batch_rate, by_model}``
    where by_model maps each model to the same three numbers. batch_rate is
    multi_tool_turns / tool_turns (0.0 when there are no tool-turns).
    Test: ``test_batch_stats_*`` in test_turn_batches.py.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if since is not None:
        clauses.append("ts >= ?")
        params.append(int(since))
    if model:
        clauses.append("model = ?")
        params.append(model)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT model, COUNT(*) AS turns, "
            "SUM(CASE WHEN tool_call_count > 1 THEN 1 ELSE 0 END) AS multi "
            "FROM turn_batches" + where + " GROUP BY model",
            tuple(params),
        ).fetchall()
    finally:
        conn.close()

    tool_turns = 0
    multi_tool_turns = 0
    by_model: dict[Optional[str], dict[str, Any]] = {}
    for r in rows:
        turns = int(r["turns"] or 0)
        multi = int(r["multi"] or 0)
        tool_turns += turns
        multi_tool_turns += multi
        by_model[r["model"]] = {
            "tool_turns": turns,
            "multi_tool_turns": multi,
            "batch_rate": (multi / turns) if turns else 0.0,
        }
    return {
        "tool_turns": tool_turns,
        "multi_tool_turns": multi_tool_turns,
        "batch_rate": (multi_tool_turns / tool_turns) if tool_turns else 0.0,
        "by_model": by_model,
    }


def record_run_turn(session_id: str, tool_call_count: int) -> None:
    """Fold one turn's batch signal into its running ``runs`` row.

    Why: ``hermes mpm runs`` should show, per subagent, whether that run ever
    batched tool calls — without a second query. We correlate by
    ``run_id == session_id`` (subagent_start stores the child's session_id as
    run_id, and the per-turn post_api_request hook carries that same session_id
    for the child's turns). The MAIN agent's turns carry the parent session_id,
    which matches no run row — a correct no-op for the PM.
    What: For the RUNNING run whose run_id == session_id, set turn_count =
    COALESCE(turn_count,0)+1 and max_batch_size = MAX(COALESCE(max_batch_size,0),
    tool_call_count). Caller invokes this only for turns with count >= 1, so
    turn_count counts TOOL-EMITTING turns. No-op when no running run matches.
    Test: ``test_record_run_turn_updates_running_run``,
    ``test_record_run_turn_max_logic_does_not_lower``,
    ``test_record_run_turn_noop_when_no_matching_running_run``.
    """
    _write(
        "UPDATE runs "
        "SET turn_count = COALESCE(turn_count, 0) + 1, "
        "    max_batch_size = MAX(COALESCE(max_batch_size, 0), ?) "
        "WHERE run_id = ? AND status = ?",
        (int(tool_call_count), session_id, STATUS_RUNNING),
    )


def purge_old_turn_batches(retention_days: int, now: Optional[int] = None) -> int:
    """Delete turn-batch rows older than ``retention_days``. Returns rows deleted.

    Why: Keeps the turn_batches table bounded without an external cron — called
    from the same startup maintenance sweep as ``purge_old``. Turn rows are pure
    telemetry (no in-flight state), so any row past the cutoff is safe to drop.
    What: DELETEs rows with ts < (now - retention). A non-positive retention is a
    no-op (retention disabled), mirroring ``purge_old``.
    Test: ``test_purge_old_turn_batches_deletes_only_old`` +
    ``test_purge_old_turn_batches_disabled_when_non_positive``.
    """
    if retention_days <= 0:
        return 0
    if now is None:
        now = int(time.time())
    cutoff = int(now) - retention_days * 86400
    return _write("DELETE FROM turn_batches WHERE ts < ?", (cutoff,))

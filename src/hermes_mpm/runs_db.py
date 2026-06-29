"""Durable SQLite run-tracking + observability for hermes-mpm subagent runs.

Why: Delegations (sync + async/background) are fire-and-forget from the PM's
view — when the gateway restarts mid-run, or an async child finishes while the
parent has moved on, there is no durable record of what ran, how long, and how
it ended. This module is the spine of that observability: every subagent start
and stop is written to ``<hermes_home>/mpm_runs.db`` so ``hermes mpm runs`` can
answer "what ran / is running / crashed" across restarts. The durability fill is
``sweep_orphaned`` at startup: any run left ``running`` by a prior process is
marked ``crashed`` instead of lingering forever.

What: A self-contained SQLite layer (no hermes_cli internals imported — the
connection pattern is *replicated* from kanban_db/hermes_state to avoid version
coupling) exposing ``init_db``, ``record_start``, ``record_end``,
``sweep_orphaned``, ``query_runs``, and ``purge_old``. DDL runs only in
``init_db`` (never on a hot path — concurrent DDL is what corrupted the kanban
DB historically). Writes use ``BEGIN IMMEDIATE`` with app-level
retry-with-jitter on transient "database is locked".

Test: ``pytest src/hermes_mpm/tests/test_runs_db.py`` — covers create/close,
orphan-sweep, filtered query, purge, idempotent re-init, and concurrent writers.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("hermes_mpm.runs_db")

DB_FILENAME = "mpm_runs.db"

# Lock-wait tuning. busy_timeout lets SQLite serialize writers in C; the
# app-level retry below is a second belt for the rare case the C-level wait
# still surfaces "database is locked" under a write burst.
_BUSY_TIMEOUT_MS = 5000
_MAX_WRITE_RETRIES = 12
_RETRY_MIN_MS = 20
_RETRY_MAX_MS = 150

# Filesystems where ``PRAGMA journal_mode=WAL`` fails (NFS/SMB/some FUSE). On
# these we fall back to DELETE. Markers mirror hermes_state._WAL_INCOMPAT_MARKERS.
_WAL_INCOMPAT_MARKERS = ("locking protocol", "disk i/o error", "not supported")

# Run statuses we write. Vocabulary mirrors kanban.task_runs.
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CRASHED = "crashed"
STATUS_TIMED_OUT = "timed_out"

_ORPHAN_ERROR = "orphaned by restart"

# DDL — created once in init_db(). Epoch INTEGER timestamps throughout.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS subagent_runs (
    run_id            TEXT PRIMARY KEY,
    parent_session_id TEXT,
    role              TEXT,
    profile           TEXT,
    goal              TEXT,
    status            TEXT NOT NULL,
    started_at        INTEGER NOT NULL,
    ended_at          INTEGER,
    duration_ms       INTEGER,
    summary           TEXT,
    error             TEXT,
    delegation_id     TEXT,
    run_type          TEXT,
    metadata          TEXT,
    owner_pid         INTEGER,
    max_batch_size    INTEGER,
    turn_count        INTEGER
);
CREATE INDEX IF NOT EXISTS idx_runs_status  ON subagent_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_parent  ON subagent_runs(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON subagent_runs(started_at);

CREATE TABLE IF NOT EXISTS turn_batches (
    api_request_id  TEXT PRIMARY KEY,
    turn_id         TEXT,
    session_id      TEXT,
    model           TEXT,
    tool_call_count INTEGER NOT NULL,
    ts              INTEGER NOT NULL
);
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


def _apply_wal_with_fallback(conn: sqlite3.Connection) -> None:
    """Set journal_mode=WAL, falling back to DELETE on WAL-incompatible FS.

    Why: WAL gives concurrent readers + one writer (what the hooks need), but it
    is unsupported on NFS/SMB where it raises OperationalError; DELETE works
    everywhere. Replicated locally to avoid importing hermes_state.
    What: Tries WAL; on a recognized incompat marker switches to DELETE; an
    unrelated OperationalError is re-raised (not silently swallowed).
    Test: Implicit — on the test/local ext4 FS WAL succeeds; the DELETE branch is
    covered by the marker check.
    """
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        if row and (row[0] or "").lower() == "wal":
            return
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError as exc:
        if not any(m in str(exc).lower() for m in _WAL_INCOMPAT_MARKERS):
            raise
        logger.warning("mpm-runs: WAL unsupported on this filesystem (%s) — using DELETE", exc)
        conn.execute("PRAGMA journal_mode=DELETE")


def _connect() -> sqlite3.Connection:
    """Open a run-DB connection with the standard lock-waiting PRAGMAs.

    Why: Every hook handler and the CLI need a consistently-configured
    connection (autocommit + busy wait) so explicit ``BEGIN IMMEDIATE`` controls
    write transactions and lock contention degrades to waiting, not errors.
    What: Opens with check_same_thread=False, isolation_level=None (autocommit),
    timeout/busy_timeout=5000ms, WAL (or DELETE fallback), Row factory.
    Test: Indirectly via every test; concurrency test stresses the lock waiting.
    """
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
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

    Why: Under a delegation burst many threads write at once; ``BEGIN IMMEDIATE``
    takes the write lock up front (fail fast instead of mid-statement), and the
    jittered retry absorbs the rare "database is locked" that slips past the
    busy_timeout — so a tracking write never raises into a hook.
    What: Opens a connection, retries the txn up to _MAX_WRITE_RETRIES times on
    OperationalError "database is locked" with 20–150ms jitter; returns rowcount.
    Test: ``test_concurrent_start_end_no_corruption`` proves 8 threads × 25 runs
    all land with no corruption / lost rows.
    """
    last_exc: Optional[Exception] = None
    for _attempt in range(_MAX_WRITE_RETRIES):
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(sql, params)
            conn.execute("COMMIT")
            return cur.rowcount
        except sqlite3.OperationalError as exc:
            last_exc = exc
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            if "database is locked" not in str(exc).lower():
                raise
            time.sleep(random.uniform(_RETRY_MIN_MS, _RETRY_MAX_MS) / 1000.0)
        finally:
            conn.close()
    # Exhausted retries — surface so callers can log+swallow (hooks do).
    raise last_exc if last_exc else sqlite3.OperationalError("write failed")


def init_db() -> None:
    """Create the schema + indexes once. Safe to call repeatedly (idempotent).

    Why: DDL must run exactly at startup, never on a hot write path — concurrent
    CREATE/ALTER is what historically corrupted the kanban DB. Centralizing it
    here keeps record_start/record_end DDL-free. The ``owner_pid`` ALTER migrates
    DBs created before process-ownership existed so the orphan sweep can scope to
    its own runs (the data-corruption fix). The ``max_batch_size``/``turn_count``
    ALTERs migrate DBs created before batch telemetry so each subagent run can
    surface whether it batched internally.
    What: Opens a connection, executescript()s the IF NOT EXISTS schema (fresh DBs
    get owner_pid + batch columns + turn_batches in CREATE), then ALTER-adds any
    missing columns if an EXISTING table lacks them — each guarded by a PRAGMA
    column check so re-runs are a clean no-op.
    Test: ``test_init_db_is_idempotent`` + ``test_init_db_adds_owner_pid_to_existing_db``
    + ``test_init_db_migrates_legacy_subagent_runs``.
    """
    conn = _connect()
    try:
        conn.executescript(_SCHEMA_SQL)
        _ensure_owner_pid_column(conn)
        _ensure_batch_columns(conn)
    finally:
        conn.close()


def _ensure_owner_pid_column(conn: sqlite3.Connection) -> None:
    """Idempotently add the owner_pid column to an existing subagent_runs table.

    Why: A DB created before the process-ownership fix has no owner_pid column;
    the orphan sweep needs it to distinguish its own in-flight runs from a dead
    process's. CREATE TABLE IF NOT EXISTS won't add a column to an existing table,
    so we migrate with an explicit guarded ALTER.
    What: Reads PRAGMA table_info; if owner_pid is absent, ALTER TABLE ADD COLUMN
    owner_pid INTEGER. No-op when the column already exists (fresh CREATE path or
    a prior migration). Runs only inside init_db (never on a hot write path).
    Test: ``test_init_db_adds_owner_pid_to_existing_db`` (legacy DB + double init).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(subagent_runs)")}
    if "owner_pid" not in cols:
        conn.execute("ALTER TABLE subagent_runs ADD COLUMN owner_pid INTEGER")


def _ensure_batch_columns(conn: sqlite3.Connection) -> None:
    """Idempotently add max_batch_size/turn_count to an existing subagent_runs.

    Why: A DB created before batch telemetry has neither column; record_run_turn
    needs them to record per-subagent batch behaviour (did this run ever emit a
    parallel tool batch, and how many turns did it take). CREATE TABLE IF NOT
    EXISTS won't add columns to an existing table, so we migrate with guarded
    ALTERs — mirroring the owner_pid migration.
    What: Reads PRAGMA table_info; ALTER-adds each of max_batch_size / turn_count
    that is absent. No-op when both already exist. Runs only inside init_db.
    Test: ``test_init_db_migrates_legacy_subagent_runs`` (legacy DB + double init).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(subagent_runs)")}
    if "max_batch_size" not in cols:
        conn.execute("ALTER TABLE subagent_runs ADD COLUMN max_batch_size INTEGER")
    if "turn_count" not in cols:
        conn.execute("ALTER TABLE subagent_runs ADD COLUMN turn_count INTEGER")


def record_start(
    run_id: str,
    parent_session_id: Optional[str],
    role: Optional[str],
    profile: Optional[str],
    goal: Optional[str],
    started_at: int,
    run_type: Optional[str],
    delegation_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Record a run as ``running`` (INSERT OR IGNORE — first write wins).

    Why: The subagent_start hook may fire more than once for a logically-single
    run (retries, re-entry); OR IGNORE keeps the original start authoritative and
    makes the hook safe to call repeatedly. ``owner_pid`` is stamped here so the
    orphan sweep can reap only PRIOR-process runs, never the current process's own
    in-flight rows (the data-corruption fix).
    What: Inserts one ``running`` row keyed by run_id (= child_session_id),
    stamped with owner_pid = os.getpid().
    Test: ``test_record_start_creates_running_row`` +
    ``test_record_start_insert_or_ignore_is_idempotent`` +
    ``test_record_start_stamps_owner_pid``.
    """
    meta_json = json.dumps(metadata) if metadata else None
    _write(
        """
        INSERT OR IGNORE INTO subagent_runs (
            run_id, parent_session_id, role, profile, goal, status,
            started_at, delegation_id, run_type, metadata, owner_pid
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            parent_session_id,
            role,
            profile,
            goal,
            STATUS_RUNNING,
            int(started_at),
            delegation_id,
            run_type,
            meta_json,
            os.getpid(),
        ),
    )


def record_end(
    run_id: str,
    status: str,
    ended_at: int,
    duration_ms: Optional[int] = None,
    summary: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Close a run: set its terminal status + end metadata.

    Why: The subagent_stop hook (and the async-complete fallback) must mark a run
    finished so it stops counting as in-flight and the orphan sweep leaves it
    alone on the next restart.
    What: UPDATEs the row's status/ended_at/duration_ms/summary/error by run_id.
    No-op if the run_id is unknown (defensive — a stop without a start).
    Test: ``test_record_end_closes_row`` + ``test_record_end_records_error``.
    """
    _write(
        """
        UPDATE subagent_runs
           SET status = ?, ended_at = ?, duration_ms = ?, summary = ?, error = ?
         WHERE run_id = ?
        """,
        (status, int(ended_at), duration_ms, summary, error, run_id),
    )


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check for a pid via os.kill(pid, 0).

    Why: A defensive guard against the rare same-pid edge case (a recycled pid).
    If a row's owner pid is still alive, the owner is genuinely running and its
    run must NOT be reaped — even if it isn't the current process.
    What: Returns True if os.kill(pid, 0) does not raise ProcessLookupError; a
    PermissionError means the pid exists but is owned by another user → alive.
    Any other OSError → treat as not-alive (fail toward reaping a stuck row).
    Test: Exercised indirectly via the ownership sweep tests; os.kill(getpid(),0)
    confirms our own process is reported alive.
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def sweep_orphaned(now: int, current_pid: Optional[int] = None) -> int:
    """Mark PRIOR-process ``running`` rows ``crashed`` (restart durability fill).

    Why: A process that dies mid-run can never fire subagent_stop, so its run
    would linger ``running`` forever. The original sweep reaped EVERY running row,
    which corrupted data: any process loading the plugin (the CLI, the dashboard)
    swept the live gateway's in-flight runs to ``crashed`` and made async runs
    permanently un-closable. The fix scopes the sweep to runs the CURRENT process
    does NOT own — a freshly-started gateway owns no prior runs, so this reaps the
    dead process's runs while never touching anything the current process created.
    What: UPDATEs running rows whose owner_pid IS NULL (legacy/pre-migration) OR
    differs from current_pid (default os.getpid()) AND whose owner is not still
    alive (best-effort os.kill check), setting status='crashed',
    error='orphaned by restart', ended_at=now. Returns the count reaped.
    Test: ``test_sweep_orphaned_marks_running_crashed`` +
    ``test_sweep_orphaned_only_reaps_other_pid_rows``.
    """
    if current_pid is None:
        current_pid = os.getpid()

    conn = _connect()
    try:
        candidates = conn.execute(
            """
            SELECT run_id, owner_pid FROM subagent_runs
             WHERE status = ? AND ended_at IS NULL
               AND (owner_pid IS NULL OR owner_pid != ?)
            """,
            (STATUS_RUNNING, current_pid),
        ).fetchall()
    finally:
        conn.close()

    # Defensive: never reap a row whose (other) owner pid is still alive — that
    # owner is genuinely running; only NULL-owner or dead-owner rows are orphans.
    reap_ids = [
        r["run_id"]
        for r in candidates
        if r["owner_pid"] is None or not _pid_alive(int(r["owner_pid"]))
    ]
    count = 0
    for run_id in reap_ids:
        count += _write(
            """
            UPDATE subagent_runs
               SET status = ?, error = ?, ended_at = ?
             WHERE run_id = ? AND status = ? AND ended_at IS NULL
            """,
            (STATUS_CRASHED, _ORPHAN_ERROR, int(now), run_id, STATUS_RUNNING),
        )
    return count


def query_runs(
    status: Optional[str] = None,
    session: Optional[str] = None,
    since: Optional[int] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return runs matching the filters, newest first.

    Why: Backs ``hermes mpm runs`` — a no-LLM query of run history/state.
    What: SELECT with optional status / parent_session_id / started_at>=since
    filters, ORDER BY started_at DESC, LIMIT. Returns plain dicts.
    Test: ``test_query_runs_filters_and_orders``.
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

    conn = _connect()
    try:
        return [dict(r) for r in conn.execute(sql, tuple(params))]
    finally:
        conn.close()


def find_running_by_delegation(delegation_id: str) -> Optional[str]:
    """Return the run_id of a running row carrying ``delegation_id``, if any.

    Why: The async-complete fallback needs to close the right run by its
    delegation_id when one was recorded at start.
    What: SELECTs the newest running run_id whose delegation_id matches.
    Test: Exercised via the hook-fallback test in test_init_register.py.
    """
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT run_id FROM subagent_runs
             WHERE delegation_id = ? AND status = ?
             ORDER BY started_at DESC LIMIT 1
            """,
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
    lets the async-complete marker close the run by EXACT delegation_id instead
    of fragile goal-text matching. The ``delegation_id IS NULL`` filter is the
    disambiguator for two sequential identical-goal async runs: the second stamp
    cannot overwrite the first (it falls through to the next still-null row).
    What: Finds the newest ``running`` row with this goal AND delegation_id NULL
    (SELECT by run_id, then UPDATE by run_id — portable across SQLite builds that
    lack UPDATE…ORDER BY/LIMIT), sets its delegation_id. Returns True iff a row
    was stamped. ``now`` is accepted for signature symmetry with other writers
    (unused here — no timestamp is mutated).
    Test: ``test_stamp_delegation_id_*`` in test_runs_db.py.
    """
    del now  # accepted for symmetry; this write mutates no timestamp
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT run_id FROM subagent_runs
             WHERE goal = ? AND status = ? AND delegation_id IS NULL
             ORDER BY started_at DESC LIMIT 1
            """,
            (goal, STATUS_RUNNING),
        ).fetchone()
        run_id = row["run_id"] if row else None
    finally:
        conn.close()
    if run_id is None:
        return False
    # Re-assert the NULL guard in the UPDATE so a concurrent stamp can't double-write.
    updated = _write(
        """
        UPDATE subagent_runs SET delegation_id = ?
         WHERE run_id = ? AND delegation_id IS NULL
        """,
        (delegation_id, run_id),
    )
    return updated > 0


def find_running_by_goal(goal: str) -> Optional[str]:
    """Return the run_id of the SOLE running row whose goal matches, else None.

    Why: This is now only a LAST-RESORT fallback — delegation_id correlation
    (stamp_delegation_id + find_running_by_delegation) is the primary path. Goal
    text at completion can be truncated/reformatted, and two async runs can share
    a goal, so this must never GUESS: if more than one running row matches, return
    None and leave the ambiguous pair for the honest crash-sweep rather than
    closing the wrong row.
    What: Counts running rows with an exact goal match; returns the run_id only
    when EXACTLY one matches (ambiguity guard), else None.
    Test: ``test_find_running_by_goal_returns_none_when_ambiguous`` +
    ``test_find_running_by_goal_returns_single_match``.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT run_id FROM subagent_runs
             WHERE goal = ? AND status = ?
             ORDER BY started_at DESC
            """,
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
        "DELETE FROM subagent_runs WHERE ended_at IS NOT NULL AND ended_at < ?",
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

    Why: This is the global, queryable parallelism signal — one row per turn that
    emitted >=1 tool call, for BOTH the main agent and every subagent. >1 means
    the assistant batched tool calls in a single turn (parallelism working); 1
    means a single call. Recording it durably lets ``batch_stats`` compute the
    batch rate by SELECT/GROUP BY instead of a fragile ad-hoc classifier. Callers
    record only turns with ``tool_call_count >= 1`` (a 0-tool turn carries no batch
    signal); this writes whatever it is given.
    What: INSERT OR IGNORE one row keyed by api_request_id (first write wins — the
    post_api_request hook may fire more than once for a logically-single request),
    using the same write-path/retry pattern as the rest of the module. Returns the
    rowcount: 1 when the row was newly inserted, 0 on a duplicate (OR IGNORE) — so
    the caller can gate the per-run fold (record_run_turn) on a NEW turn and avoid
    double-counting turn_count on a duplicate fire.
    Test: ``test_record_turn_batch_inserts_row`` +
    ``test_record_turn_batch_idempotent_on_api_request_id`` +
    ``test_record_turn_batch_returns_rowcount``.
    """
    return _write(
        """
        INSERT OR IGNORE INTO turn_batches (
            api_request_id, turn_id, session_id, model, tool_call_count, ts
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (api_request_id, turn_id, session_id, model, int(tool_call_count), int(ts)),
    )


def batch_stats(
    since: Optional[int] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregate the turn-batch store into a parallelism scorecard.

    Why: Backs ``hermes mpm parallelism`` — turns the raw per-turn rows into the
    one number operators ask for: the batch rate (fraction of tool-turns that
    batched >1 call), overall and per model. No LLM, no ad-hoc classifier.
    What: SELECT/GROUP BY over turn_batches with optional ts>=since and model=
    filters. Returns ``{tool_turns, multi_tool_turns, batch_rate, by_model}`` where
    by_model maps each model to the same three numbers. batch_rate is
    multi_tool_turns / tool_turns (0.0 when there are no tool-turns).
    Test: ``test_batch_stats_computes_rate`` (3,1,2 -> rate 0.667),
    ``test_batch_stats_empty_is_zero_rate``, ``test_batch_stats_per_model_breakdown``,
    ``test_batch_stats_since_filter``, ``test_batch_stats_model_filter``.
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
    # Keys are the stored model id, which is nullable (the hook records
    # model=kw.get("model")), so a None key is possible; callers must be None-safe.
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
    """Fold one turn's batch signal into its running subagent_run row.

    Why: ``hermes mpm runs`` should show, per subagent, whether that run ever
    batched tool calls — without a second query. We correlate by
    ``run_id == session_id``: the subagent_start hook stores the
    child's ``session_id`` as ``run_id`` (verified in delegate_tool.py), and the
    per-turn post_api_request hook carries that same ``session_id`` for the child's
    turns. The MAIN agent's turns carry the parent session_id, which matches no run
    row — so this is a correct no-op for the PM (the global turn_batches store
    still captures it).
    What: For the RUNNING run whose run_id == session_id, set turn_count =
    COALESCE(turn_count,0)+1 and max_batch_size = MAX(COALESCE(max_batch_size,0),
    tool_call_count). NOTE: the caller (the post_api_request hook) only invokes
    this for turns with tool_call_count >= 1, so turn_count counts TOOL-EMITTING
    turns, not total turns. No-op when no running run matches (unknown id, or
    already ended — an ended run is left frozen). Uses the standard
    write-path/retry.
    Test: ``test_record_run_turn_updates_running_run``,
    ``test_record_run_turn_max_logic_does_not_lower``,
    ``test_record_run_turn_noop_when_no_matching_running_run``.
    """
    _write(
        """
        UPDATE subagent_runs
           SET turn_count = COALESCE(turn_count, 0) + 1,
               max_batch_size = MAX(COALESCE(max_batch_size, 0), ?)
         WHERE run_id = ? AND status = ?
        """,
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

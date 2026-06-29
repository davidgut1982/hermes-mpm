"""Tests for the durable subagent run-tracking DB (``hermes_mpm.runs_db``).

Why: The run-tracking layer is the observability spine — if its write/close/sweep
semantics are wrong, runs are silently lost or never marked crashed. These tests
pin the exact behavior (insert-once, close, orphan-sweep, filtered query, purge,
idempotent init, concurrency-safe) so a regression is caught before it ships.
What: Each test drives one ``runs_db`` function against a throwaway DB file under
``tmp_path`` (via the ``db`` fixture that points ``DB_PATH`` at tmp) and asserts
the observable row state.
Test: Run ``pytest src/hermes_mpm/tests/test_runs_db.py`` — all must pass.
"""

from __future__ import annotations

import os
import sqlite3
import threading

import pytest

from hermes_mpm import runs_db


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Point runs_db at an isolated tmp DB and init it once.

    Why: Each test needs a clean DB with no cross-test bleed and no dependence on
    a real Hermes home.
    What: Monkeypatches ``_db_path`` to return ``tmp_path/mpm_runs.db``, runs
    init_db(), and yields the path.
    Test: Implicit — every test using this fixture starts from an empty schema.
    """
    path = tmp_path / "mpm_runs.db"
    monkeypatch.setattr(runs_db, "_db_path", lambda: path)
    runs_db.init_db()
    yield path


def _rows(path):
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM subagent_runs")]
    finally:
        conn.close()


def test_record_start_creates_running_row(db):
    runs_db.record_start(
        run_id="child-1",
        parent_session_id="parent-1",
        role="engineer",
        profile="engineer",
        goal="do the thing",
        started_at=1000,
        run_type="subagent",
    )
    rows = _rows(db)
    assert len(rows) == 1
    r = rows[0]
    assert r["run_id"] == "child-1"
    assert r["status"] == "running"
    assert r["parent_session_id"] == "parent-1"
    assert r["role"] == "engineer"
    assert r["started_at"] == 1000
    assert r["ended_at"] is None
    assert r["run_type"] == "subagent"


def test_record_start_insert_or_ignore_is_idempotent(db):
    runs_db.record_start("c", "p", "r", None, "g", 1, "subagent")
    runs_db.record_start("c", "p", "r", None, "g2", 99, "subagent")
    rows = _rows(db)
    assert len(rows) == 1
    # First write wins (INSERT OR IGNORE) — goal stays "g".
    assert rows[0]["goal"] == "g"


def test_record_end_closes_row(db):
    runs_db.record_start("c", "p", "r", None, "g", 1000, "subagent")
    runs_db.record_end("c", status="done", ended_at=1005, duration_ms=5000, summary="ok")
    r = _rows(db)[0]
    assert r["status"] == "done"
    assert r["ended_at"] == 1005
    assert r["duration_ms"] == 5000
    assert r["summary"] == "ok"


def test_record_end_records_error(db):
    runs_db.record_start("c", "p", "r", None, "g", 1, "subagent")
    runs_db.record_end("c", status="failed", ended_at=2, error="boom")
    r = _rows(db)[0]
    assert r["status"] == "failed"
    assert r["error"] == "boom"


def test_sweep_orphaned_marks_running_crashed(db, monkeypatch):
    # Rows "a"/"b" are stamped with THIS process's pid by record_start. To
    # exercise the "prior dead process" path the sweep must run with a DIFFERENT
    # current_pid than the one that owns the rows, and that owner must be dead.
    monkeypatch.setattr(runs_db, "_pid_alive", lambda pid: False)
    runs_db.record_start("a", "p", "r", None, "g", 1, "subagent")
    runs_db.record_start("b", "p", "r", None, "g", 1, "subagent")
    runs_db.record_end("b", status="done", ended_at=2)
    other_pid = os.getpid() + 1
    n = runs_db.sweep_orphaned(now=500, current_pid=other_pid)
    assert n == 1
    rows = {r["run_id"]: r for r in _rows(db)}
    assert rows["a"]["status"] == "crashed"
    assert rows["a"]["error"] == "orphaned by restart"
    assert rows["a"]["ended_at"] == 500
    assert rows["b"]["status"] == "done"  # untouched (already ended)


def test_query_runs_filters_and_orders(db):
    runs_db.record_start("a", "p1", "r", None, "ga", 100, "subagent")
    runs_db.record_start("b", "p2", "r", None, "gb", 200, "subagent")
    runs_db.record_start("c", "p1", "r", None, "gc", 300, "subagent")
    runs_db.record_end("a", status="done", ended_at=150)

    # Newest first by started_at.
    all_runs = runs_db.query_runs()
    assert [r["run_id"] for r in all_runs] == ["c", "b", "a"]

    # Filter by status.
    running = runs_db.query_runs(status="running")
    assert {r["run_id"] for r in running} == {"b", "c"}

    # Filter by session.
    p1 = runs_db.query_runs(session="p1")
    assert {r["run_id"] for r in p1} == {"a", "c"}

    # since (epoch) — only runs started at/after the cutoff.
    recent = runs_db.query_runs(since=250)
    assert {r["run_id"] for r in recent} == {"c"}

    # limit.
    limited = runs_db.query_runs(limit=1)
    assert len(limited) == 1 and limited[0]["run_id"] == "c"


def test_purge_old_deletes_only_old_ended_rows(db):
    # Old + ended -> purged.
    runs_db.record_start("old", "p", "r", None, "g", 1, "subagent")
    runs_db.record_end("old", status="done", ended_at=10)
    # Recent + ended -> kept.
    runs_db.record_start("recent", "p", "r", None, "g", 1, "subagent")
    runs_db.record_end("recent", status="done", ended_at=10_000_000_000)
    # Old but still running (ended_at NULL) -> kept (never purge in-flight).
    runs_db.record_start("running", "p", "r", None, "g", 1, "subagent")

    # retention 1 day; cutoff computed against a fixed "now".
    deleted = runs_db.purge_old(retention_days=1, now=10_000_000_100)
    assert deleted == 1
    ids = {r["run_id"] for r in _rows(db)}
    assert ids == {"recent", "running"}


def test_init_db_is_idempotent(db):
    # Calling init_db twice must not raise or wipe data.
    runs_db.record_start("c", "p", "r", None, "g", 1, "subagent")
    runs_db.init_db()
    runs_db.init_db()
    assert len(_rows(db)) == 1


def test_concurrent_start_end_no_corruption(db):
    """~8 threads each record_start+record_end concurrently → all rows present.

    Why: Proves the WAL + BEGIN IMMEDIATE + retry-with-jitter pattern survives
    concurrent writers without "database is locked" surfacing or lost rows — the
    durability guarantee the whole layer rests on.
    What: 8 threads × 25 runs each; assert 200 closed rows, all status='done'.
    Test: This test itself.
    """
    n_threads = 8
    per_thread = 25

    def worker(tid: int) -> None:
        for i in range(per_thread):
            rid = f"t{tid}-{i}"
            runs_db.record_start(rid, "p", "r", None, "g", 1, "subagent")
            runs_db.record_end(rid, status="done", ended_at=2, duration_ms=1)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = _rows(db)
    assert len(rows) == n_threads * per_thread
    assert all(r["status"] == "done" for r in rows)


def test_record_start_stamps_owner_pid(db):
    """record_start must stamp owner_pid = os.getpid() so the sweep can tell its
    own in-flight rows from a prior dead process's rows."""
    runs_db.record_start("c", "p", "r", None, "g", 1000, "subagent")
    r = _rows(db)[0]
    assert r["owner_pid"] == os.getpid()


def test_sweep_orphaned_only_reaps_other_pid_rows(db, monkeypatch):
    """A gateway sweep with a fresh current_pid marks PRIOR-pid running rows
    crashed but leaves rows owned by current_pid untouched.

    A freshly-started gateway owns no prior runs, so pid-scoping reaps the
    dead process's runs while never touching anything the current process will
    create."""
    # The prior owner pid is treated as dead so liveness can't mask the scoping.
    monkeypatch.setattr(runs_db, "_pid_alive", lambda pid: False)
    current_pid = os.getpid()
    prior_pid = current_pid + 1  # a different (dead) process

    # Row owned by a prior (dead) process — must be reaped.
    runs_db._write(
        "INSERT INTO subagent_runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("prior", runs_db.STATUS_RUNNING, 1, prior_pid),
    )
    # Row with NULL owner_pid (legacy / pre-migration) — must be reaped.
    runs_db._write(
        "INSERT INTO subagent_runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("legacy", runs_db.STATUS_RUNNING, 1, None),
    )
    # Row owned by the CURRENT process — must be left untouched.
    runs_db._write(
        "INSERT INTO subagent_runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("mine", runs_db.STATUS_RUNNING, 1, current_pid),
    )

    n = runs_db.sweep_orphaned(now=500, current_pid=current_pid)
    assert n == 2
    rows = {r["run_id"]: r for r in _rows(db)}
    assert rows["prior"]["status"] == "crashed"
    assert rows["prior"]["error"] == "orphaned by restart"
    assert rows["prior"]["ended_at"] == 500
    assert rows["legacy"]["status"] == "crashed"
    assert rows["mine"]["status"] == "running"  # current pid — left alone
    assert rows["mine"]["ended_at"] is None


def test_sweep_orphaned_skips_alive_other_owner(db, monkeypatch):
    """Defensive hardening: a running row owned by a DIFFERENT but still-ALIVE
    process must NOT be reaped (that owner is genuinely running)."""
    monkeypatch.setattr(runs_db, "_pid_alive", lambda pid: True)
    other_pid = os.getpid() + 1
    runs_db._write(
        "INSERT INTO subagent_runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("alive_other", runs_db.STATUS_RUNNING, 1, other_pid),
    )
    n = runs_db.sweep_orphaned(now=500, current_pid=os.getpid())
    assert n == 0
    assert _rows(db)[0]["status"] == "running"


def test_init_db_adds_owner_pid_to_existing_db(tmp_path, monkeypatch):
    """init_db must be idempotent on an EXISTING db that lacks owner_pid: the
    ALTER adds the column without error, and re-running init_db is a no-op."""
    path = tmp_path / "mpm_runs.db"
    monkeypatch.setattr(runs_db, "_db_path", lambda: path)

    # Build a legacy DB by hand with the ORIGINAL schema (all columns) but
    # WITHOUT owner_pid — exactly what a pre-fix DB on disk looks like.
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE subagent_runs ("
            "run_id TEXT PRIMARY KEY, parent_session_id TEXT, role TEXT, profile TEXT, "
            "goal TEXT, status TEXT NOT NULL, started_at INTEGER NOT NULL, ended_at INTEGER, "
            "duration_ms INTEGER, summary TEXT, error TEXT, delegation_id TEXT, "
            "run_type TEXT, metadata TEXT)"
        )
        conn.execute(
            "INSERT INTO subagent_runs (run_id, status, started_at) VALUES (?, ?, ?)",
            ("legacy", runs_db.STATUS_RUNNING, 1),
        )
        conn.commit()
    finally:
        conn.close()

    # First init_db must ALTER-add owner_pid without raising.
    runs_db.init_db()
    cols = {r["name"] for r in _pragma_cols(path)}
    assert "owner_pid" in cols
    # Existing legacy row preserved; its owner_pid defaults to NULL.
    rows = {r["run_id"]: r for r in _rows(path)}
    assert rows["legacy"]["owner_pid"] is None

    # Second init_db on the now-migrated DB is a clean no-op (idempotent ALTER).
    runs_db.init_db()
    assert "owner_pid" in {r["name"] for r in _pragma_cols(path)}


def _pragma_cols(path):
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("PRAGMA table_info(subagent_runs)")]
    finally:
        conn.close()


# --- delegation_id correlation: stamp + ambiguity guard -------------------


def test_stamp_delegation_id_stamps_newest_null_running_row(db):
    """stamp_delegation_id stamps the newest matching un-stamped running row.

    Why: direct-async subagent_start creates the row with delegation_id NULL;
    the synchronous post_tool_call later carries the delegation_id and must
    stamp it onto that exact row so the async-complete marker can close it by
    id. Two rows share the goal here — only the NEWEST null-delegation running
    one must be stamped.
    What: two running rows same goal (older started_at, newer started_at), both
    delegation_id NULL → stamp targets the newer; returns True.
    Test: this test.
    """
    runs_db.record_start("old", "p", "r", None, "shared goal", 100, "subagent")
    runs_db.record_start("new", "p", "r", None, "shared goal", 200, "subagent")

    stamped = runs_db.stamp_delegation_id("shared goal", "deleg_aaaa1111")
    assert stamped is True

    rows = {r["run_id"]: r for r in _rows(db)}
    assert rows["new"]["delegation_id"] == "deleg_aaaa1111"
    assert rows["old"]["delegation_id"] is None  # untouched


def test_stamp_delegation_id_skips_already_stamped_rows(db):
    """The ``delegation_id IS NULL`` filter prevents cross-stamping a row that
    already carries a delegation_id — the disambiguator for sequential
    identical-goal async runs.

    Why: two sequential identical-goal async dispatches each get their own
    post_tool_call; the second must NOT overwrite the first's delegation_id.
    What: stamp the newest row, then stamp again for the same goal → the second
    stamp lands on the OLDER still-null row (not the already-stamped newest).
    Test: this test.
    """
    runs_db.record_start("first", "p", "r", None, "g", 100, "subagent")
    runs_db.record_start("second", "p", "r", None, "g", 200, "subagent")

    # First dispatch's post_tool_call stamps the newest (second) row.
    assert runs_db.stamp_delegation_id("g", "deleg_second") is True
    # Second dispatch's post_tool_call must fall through to the older null row.
    assert runs_db.stamp_delegation_id("g", "deleg_first") is True

    rows = {r["run_id"]: r for r in _rows(db)}
    assert rows["second"]["delegation_id"] == "deleg_second"
    assert rows["first"]["delegation_id"] == "deleg_first"


def test_stamp_delegation_id_returns_false_when_no_match(db):
    """No running null-delegation row for the goal → returns False, no write."""
    # A row that does NOT match (different goal).
    runs_db.record_start("x", "p", "r", None, "other", 1, "subagent")
    assert runs_db.stamp_delegation_id("nonexistent goal", "deleg_zzz") is False
    # A running row whose goal matches but is ALREADY stamped — still no match.
    runs_db.record_start("y", "p", "r", None, "taken", 1, "subagent", delegation_id="d0")
    assert runs_db.stamp_delegation_id("taken", "deleg_new") is False


def test_stamp_delegation_id_ignores_ended_rows(db):
    """Only RUNNING rows are eligible — an ended row of the same goal is skipped."""
    runs_db.record_start("done", "p", "r", None, "g", 1, "subagent")
    runs_db.record_end("done", status="done", ended_at=2)
    assert runs_db.stamp_delegation_id("g", "deleg_x") is False
    assert _rows(db)[0]["delegation_id"] is None


def test_find_running_by_goal_returns_none_when_ambiguous(db):
    """AMBIGUITY GUARD: >1 running row matches the goal → return None.

    Why: goal-match is now only a last-resort fallback. If two async runs share
    a goal and neither was stamped, we must NOT guess which to close — closing
    the wrong one corrupts data. Returning None leaves the ambiguous pair for
    the honest crash-sweep instead.
    What: two running rows same goal → find_running_by_goal returns None.
    Test: this test.
    """
    runs_db.record_start("a", "p", "r", None, "dup goal", 100, "subagent")
    runs_db.record_start("b", "p", "r", None, "dup goal", 200, "subagent")
    assert runs_db.find_running_by_goal("dup goal") is None


def test_find_running_by_goal_returns_single_match(db):
    """The guard does not break the unambiguous case: exactly one running row
    matching the goal is still returned."""
    runs_db.record_start("only", "p", "r", None, "unique goal", 1, "subagent")
    # An ended row with the same goal must not count toward ambiguity.
    runs_db.record_start("ended", "p", "r", None, "unique goal", 2, "subagent")
    runs_db.record_end("ended", status="done", ended_at=3)
    assert runs_db.find_running_by_goal("unique goal") == "only"

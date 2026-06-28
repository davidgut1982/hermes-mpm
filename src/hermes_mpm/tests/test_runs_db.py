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
    runs_db.record_end(
        "c", status="done", ended_at=1005, duration_ms=5000, summary="ok"
    )
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


def test_sweep_orphaned_marks_running_crashed(db):
    runs_db.record_start("a", "p", "r", None, "g", 1, "subagent")
    runs_db.record_start("b", "p", "r", None, "g", 1, "subagent")
    runs_db.record_end("b", status="done", ended_at=2)
    n = runs_db.sweep_orphaned(now=500)
    assert n == 1
    rows = {r["run_id"]: r for r in _rows(db)}
    assert rows["a"]["status"] == "crashed"
    assert rows["a"]["error"] == "orphaned by restart"
    assert rows["a"]["ended_at"] == 500
    assert rows["b"]["status"] == "done"  # untouched


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

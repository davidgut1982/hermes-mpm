"""Tests for the per-turn batch-telemetry store in ``hermes_mpm.runs_db``.

Why: "Is parallelism working?" must become a trustworthy, queryable number, not a
guess. This layer records the assistant's tool-call count per LLM turn
(>1 = batched/parallel, 1 = single, 0 = no tools — not recorded) so the batch
rate can be computed by SELECT/GROUP BY instead of a buggy ad-hoc classifier.
These tests pin the insert/aggregate/migrate/purge semantics so a regression is
caught before it ships.
What: Each test drives one ``runs_db`` turn-batch function against a throwaway DB
under ``tmp_path`` (via the ``db`` fixture that points ``_db_path`` at tmp) and
asserts the observable row/aggregate state.
Test: Run ``pytest src/hermes_mpm/tests/test_turn_batches.py`` — all must pass.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_mpm import runs_db


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Point runs_db at an isolated tmp DB and init it once."""
    path = tmp_path / "mpm_runs.db"
    monkeypatch.setattr(runs_db, "_db_path", lambda: path)
    runs_db.init_db()
    yield path


def _turn_rows(path):
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM turn_batches")]
    finally:
        conn.close()


def _run_rows(path):
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        return {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM subagent_runs")}
    finally:
        conn.close()


# --- record_turn_batch -----------------------------------------------------


def test_record_turn_batch_inserts_row(db):
    runs_db.record_turn_batch(
        turn_id="t1",
        api_request_id="api-1",
        session_id="s1",
        model="glm-4.6",
        tool_call_count=3,
        ts=1000,
    )
    rows = _turn_rows(db)
    assert len(rows) == 1
    r = rows[0]
    assert r["api_request_id"] == "api-1"
    assert r["turn_id"] == "t1"
    assert r["session_id"] == "s1"
    assert r["model"] == "glm-4.6"
    assert r["tool_call_count"] == 3
    assert r["ts"] == 1000


def test_record_turn_batch_idempotent_on_api_request_id(db):
    """Same api_request_id must not double-insert (PK / OR IGNORE — first wins)."""
    runs_db.record_turn_batch("t1", "api-1", "s1", "m", 2, 1000)
    runs_db.record_turn_batch("t1", "api-1", "s1", "m", 9, 2000)
    rows = _turn_rows(db)
    assert len(rows) == 1
    assert rows[0]["tool_call_count"] == 2  # first write wins


def test_record_turn_batch_returns_rowcount(db):
    """Returns 1 when the row is newly inserted, 0 on a duplicate (OR IGNORE) —
    so the caller can gate the per-run fold on the turn being newly recorded."""
    assert runs_db.record_turn_batch("t1", "api-1", "s1", "m", 2, 1000) == 1
    assert runs_db.record_turn_batch("t1", "api-1", "s1", "m", 2, 1000) == 0


# --- batch_stats -----------------------------------------------------------


def test_batch_stats_computes_rate(db):
    # 3 tool-turns: counts 3, 1, 2 -> tool_turns=3, multi=2, rate=0.667.
    runs_db.record_turn_batch("t1", "a1", "s1", "m", 3, 100)
    runs_db.record_turn_batch("t2", "a2", "s1", "m", 1, 200)
    runs_db.record_turn_batch("t3", "a3", "s1", "m", 2, 300)

    stats = runs_db.batch_stats()
    assert stats["tool_turns"] == 3
    assert stats["multi_tool_turns"] == 2
    assert round(stats["batch_rate"], 3) == 0.667


def test_batch_stats_empty_is_zero_rate(db):
    stats = runs_db.batch_stats()
    assert stats["tool_turns"] == 0
    assert stats["multi_tool_turns"] == 0
    assert stats["batch_rate"] == 0.0
    assert stats["by_model"] == {}


def test_batch_stats_per_model_breakdown(db):
    runs_db.record_turn_batch("t1", "a1", "s1", "glm-4.6", 3, 100)
    runs_db.record_turn_batch("t2", "a2", "s1", "glm-4.6", 1, 200)
    runs_db.record_turn_batch("t3", "a3", "s1", "claude", 2, 300)

    stats = runs_db.batch_stats()
    by_model = stats["by_model"]
    assert by_model["glm-4.6"]["tool_turns"] == 2
    assert by_model["glm-4.6"]["multi_tool_turns"] == 1
    assert round(by_model["glm-4.6"]["batch_rate"], 3) == 0.5
    assert by_model["claude"]["tool_turns"] == 1
    assert by_model["claude"]["multi_tool_turns"] == 1
    assert by_model["claude"]["batch_rate"] == 1.0


def test_batch_stats_since_filter(db):
    runs_db.record_turn_batch("t1", "a1", "s1", "m", 2, 100)  # old
    runs_db.record_turn_batch("t2", "a2", "s1", "m", 3, 500)  # recent

    stats = runs_db.batch_stats(since=300)
    assert stats["tool_turns"] == 1
    assert stats["multi_tool_turns"] == 1
    assert stats["batch_rate"] == 1.0


def test_batch_stats_model_filter(db):
    runs_db.record_turn_batch("t1", "a1", "s1", "glm-4.6", 2, 100)
    runs_db.record_turn_batch("t2", "a2", "s1", "claude", 1, 200)

    stats = runs_db.batch_stats(model="glm-4.6")
    assert stats["tool_turns"] == 1
    assert stats["multi_tool_turns"] == 1
    # by_model still scoped to the filtered model only.
    assert set(stats["by_model"].keys()) == {"glm-4.6"}


# --- record_run_turn -------------------------------------------------------


def test_record_run_turn_updates_running_run(db):
    """A running subagent_run whose run_id == the turn's session_id gets its
    turn_count incremented and max_batch_size raised."""
    runs_db.record_start("run-s1", "p", "engineer", "engineer", "g", 1000, "subagent")
    runs_db.record_run_turn(session_id="run-s1", tool_call_count=3)
    runs_db.record_run_turn(session_id="run-s1", tool_call_count=1)

    r = _run_rows(db)["run-s1"]
    assert r["turn_count"] == 2
    assert r["max_batch_size"] == 3  # MAX(3, 1)


def test_record_run_turn_max_logic_does_not_lower(db):
    """A later SMALLER batch must not lower the recorded max_batch_size."""
    runs_db.record_start("run-s1", "p", "r", None, "g", 1000, "subagent")
    runs_db.record_run_turn("run-s1", 5)
    runs_db.record_run_turn("run-s1", 2)
    r = _run_rows(db)["run-s1"]
    assert r["max_batch_size"] == 5
    assert r["turn_count"] == 2


def test_record_run_turn_noop_when_no_matching_running_run(db):
    """No running run with run_id == session_id -> no row touched, no error."""
    # An ENDED run with that id must not be updated either.
    runs_db.record_start("run-s1", "p", "r", None, "g", 1000, "subagent")
    runs_db.record_end("run-s1", status="done", ended_at=2000)
    runs_db.record_run_turn("run-s1", 4)
    runs_db.record_run_turn("nonexistent", 4)

    r = _run_rows(db)["run-s1"]
    # Ended run untouched: counts stay at their post-init defaults.
    assert (r["turn_count"] or 0) == 0
    assert (r["max_batch_size"] or 0) == 0


# --- purge_old_turn_batches ------------------------------------------------


def test_purge_old_turn_batches_deletes_only_old(db):
    runs_db.record_turn_batch("t-old", "a-old", "s", "m", 2, 10)
    runs_db.record_turn_batch("t-new", "a-new", "s", "m", 2, 10_000_000_000)

    deleted = runs_db.purge_old_turn_batches(retention_days=1, now=10_000_000_100)
    assert deleted == 1
    rows = {r["api_request_id"] for r in _turn_rows(db)}
    assert rows == {"a-new"}


def test_purge_old_turn_batches_disabled_when_non_positive(db):
    runs_db.record_turn_batch("t", "a", "s", "m", 2, 10)
    assert runs_db.purge_old_turn_batches(retention_days=0, now=10_000) == 0
    assert len(_turn_rows(db)) == 1


# --- init_db migration -----------------------------------------------------


def test_init_db_creates_turn_batches_and_run_columns(db):
    """Fresh init_db creates turn_batches and the new subagent_runs columns."""
    cols = {r["name"] for r in _pragma_cols(db, "subagent_runs")}
    assert "max_batch_size" in cols
    assert "turn_count" in cols
    # turn_batches exists and is queryable.
    assert runs_db.batch_stats()["tool_turns"] == 0


def test_init_db_migrates_legacy_subagent_runs(tmp_path, monkeypatch):
    """init_db must ALTER-add max_batch_size/turn_count to an EXISTING table that
    lacks them (legacy DB migration), without error, idempotently."""
    path = tmp_path / "mpm_runs.db"
    monkeypatch.setattr(runs_db, "_db_path", lambda: path)

    conn = sqlite3.connect(str(path))
    try:
        # Legacy schema: has owner_pid but NOT the batch columns, and no turn_batches.
        conn.execute(
            "CREATE TABLE subagent_runs ("
            "run_id TEXT PRIMARY KEY, parent_session_id TEXT, role TEXT, profile TEXT, "
            "goal TEXT, status TEXT NOT NULL, started_at INTEGER NOT NULL, ended_at INTEGER, "
            "duration_ms INTEGER, summary TEXT, error TEXT, delegation_id TEXT, "
            "run_type TEXT, metadata TEXT, owner_pid INTEGER)"
        )
        conn.execute(
            "INSERT INTO subagent_runs (run_id, status, started_at) VALUES (?, ?, ?)",
            ("legacy", runs_db.STATUS_RUNNING, 1),
        )
        conn.commit()
    finally:
        conn.close()

    runs_db.init_db()  # must ALTER without raising
    cols = {r["name"] for r in _pragma_cols(path, "subagent_runs")}
    assert "max_batch_size" in cols
    assert "turn_count" in cols

    # record_run_turn now works on the migrated legacy row.
    runs_db.record_run_turn("legacy", 4)
    assert _run_rows(path)["legacy"]["max_batch_size"] == 4

    runs_db.init_db()  # second init is a clean no-op
    assert "turn_count" in {r["name"] for r in _pragma_cols(path, "subagent_runs")}


def _pragma_cols(path, table):
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()

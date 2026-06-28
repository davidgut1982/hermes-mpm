"""Tests for the runs-tracking hook wiring in ``hermes_mpm.register``.

Why: The hooks are the only thing that actually populates the run DB in
production; if they don't map statuses, don't close async runs, or raise into the
engine, the whole layer is dead weight or — worse — breaks delegations. These
tests pin: start→running, stop→closed-with-mapped-status, async-complete→closed,
and the hard requirement that a DB error is swallowed (never raised).
What: Drive the registered handlers directly against a tmp DB, plus a register()
smoke test asserting the subagent hooks are wired.
Test: ``pytest src/hermes_mpm/tests/test_runs_hooks.py``.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

import hermes_mpm
from hermes_mpm import runs_db


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = tmp_path / "mpm_runs.db"
    monkeypatch.setattr(runs_db, "_db_path", lambda: path)
    runs_db.init_db()
    return path


def _rows(path):
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        return {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM subagent_runs")}
    finally:
        conn.close()


def test_status_mapping_table():
    m = hermes_mpm._map_child_status
    assert m("completed") == runs_db.STATUS_DONE
    assert m("success") == runs_db.STATUS_DONE
    assert m("error") == runs_db.STATUS_FAILED
    assert m("spawn_failed") == runs_db.STATUS_FAILED
    assert m("timed_out") == runs_db.STATUS_TIMED_OUT
    assert m("crashed") == runs_db.STATUS_CRASHED
    # Unknown / None -> failed (defensive: a finished-but-unclassified run is
    # not a success).
    assert m(None) == runs_db.STATUS_FAILED
    assert m("weird") == runs_db.STATUS_FAILED


def test_subagent_start_handler_creates_running(db):
    handler = hermes_mpm._make_subagent_start_handler()
    handler(
        parent_session_id="p1",
        child_session_id="c1",
        child_role="engineer",
        child_goal="build it",
    )
    rows = _rows(db)
    assert "c1" in rows
    assert rows["c1"]["status"] == "running"
    assert rows["c1"]["role"] == "engineer"
    assert rows["c1"]["goal"] == "build it"
    assert rows["c1"]["run_type"] == "subagent"


def test_subagent_stop_handler_closes_with_mapped_status(db):
    start = hermes_mpm._make_subagent_start_handler()
    stop = hermes_mpm._make_subagent_stop_handler()
    start(child_session_id="c1", parent_session_id="p1", child_goal="g")
    stop(
        child_session_id="c1",
        child_status="completed",
        child_summary="all good",
        duration_ms=4200,
    )
    r = _rows(db)["c1"]
    assert r["status"] == "done"
    assert r["summary"] == "all good"
    assert r["duration_ms"] == 4200
    assert r["ended_at"] is not None


def test_subagent_start_handler_swallows_db_error(db, monkeypatch):
    """A DB failure inside the hook must never raise into the engine."""

    def boom(*a, **k):
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr(runs_db, "record_start", boom)
    handler = hermes_mpm._make_subagent_start_handler()
    # Must NOT raise.
    handler(child_session_id="c1", child_goal="g")


def test_subagent_stop_handler_swallows_db_error(db, monkeypatch):
    def boom(*a, **k):
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr(runs_db, "record_end", boom)
    handler = hermes_mpm._make_subagent_stop_handler()
    handler(child_session_id="c1", child_status="completed")  # must not raise


def test_async_complete_handler_closes_run_by_goal(db):
    """An async-complete marker closes the matching running run by goal.

    subagent_start fires for async children (with child_session_id + goal) but
    subagent_stop does NOT — so the pre_llm_call fallback must close the run when
    the ``[ASYNC DELEGATION COMPLETE — deleg_…]`` marker re-enters.
    """
    start = hermes_mpm._make_subagent_start_handler()
    start(child_session_id="c-async", parent_session_id="p1", child_goal="run nightly report")

    marker = (
        "[ASYNC DELEGATION COMPLETE — deleg_abc12345]\n"
        "A background subagent you dispatched earlier has finished.\n"
        "Original goal: run nightly report\n"
        "Role: leaf   Model: x\n"
        "Status: completed   API calls: 3   Duration: 12s\n"
        "--- RESULT ---\n"
        "report done\n"
    )
    handler = hermes_mpm._make_async_complete_handler()
    handler(user_message=marker)

    r = _rows(db)["c-async"]
    assert r["status"] == "done"
    assert r["delegation_id"] == "deleg_abc12345"
    assert r["ended_at"] is not None


def test_async_complete_handler_ignores_non_marker(db):
    handler = hermes_mpm._make_async_complete_handler()
    # Returns None and does nothing for a normal message.
    assert handler(user_message="hello there") is None


def test_runs_retention_days_reads_config():
    assert hermes_mpm._runs_retention_days({"runs": {"retention_days": 7}}) == 7
    # Default when absent.
    assert hermes_mpm._runs_retention_days({}) == hermes_mpm.DEFAULT_RETENTION_DAYS
    # Invalid value falls back to default.
    assert (
        hermes_mpm._runs_retention_days({"runs": {"retention_days": "nope"}})
        == hermes_mpm.DEFAULT_RETENTION_DAYS
    )


def test_register_wires_subagent_hooks(monkeypatch, tmp_path):
    """register(ctx) registers subagent_start + subagent_stop hooks."""
    monkeypatch.setattr(runs_db, "_db_path", lambda: tmp_path / "mpm_runs.db")

    class FakeCtx:
        def __init__(self):
            self.hooks = []

        def register_cli_command(self, **k):
            pass

        def register_command(self, **k):
            pass

        def register_hook(self, hook_name, callback):
            self.hooks.append(hook_name)

        def register_tool(self, **k):
            pass

        def register_skill(self, **k):
            pass

    ctx = FakeCtx()
    hermes_mpm.register(ctx)
    assert "subagent_start" in ctx.hooks
    assert "subagent_stop" in ctx.hooks


class _FakeCtx:
    """Minimal ctx recording only the hooks it was given (register smoke tests)."""

    def __init__(self):
        self.hooks = []

    def register_cli_command(self, **k):
        pass

    def register_command(self, **k):
        pass

    def register_hook(self, hook_name, callback):
        self.hooks.append(hook_name)

    def register_tool(self, **k):
        pass

    def register_skill(self, **k):
        pass


def test_subagent_start_handler_stamps_owner_pid(db):
    """The start hook must persist owner_pid = this process's pid so the gateway
    sweep can distinguish its own in-flight runs from a dead process's."""
    handler = hermes_mpm._make_subagent_start_handler()
    handler(parent_session_id="p1", child_session_id="c1", child_goal="g")
    assert _rows(db)["c1"]["owner_pid"] == os.getpid()


def test_register_does_not_sweep_when_not_gateway(db, monkeypatch):
    """A non-gateway process (no _HERMES_GATEWAY=1) must NOT sweep — running rows
    survive. This is the core data-corruption fix: `hermes mpm runs` and the
    dashboard load the plugin too, and must never mark the gateway's live runs
    crashed."""
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    # A live run owned by some other (gateway) process.
    runs_db._write(
        "INSERT INTO subagent_runs (run_id, status, started_at, owner_pid) "
        "VALUES (?, ?, ?, ?)",
        ("live", runs_db.STATUS_RUNNING, 1, os.getpid() + 1),
    )

    hermes_mpm.register(_FakeCtx())

    # The CLI/dashboard load must have left the live run alone.
    assert _rows(db)["live"]["status"] == "running"


def test_register_sweeps_when_gateway(db, monkeypatch):
    """Inside the gateway process (_HERMES_GATEWAY=1) the sweep DOES run, reaping
    prior-process running rows while leaving this process's own rows alone."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    # Prior dead process's run — should be reaped.
    runs_db._write(
        "INSERT INTO subagent_runs (run_id, status, started_at, owner_pid) "
        "VALUES (?, ?, ?, ?)",
        ("prior", runs_db.STATUS_RUNNING, 1, os.getpid() + 1),
    )
    # This process's own run — should survive (current pid).
    runs_db._write(
        "INSERT INTO subagent_runs (run_id, status, started_at, owner_pid) "
        "VALUES (?, ?, ?, ?)",
        ("mine", runs_db.STATUS_RUNNING, 1, os.getpid()),
    )

    hermes_mpm.register(_FakeCtx())

    rows = _rows(db)
    assert rows["prior"]["status"] == "crashed"
    assert rows["mine"]["status"] == "running"


def test_async_complete_handler_parses_duration(db):
    """The async-complete marker carries a `Duration: Ns` line; the handler must
    parse it and pass duration_ms to record_end so async runs show a duration
    like sync runs do."""
    start = hermes_mpm._make_subagent_start_handler()
    start(child_session_id="c-async", parent_session_id="p1", child_goal="nightly report")

    marker = (
        "[ASYNC DELEGATION COMPLETE — deleg_abc12345]\n"
        "Original goal: nightly report\n"
        "Status: completed   API calls: 3   Duration: 5s\n"
        "--- RESULT ---\n"
        "done\n"
    )
    hermes_mpm._make_async_complete_handler()(user_message=marker)

    r = _rows(db)["c-async"]
    assert r["status"] == "done"
    assert r["duration_ms"] == 5000


def test_async_run_end_to_end_not_swept_then_closed(db, monkeypatch):
    """Full async lifecycle: started (running) → a CLI/non-gateway register()
    runs (must NOT sweep it) → closed by the async-complete marker (done).

    This is the end-to-end proof of the data-corruption fix: an in-flight async
    run survives an interleaved `hermes mpm runs` load and is still closable."""
    # 1) Async child starts → running row, owner_pid stamped.
    start = hermes_mpm._make_subagent_start_handler()
    start(child_session_id="c-async", parent_session_id="p1", child_goal="long async job")
    assert _rows(db)["c-async"]["status"] == "running"

    # 2) A non-gateway process (the CLI) loads the plugin — must NOT sweep.
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    hermes_mpm.register(_FakeCtx())
    assert _rows(db)["c-async"]["status"] == "running"  # still alive — not crashed

    # 3) The async-complete marker re-enters and closes the run.
    marker = (
        "[ASYNC DELEGATION COMPLETE — deleg_ffff0001]\n"
        "Original goal: long async job\n"
        "Status: completed   API calls: 1   Duration: 7s\n"
        "--- RESULT ---\n"
        "ok\n"
    )
    hermes_mpm._make_async_complete_handler()(user_message=marker)

    r = _rows(db)["c-async"]
    assert r["status"] == "done"
    assert r["duration_ms"] == 7000
    assert r["ended_at"] is not None

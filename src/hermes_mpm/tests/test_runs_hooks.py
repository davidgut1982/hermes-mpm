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
        return {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM runs")}
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _reset_startup_maintenance():
    """Reset the once-per-process startup-maintenance latch between tests.

    Why: ``_startup_maintenance_done`` is module-level state that persists across
    tests in a process; without resetting it a test that ran maintenance would make
    every later test's maintenance a no-op, hiding regressions.
    What: Sets ``_startup_maintenance_done = False`` before and after each test.
    Test: This fixture's effect is exercised by the maintenance tests below, which
    assume a fresh (not-done) latch on entry.
    """
    hermes_mpm._startup_maintenance_done = False
    yield
    hermes_mpm._startup_maintenance_done = False


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
        "INSERT INTO runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("live", runs_db.STATUS_RUNNING, 1, os.getpid() + 1),
    )

    hermes_mpm.register(_FakeCtx())

    # The CLI/dashboard load must have left the live run alone.
    assert _rows(db)["live"]["status"] == "running"


def test_register_does_not_sweep_even_when_gateway(db, monkeypatch):
    """register() must NOT sweep at load time even in the gateway process.

    The gateway sets _HERMES_GATEWAY=1 only AFTER plugin register() runs (run.py),
    so at register() time the flag is unset and a register-time guard would always
    miss. The sweep is therefore deferred to the first hook-driven turn (see
    _maybe_run_startup_maintenance). Even if the env IS set when register() runs,
    register() only init_db()s — it never sweeps."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    # Prior dead process's run — would be reaped by a sweep, but register() must
    # NOT sweep, so it must survive untouched here.
    runs_db._write(
        "INSERT INTO runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("prior", runs_db.STATUS_RUNNING, 1, os.getpid() + 1),
    )

    hermes_mpm.register(_FakeCtx())

    assert _rows(db)["prior"]["status"] == "running"  # register did not sweep


def test_maybe_startup_maintenance_sweeps_once_when_gateway(db, monkeypatch):
    """With _HERMES_GATEWAY=1 and the latch not yet set, the lazy maintenance runs
    the sweep+purge exactly once: a prior-process running row is reaped, this
    process's own row survives, and a SECOND call is a no-op (no double sweep)."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    # Treat the prior owner pid as dead so liveness can't mask the scoping: under
    # pytest os.getpid()+1 is frequently a live sibling process, which the sweep
    # (correctly) refuses to reap. Forcing _pid_alive False makes the "prior dead
    # process" premise deterministic — matching test_runs_db.py's sweep tests.
    monkeypatch.setattr(runs_db, "_pid_alive", lambda pid: False)
    # Prior dead process's run — should be reaped by the lazy sweep.
    runs_db._write(
        "INSERT INTO runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("prior", runs_db.STATUS_RUNNING, 1, os.getpid() + 1),
    )
    # This process's own run — must survive (current pid).
    runs_db._write(
        "INSERT INTO runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("mine", runs_db.STATUS_RUNNING, 1, os.getpid()),
    )

    assert hermes_mpm._startup_maintenance_done is False
    hermes_mpm._maybe_run_startup_maintenance()

    rows = _rows(db)
    assert rows["prior"]["status"] == "crashed"
    assert rows["mine"]["status"] == "running"
    assert hermes_mpm._startup_maintenance_done is True

    # A second call must be a no-op: re-insert a fresh prior-process orphan and
    # confirm the latch keeps it from being swept again.
    runs_db._write(
        "INSERT INTO runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("prior2", runs_db.STATUS_RUNNING, 1, os.getpid() + 1),
    )
    hermes_mpm._maybe_run_startup_maintenance()
    assert _rows(db)["prior2"]["status"] == "running"  # not swept — already done


def test_maybe_startup_maintenance_noop_when_not_gateway(db, monkeypatch):
    """Without _HERMES_GATEWAY the lazy maintenance never sweeps, even called
    repeatedly — preserving the cross-process safety fix (CLI/dashboard never
    touch the live gateway's in-flight runs)."""
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    runs_db._write(
        "INSERT INTO runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("live", runs_db.STATUS_RUNNING, 1, os.getpid() + 1),
    )

    for _ in range(3):
        hermes_mpm._maybe_run_startup_maintenance()

    assert _rows(db)["live"]["status"] == "running"  # never swept without the env
    # The latch is NOT consumed in the non-gateway path: it returns before setting
    # the done flag, so a later gateway turn in the same process could still sweep.
    assert hermes_mpm._startup_maintenance_done is False


def test_maybe_startup_maintenance_swallows_db_error(db, monkeypatch):
    """A sweep/purge failure inside the lazy maintenance must never raise into the
    engine, and must still latch done so it doesn't retry-loop every hook."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")

    def boom(*a, **k):
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr(runs_db, "sweep_orphaned", boom)
    hermes_mpm._maybe_run_startup_maintenance()  # must NOT raise
    assert hermes_mpm._startup_maintenance_done is True  # latched despite failure


def test_maybe_startup_maintenance_runs_exactly_once_under_concurrency(db, monkeypatch):
    """Two threads entering _maybe_run_startup_maintenance simultaneously (env set)
    must run the sweep EXACTLY once (lock + double-checked latch), never twice."""
    import threading

    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    calls: list[int] = []

    def counting_sweep(now, pid):
        calls.append(1)
        return 0

    monkeypatch.setattr(runs_db, "sweep_orphaned", counting_sweep)
    monkeypatch.setattr(runs_db, "purge_old", lambda *a, **k: 0)

    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()  # maximize the race
        hermes_mpm._maybe_run_startup_maintenance()

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(calls) == 1  # sweep ran exactly once despite the race
    assert hermes_mpm._startup_maintenance_done is True


def test_subagent_start_handler_triggers_maintenance_then_records(db, monkeypatch):
    """The subagent_start handler runs startup maintenance once (gateway env) at the
    TOP, then records the run — both effects observable from a single call."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    # Force the prior owner pid dead (os.getpid()+1 is often a live sibling under
    # pytest, which the sweep correctly skips) so the reap premise is deterministic.
    monkeypatch.setattr(runs_db, "_pid_alive", lambda pid: False)
    # A prior-process orphan that the top-of-handler maintenance should reap.
    runs_db._write(
        "INSERT INTO runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("prior", runs_db.STATUS_RUNNING, 1, os.getpid() + 1),
    )

    handler = hermes_mpm._make_subagent_start_handler()
    handler(parent_session_id="p1", child_session_id="c1", child_goal="g")

    rows = _rows(db)
    assert rows["prior"]["status"] == "crashed"  # maintenance fired at the top
    assert rows["c1"]["status"] == "running"  # then the run was recorded
    assert hermes_mpm._startup_maintenance_done is True


def test_pre_llm_call_async_handler_triggers_maintenance(db, monkeypatch):
    """The pre_llm_call async-complete handler also runs startup maintenance once
    at the TOP (whichever hook fires first in the gateway does it). Driving it with
    a non-marker message still triggers maintenance, then returns None."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    # Force the prior owner pid dead (os.getpid()+1 is often a live sibling under
    # pytest, which the sweep correctly skips) so the reap premise is deterministic.
    monkeypatch.setattr(runs_db, "_pid_alive", lambda pid: False)
    runs_db._write(
        "INSERT INTO runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("prior", runs_db.STATUS_RUNNING, 1, os.getpid() + 1),
    )

    handler = hermes_mpm._make_async_complete_handler()
    assert handler(user_message="not a marker") is None  # still defers the turn

    assert _rows(db)["prior"]["status"] == "crashed"  # maintenance fired at the top
    assert hermes_mpm._startup_maintenance_done is True


def test_register_does_not_sweep_only_init_db(db, monkeypatch):
    """CLI/non-gateway register() path: register() must only init_db, never sweep —
    even with no _HERMES_GATEWAY env. A pre-existing orphan survives load."""
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    runs_db._write(
        "INSERT INTO runs (run_id, status, started_at, owner_pid) VALUES (?, ?, ?, ?)",
        ("orphan", runs_db.STATUS_RUNNING, 1, os.getpid() + 1),
    )

    hermes_mpm.register(_FakeCtx())

    assert _rows(db)["orphan"]["status"] == "running"  # register only init_db'd
    assert hermes_mpm._startup_maintenance_done is False  # register never latches


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


def test_post_tool_call_stamps_delegation_id_on_background_dispatch(db):
    """The direct-async happy path: subagent_start creates the run (delegation_id
    NULL); post_tool_call with a background delegate_task result stamps the
    delegation_id onto that row; the async-complete marker then closes it by
    EXACT delegation_id."""
    start = hermes_mpm._make_subagent_start_handler()
    start(child_session_id="c-async", parent_session_id="p1", child_goal="ship report")
    assert _rows(db)["c-async"]["delegation_id"] is None

    # delegate_task returns a JSON STRING for a background dispatch.
    result = (
        '{"status": "dispatched", "delegation_id": "deleg_abc12345", '
        '"goal": "ship report", "mode": "background", "note": "running"}'
    )
    post = hermes_mpm._make_post_tool_call_handler()
    post(tool_name="delegate_task", result=result)

    assert _rows(db)["c-async"]["delegation_id"] == "deleg_abc12345"

    # async-complete marker closes by delegation_id.
    marker = (
        "[ASYNC DELEGATION COMPLETE — deleg_abc12345]\n"
        "Original goal: ship report\n"
        "Status: completed   Duration: 9s\n"
        "--- RESULT ---\n"
        "ok\n"
    )
    hermes_mpm._make_async_complete_handler()(user_message=marker)
    r = _rows(db)["c-async"]
    assert r["status"] == "done"
    assert r["ended_at"] is not None


def test_async_closure_by_delegation_id_survives_truncated_goal(db):
    """TRUNCATION ROBUSTNESS: the marker's 'Original goal' is truncated/different
    from the stored goal, but closure STILL succeeds because correlation is by
    delegation_id — proving we no longer depend on goal text at completion."""
    start = hermes_mpm._make_subagent_start_handler()
    full_goal = "generate the very long nightly analytics report for region EMEA"
    start(child_session_id="c-trunc", parent_session_id="p1", child_goal=full_goal)

    # post_tool_call stamps the delegation_id (background dispatch carries the
    # full goal at dispatch time).
    result = {
        "status": "dispatched",
        "delegation_id": "deleg_70c99999",
        "goal": full_goal,
        "mode": "background",
    }
    hermes_mpm._make_post_tool_call_handler()(tool_name="delegate_task", result=result)

    # The completion marker's goal is TRUNCATED — would not match by goal text.
    marker = (
        "[ASYNC DELEGATION COMPLETE — deleg_70c99999]\n"
        "Original goal: generate the very long nightly analytics rep…\n"
        "Status: completed   Duration: 30s\n"
        "--- RESULT ---\n"
        "done\n"
    )
    hermes_mpm._make_async_complete_handler()(user_message=marker)

    r = _rows(db)["c-trunc"]
    assert r["status"] == "done"  # closed despite the goal mismatch
    assert r["duration_ms"] == 30000


def test_two_sequential_identical_goal_async_runs_close_independently(db):
    """Two SEQUENTIAL identical-goal direct-async runs: each subagent_start
    creates a row; each post_tool_call stamps a DISTINCT delegation_id (the
    ``delegation_id IS NULL`` filter disambiguates → no cross-stamp); each is
    closed by its OWN delegation_id."""
    start = hermes_mpm._make_subagent_start_handler()
    post = hermes_mpm._make_post_tool_call_handler()
    goal = "run the same job"

    # First dispatch.
    start(child_session_id="c1", parent_session_id="p", child_goal=goal)
    post(
        tool_name="delegate_task",
        result={
            "status": "dispatched",
            "delegation_id": "deleg_a0000001",
            "goal": goal,
            "mode": "background",
        },
    )
    # Second dispatch (same goal).
    start(child_session_id="c2", parent_session_id="p", child_goal=goal)
    post(
        tool_name="delegate_task",
        result={
            "status": "dispatched",
            "delegation_id": "deleg_b0000002",
            "goal": goal,
            "mode": "background",
        },
    )

    rows = _rows(db)
    # First stamp lands on the only row at the time (c1); second on c2.
    assert rows["c1"]["delegation_id"] == "deleg_a0000001"
    assert rows["c2"]["delegation_id"] == "deleg_b0000002"

    # Each closes by its own delegation_id, in any order.
    handler = hermes_mpm._make_async_complete_handler()
    handler(
        user_message=(
            "[ASYNC DELEGATION COMPLETE — deleg_b0000002]\nOriginal goal: x\nStatus: completed\n"
        )
    )
    handler(
        user_message=(
            "[ASYNC DELEGATION COMPLETE — deleg_a0000001]\nOriginal goal: y\nStatus: error\n"
        )
    )

    rows = _rows(db)
    assert rows["c2"]["status"] == "done"
    assert rows["c1"]["status"] == "failed"


def test_post_tool_call_ignores_sync_delegate_result(db):
    """A SYNC delegate_task result (no background dispatch markers) must NOT
    stamp anything — sync runs close via subagent_stop, not delegation_id."""
    start = hermes_mpm._make_subagent_start_handler()
    start(child_session_id="c-sync", parent_session_id="p", child_goal="sync work")

    # A typical sync result: a plain summary string, no delegation_id / mode.
    post = hermes_mpm._make_post_tool_call_handler()
    post(tool_name="delegate_task", result='{"status": "completed", "summary": "done"}')
    post(tool_name="delegate_task", result="just a plain text result")

    assert _rows(db)["c-sync"]["delegation_id"] is None


def test_post_tool_call_ignores_other_tools(db):
    """post_tool_call for a non-delegate_task tool is a no-op."""
    start = hermes_mpm._make_subagent_start_handler()
    start(child_session_id="c", parent_session_id="p", child_goal="g")
    post = hermes_mpm._make_post_tool_call_handler()
    # Even a result that LOOKS like a dispatch must be ignored for other tools.
    post(
        tool_name="some_other_tool",
        result={
            "status": "dispatched",
            "delegation_id": "deleg_x",
            "goal": "g",
            "mode": "background",
        },
    )
    assert _rows(db)["c"]["delegation_id"] is None


def test_post_tool_call_swallows_malformed_result_and_db_error(db, monkeypatch):
    """The post_tool_call handler must never raise into the engine: a malformed
    result and a DB error are both swallowed."""
    post = hermes_mpm._make_post_tool_call_handler()
    # Malformed JSON string — must not raise.
    post(tool_name="delegate_task", result="{not valid json")
    # Non-str/dict result — must not raise.
    post(tool_name="delegate_task", result=12345)
    # Missing result kw — must not raise.
    post(tool_name="delegate_task")

    # DB error path: stamp raises → swallowed.
    def boom(*a, **k):
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr(runs_db, "stamp_delegation_id", boom)
    post(
        tool_name="delegate_task",
        result={
            "status": "dispatched",
            "delegation_id": "deleg_x",
            "goal": "g",
            "mode": "background",
        },
    )


def test_register_wires_post_tool_call_hook(monkeypatch, tmp_path):
    """register(ctx) registers a post_tool_call hook for delegation stamping."""
    monkeypatch.setattr(runs_db, "_db_path", lambda: tmp_path / "mpm_runs.db")
    ctx = _FakeCtx()
    hermes_mpm.register(ctx)
    assert "post_tool_call" in ctx.hooks


def test_post_api_request_records_multi_tool_turn_and_bumps_run(db):
    """count>1: records a turn_batches row AND bumps the matching running run's
    max_batch_size/turn_count (correlated by session_id == run_id)."""
    start = hermes_mpm._make_subagent_start_handler()
    start(child_session_id="run-s1", parent_session_id="p1", child_goal="g")

    handler = hermes_mpm._make_post_api_request_handler()
    handler(
        assistant_tool_call_count=3,
        session_id="run-s1",
        turn_id="t1",
        api_request_id="api-1",
        model="glm-4.6",
    )

    # Global turn-level signal recorded.
    stats = runs_db.batch_stats()
    assert stats["tool_turns"] == 1
    assert stats["multi_tool_turns"] == 1
    # Per-subagent batch columns bumped.
    r = _rows(db)["run-s1"]
    assert r["turn_count"] == 1
    assert r["max_batch_size"] == 3


def test_post_api_request_duplicate_fire_does_not_double_count_turn(db):
    """The hook may fire more than once for a logically-single request (same
    api_request_id). record_turn_batch dedups via INSERT OR IGNORE; the per-run
    fold must be gated on the turn being NEWLY recorded so turn_count is not
    double-counted on a duplicate fire (max_batch_size stays correct either way)."""
    start = hermes_mpm._make_subagent_start_handler()
    start(child_session_id="run-s1", parent_session_id="p1", child_goal="g")

    handler = hermes_mpm._make_post_api_request_handler()
    kw = dict(
        assistant_tool_call_count=3,
        session_id="run-s1",
        turn_id="t1",
        api_request_id="api-1",
        model="m",
    )
    handler(**kw)
    handler(**kw)  # duplicate fire, same api_request_id

    # Exactly one turn-batch row (OR IGNORE) AND turn_count counted once.
    assert runs_db.batch_stats()["tool_turns"] == 1
    r = _rows(db)["run-s1"]
    assert r["turn_count"] == 1
    assert r["max_batch_size"] == 3


def test_post_api_request_count_zero_records_nothing(db):
    """count==0 (no tools that turn) carries no batch signal — record nothing."""
    start = hermes_mpm._make_subagent_start_handler()
    start(child_session_id="run-s1", parent_session_id="p1", child_goal="g")

    handler = hermes_mpm._make_post_api_request_handler()
    handler(
        assistant_tool_call_count=0,
        session_id="run-s1",
        turn_id="t1",
        api_request_id="api-1",
        model="m",
    )

    assert runs_db.batch_stats()["tool_turns"] == 0
    r = _rows(db)["run-s1"]
    assert (r["turn_count"] or 0) == 0
    assert (r["max_batch_size"] or 0) == 0


def test_post_api_request_single_tool_turn_recorded_not_multi(db):
    """count==1: recorded as a tool-turn but not a multi-tool turn."""
    handler = hermes_mpm._make_post_api_request_handler()
    handler(
        assistant_tool_call_count=1,
        session_id="main-session",
        turn_id="t1",
        api_request_id="api-1",
        model="m",
    )
    stats = runs_db.batch_stats()
    assert stats["tool_turns"] == 1
    assert stats["multi_tool_turns"] == 0


def test_post_api_request_swallows_db_error(db, monkeypatch):
    """A DB failure inside the hook must never raise into the engine."""

    def boom(*a, **k):
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr(runs_db, "record_turn_batch", boom)
    handler = hermes_mpm._make_post_api_request_handler()
    # Must NOT raise.
    handler(assistant_tool_call_count=2, session_id="s", api_request_id="api-1", model="m")


def test_post_api_request_derives_count_from_tool_calls_when_count_absent(db):
    """If assistant_tool_call_count is absent, derive the count from a carried
    list of assistant tool_calls (defensive kwarg extraction)."""
    handler = hermes_mpm._make_post_api_request_handler()
    handler(
        session_id="s",
        turn_id="t1",
        api_request_id="api-1",
        model="m",
        assistant_tool_calls=[{"id": "1"}, {"id": "2"}],
    )
    stats = runs_db.batch_stats()
    assert stats["tool_turns"] == 1
    assert stats["multi_tool_turns"] == 1


def test_post_api_request_no_api_request_id_is_noop(db):
    """Without an api_request_id there is no PK to key on — skip cleanly."""
    handler = hermes_mpm._make_post_api_request_handler()
    handler(assistant_tool_call_count=3, session_id="s", model="m")
    assert runs_db.batch_stats()["tool_turns"] == 0


def test_register_wires_post_api_request_hook(monkeypatch, tmp_path):
    """register(ctx) registers a post_api_request hook for batch telemetry."""
    monkeypatch.setattr(runs_db, "_db_path", lambda: tmp_path / "mpm_runs.db")
    ctx = _FakeCtx()
    hermes_mpm.register(ctx)
    assert "post_api_request" in ctx.hooks


def test_maybe_startup_maintenance_purges_turn_batches(db, monkeypatch):
    """The lazy startup maintenance also purges old turn_batches rows."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setattr(runs_db, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(hermes_mpm, "_RUNS_RETENTION_DAYS", 1)
    # An ancient turn-batch row that must be purged.
    runs_db.record_turn_batch("t-old", "a-old", "s", "m", 2, 10)
    # A recent one that must survive.
    import time as _t

    runs_db.record_turn_batch("t-new", "a-new", "s", "m", 2, int(_t.time()))

    hermes_mpm._maybe_run_startup_maintenance()

    stats = runs_db.batch_stats()
    assert stats["tool_turns"] == 1  # only the recent row remains


def test_async_complete_prefers_delegation_id_over_goal(db):
    """When BOTH a delegation_id match and a goal match exist, the handler closes
    by delegation_id (exact) — the goal-matching row is left alone."""
    start = hermes_mpm._make_subagent_start_handler()
    # Row that will be stamped + matched by delegation_id.
    start(child_session_id="by-id", parent_session_id="p", child_goal="ambiguous")
    hermes_mpm._make_post_tool_call_handler()(
        tool_name="delegate_task",
        result={
            "status": "dispatched",
            "delegation_id": "deleg_d1c10000",
            "goal": "ambiguous",
            "mode": "background",
        },
    )
    # A second running row sharing the goal but with NO delegation_id.
    start(child_session_id="by-goal", parent_session_id="p", child_goal="ambiguous")

    marker = (
        "[ASYNC DELEGATION COMPLETE — deleg_d1c10000]\n"
        "Original goal: ambiguous\n"
        "Status: completed\n"
    )
    hermes_mpm._make_async_complete_handler()(user_message=marker)

    rows = _rows(db)
    assert rows["by-id"]["status"] == "done"  # closed by exact delegation_id
    assert rows["by-goal"]["status"] == "running"  # untouched


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

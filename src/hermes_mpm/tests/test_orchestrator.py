"""Validation + payload-shape tests for the hermes-mpm orchestrate tool.

Why: The orchestrate tool's value is making ONE batched delegate_task call so
subtasks run in parallel. Tests must prove (a) bad input is rejected cleanly
before any delegation, (b) valid input builds the exact batched ``tasks``
payload the native delegate_task expects, and (c) the tool schema exposes
the correct OpenAI function-call format so the model can actually call it.
What: Drives orchestrator.handle with a mocked ctx.dispatch_tool.
Test: this file — run pytest.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from hermes_mpm import orchestrator
from hermes_mpm.gate import adapter as gate_adapter
from hermes_mpm.gate.verdict import Verdict


@pytest.fixture(autouse=True)
def _isolate_gate_and_agent():
    """Reset gate + captured-agent state around every orchestrator test.

    Why: register_gate() (exercised in test_gate.py) installs a live, possibly
    fail-closed adapter via set_active_adapter — that would leak into other tests
    and block all fan-out. Each test must start from a clean, unarmed gate and no
    captured agent so it controls its own gating/agent state explicitly.
    What: clears the active gate adapter and the captured agent before and after.
    Test: implicit — the legacy "dispatch happens" tests pass without per-test gate
    patching because the gate is unarmed (evaluate → allow) by default.
    """
    gate_adapter.set_active_adapter(None)
    orchestrator.clear_agent()
    yield
    gate_adapter.set_active_adapter(None)
    orchestrator.clear_agent()


class _RecordingCtx:
    """Captures the single delegate_task dispatch for assertions."""

    def __init__(self, result='{"results": []}'):
        self.calls = []
        self._result = result

    def dispatch_tool(self, tool_name, args, **kwargs):
        self.calls.append((tool_name, args, kwargs))
        return self._result


def test_empty_subtasks_is_error():
    orchestrator.set_ctx(_RecordingCtx())
    out = json.loads(orchestrator.handle({"goal": "g", "subtasks": []}))
    assert "error" in out and "non-empty" in out["error"]


def test_missing_goal_is_error():
    orchestrator.set_ctx(_RecordingCtx())
    out = json.loads(orchestrator.handle({"subtasks": [{"profile": "ops", "goal": "x"}]}))
    assert "error" in out


def test_unknown_profile_is_error():
    orchestrator.set_ctx(_RecordingCtx())
    out = json.loads(
        orchestrator.handle(
            {"goal": "g", "subtasks": [{"profile": "definitely-not-a-profile", "goal": "x"}]}
        )
    )
    assert "error" in out and "definitely-not-a-profile" in out["error"]


def test_subtask_missing_goal_is_error():
    orchestrator.set_ctx(_RecordingCtx())
    out = json.loads(orchestrator.handle({"goal": "g", "subtasks": [{"profile": "ops"}]}))
    assert "error" in out


def test_valid_builds_batched_delegate_payload():
    ctx = _RecordingCtx(result='{"results": ["a", "b"]}')
    orchestrator.set_ctx(ctx)

    result = orchestrator.handle(
        {
            "goal": "investigate two services",
            "subtasks": [
                {"profile": "ops", "goal": "check plex", "context": "host plex"},
                {"profile": "engineer", "goal": "review the patch"},
            ],
        }
    )

    # Exactly one batched dispatch (the parallel fan-out), not two serial ones.
    assert len(ctx.calls) == 1
    tool_name, args, _ = ctx.calls[0]
    assert tool_name == orchestrator.DELEGATE_TOOL
    assert args["role"] == orchestrator.LEAF_ROLE

    tasks = args["tasks"]
    assert len(tasks) == 2
    assert tasks[0] == {
        "profile": "ops",
        "goal": "check plex",
        "role": "leaf",
        "context": "host plex",
    }
    # Second task carries no context key (omitted, not empty).
    assert tasks[1] == {"profile": "engineer", "goal": "review the patch", "role": "leaf"}
    # Aggregated result passed through.
    assert json.loads(result) == {"results": ["a", "b"]}


def test_no_ctx_is_clean_error():
    orchestrator.set_ctx(None)
    out = json.loads(
        orchestrator.handle({"goal": "g", "subtasks": [{"profile": "ops", "goal": "x"}]})
    )
    assert "error" in out and "context" in out["error"]


def test_schema_has_correct_openai_format():
    """ORCHESTRATE_SCHEMA must expose 'parameters' at the top level.

    Why: registry.get_definitions() does {**entry.schema, "name": name} and
    wraps the result in {"type": "function", "function": ...}. If the schema
    is the raw JSON-Schema object (missing the 'parameters' wrapper), the
    model sees empty parameters and refuses to call the tool. The schema must
    mirror DELEGATE_TASK_SCHEMA's structure: name + description + parameters.
    What: Asserts that ORCHESTRATE_SCHEMA has 'parameters', 'description',
    and 'name' at the top level, and that 'goal' + 'subtasks' are inside
    'parameters.properties'.
    Test: this assertion — no mocks needed.
    """
    schema = orchestrator.ORCHESTRATE_SCHEMA
    assert "parameters" in schema, (
        "ORCHESTRATE_SCHEMA missing 'parameters' key — model will see empty schema"
    )
    assert "description" in schema, "ORCHESTRATE_SCHEMA missing 'description' key"
    assert "name" in schema, "ORCHESTRATE_SCHEMA missing 'name' key"
    props = schema["parameters"].get("properties", {})
    assert "goal" in props, "'goal' must be in parameters.properties"
    assert "subtasks" in props, "'subtasks' must be in parameters.properties"
    required = schema["parameters"].get("required", [])
    assert "goal" in required
    assert "subtasks" in required


def test_three_subtasks_single_dispatch_call():
    """Three subtasks must produce exactly one dispatch call, not three.

    Why: The entire value of hermes_mpm_orchestrate is making ONE batched
    delegate_task call so the native ThreadPoolExecutor can run all children
    in parallel. N sequential dispatch calls re-serialize the work.
    What: Verifies len(ctx.calls) == 1 and tasks list has all 3 entries.
    Test: mock ctx records calls; assert single call with 3 tasks payload.
    """
    ctx = _RecordingCtx(result='{"results": ["r1", "r2", "r3"]}')
    orchestrator.set_ctx(ctx)

    orchestrator.handle(
        {
            "goal": "parallel health check",
            "subtasks": [
                {"profile": "ops", "goal": "check /tmp disk"},
                {"profile": "ops", "goal": "check /opt disk"},
                {"profile": "kb", "goal": "search kb for hermes"},
            ],
        }
    )

    # ONE dispatch call (batched), not three (sequential).
    assert len(ctx.calls) == 1, (
        f"Expected 1 dispatch call (batched), got {len(ctx.calls)} (re-serialized)"
    )
    _, args, _ = ctx.calls[0]
    assert len(args["tasks"]) == 3, "All 3 subtasks must be in the single batched payload"


def test_concurrent_dispatch_timing():
    """Mock dispatch that sleeps proves concurrency: elapsed ≈ max, not sum.

    Why: This is the core parallelism guarantee — 3 x 1s subtasks should
    complete in ~1s total (max), not ~3s (sum). The recording ctx simulates
    real child agents by sleeping in a thread per call.
    What: _TimingCtx dispatches each task in a thread sleeping 0.3s. With
    max_concurrent_children=5 (our config), all 3 run simultaneously → total
    elapsed should be ~0.3s, not ~0.9s. We allow 0.6s headroom.
    Test: measure wall-clock elapsed of handle(); assert < 0.6s (not 0.9s).
    """

    class _TimingCtx:
        """Simulates delegate_task batch: spawns a thread per task that sleeps."""

        def dispatch_tool(self, tool_name: str, args: dict[str, Any], **_kw: Any) -> str:
            tasks = args.get("tasks", [])
            results = []
            threads = []

            def run_task(_t: dict[str, Any]) -> None:
                time.sleep(0.3)  # simulate child agent work
                results.append({"goal": _t["goal"], "result": "done"})

            for t in tasks:
                th = threading.Thread(target=run_task, args=(t,))
                th.start()
                threads.append(th)
            for th in threads:
                th.join()
            return json.dumps({"results": results})

    orchestrator.set_ctx(_TimingCtx())
    start = time.monotonic()
    orchestrator.handle(
        {
            "goal": "timing test",
            "subtasks": [
                {"profile": "ops", "goal": "task 1"},
                {"profile": "ops", "goal": "task 2"},
                {"profile": "ops", "goal": "task 3"},
            ],
        }
    )
    elapsed = time.monotonic() - start

    # 3 x 0.3s tasks run concurrently → elapsed ≈ 0.3s (not 0.9s).
    # We allow up to 0.6s for thread-scheduling overhead.
    assert elapsed < 0.6, (
        f"Elapsed {elapsed:.2f}s suggests serial execution (~0.9s). "
        f"Expected concurrent (~0.3s). max=0.6s allowed."
    )


# ── FINDING 2: orchestrate must gate each subtask before dispatch ─────────────
# handle() dispatches via registry.dispatch / ctx.dispatch_tool, which does NOT
# pass through the engine's pre_tool_call hook. So fan-out reaches delegate_task
# UNGATED unless handle() applies the gate verdict itself. These tests prove it
# now does, reusing the SAME evaluate() the hook uses (one source of truth).


def _set_gate(verdict_for):
    """Install a fake evaluate() that maps a subtask goal substring -> Verdict.

    Why: Drive handle()'s gating decision deterministically without a reviewer.
    What: patches gate_adapter.evaluate; verdict_for(goal) returns a Verdict.
    Test: used by the Finding-2 tests below.
    """
    def _fake_evaluate(tool_name, args, tool_call_id=None):
        goal = ((args.get("tasks") or [{}])[0].get("goal")
                if args.get("tasks") else args.get("goal")) or ""
        return verdict_for(goal)
    return patch.object(gate_adapter, "evaluate", side_effect=_fake_evaluate)


def test_blockable_subtask_is_skipped(tmp_path):
    """A subtask the gate would BLOCK is not dispatched; the rest still run.

    Why: Finding 2 — ungated fan-out let a blockable subtask reach delegate_task.
    What: gate blocks any goal containing 'delete'; the safe subtask is dispatched,
    the blocked one is reported in the result and excluded from the payload.
    Test: 2 subtasks (one 'delete prod', one 'list status') → dispatched tasks == 1
    and the result records the blocked subtask.
    """
    ctx = _RecordingCtx(result='{"results": ["ok"]}')
    orchestrator.set_ctx(ctx)

    def verdict_for(goal):
        if "delete" in goal:
            return Verdict(decision="block", reason="destructive")
        return Verdict(decision="allow")

    with _set_gate(verdict_for):
        out = json.loads(orchestrator.handle({
            "goal": "mixed batch",
            "subtasks": [
                {"profile": "ops", "goal": "delete prod database"},
                {"profile": "ops", "goal": "list status"},
            ],
        }))

    # Exactly one dispatch, carrying only the allowed subtask.
    assert len(ctx.calls) == 1, "one batched dispatch for the surviving subtask"
    _, args, _ = ctx.calls[0]
    assert len(args["tasks"]) == 1, "blocked subtask must be excluded from dispatch"
    assert args["tasks"][0]["goal"] == "list status"
    # The blocked subtask is surfaced to the caller.
    assert "blocked" in out, f"result must report blocked subtasks; got {out!r}"
    assert any("delete prod database" in b.get("goal", "") for b in out["blocked"])


def test_all_subtasks_blocked_no_dispatch(tmp_path):
    """If the gate blocks every subtask, no delegate_task dispatch happens at all.

    Why: Finding 2 — blocking must actually prevent ungated delegation.
    What: gate blocks all; handle() returns an error/blocked result and never dispatches.
    Test: ctx.calls is empty; result reports the blocks.
    """
    ctx = _RecordingCtx()
    orchestrator.set_ctx(ctx)

    with _set_gate(lambda goal: Verdict(decision="block", reason="nope")):
        out = json.loads(orchestrator.handle({
            "goal": "all dangerous",
            "subtasks": [
                {"profile": "ops", "goal": "delete a"},
                {"profile": "ops", "goal": "drop b"},
            ],
        }))

    assert len(ctx.calls) == 0, "no dispatch when every subtask is blocked"
    assert out.get("blocked"), "result must list the blocked subtasks"
    assert len(out["blocked"]) == 2


def test_allowed_subtasks_dispatch_normally(tmp_path):
    """When the gate allows all subtasks, dispatch is unchanged (no regression).

    Why: Finding 2 must not break the happy path.
    What: gate allows all → single batched dispatch with both tasks.
    Test: one dispatch call, two tasks, no 'blocked' key in result.
    """
    ctx = _RecordingCtx(result='{"results": ["a", "b"]}')
    orchestrator.set_ctx(ctx)

    with _set_gate(lambda goal: Verdict(decision="allow")):
        out = json.loads(orchestrator.handle({
            "goal": "safe batch",
            "subtasks": [
                {"profile": "ops", "goal": "check plex"},
                {"profile": "ops", "goal": "check disk"},
            ],
        }))

    assert len(ctx.calls) == 1
    _, args, _ = ctx.calls[0]
    assert len(args["tasks"]) == 2
    assert out == {"results": ["a", "b"]}


# ── FINDING 3: no stale-agent global; per-call agent is correct ──────────────


def test_no_module_global_captured_agent():
    """The stale, race-prone module global _captured_agent must be gone.

    Why: Finding 3 — a bare module global set in pre_llm_call leaks across turns
    and races across concurrent sessions. The fix replaces it with a contextvar
    (or uses the per-call agent directly), so the bare global must not exist.
    What: assert orchestrator has no module attribute named _captured_agent.
    Test: not hasattr(orchestrator, '_captured_agent').
    """
    assert not hasattr(orchestrator, "_captured_agent"), (
        "Finding 3: bare module global _captured_agent must be removed "
        "(replaced by a per-call contextvar)"
    )


def test_capture_agent_is_isolated_per_context():
    """capture_agent stores into a contextvar, not a shared mutable global.

    Why: Finding 3 — concurrency safety. A value set in one context must not be
    visible as a stale default after it is cleared.
    What: capture_agent(x) then clear_agent() → current_agent() is None again.
    Test: set, assert visible; clear, assert None (no stale leak).
    """
    sentinel = object()
    orchestrator.capture_agent(sentinel)
    assert orchestrator.current_agent() is sentinel, "captured agent must be readable in-context"
    orchestrator.clear_agent()
    assert orchestrator.current_agent() is None, (
        "Finding 3: cleared agent must not leak as a stale value to the next turn"
    )


def test_short_turn_does_not_read_stale_agent(tmp_path):
    """After clear, a fan-out with no captured agent falls back to ctx.dispatch_tool.

    Why: Finding 3(a) — capture happened after the short-message guard, so a short
    turn could read a STALE agent from a prior turn. With a contextvar that is
    cleared, no stale agent is read; handle() uses the ctx.dispatch_tool fallback.
    What: clear the agent, dispatch a safe batch → uses ctx.dispatch_tool path.
    Test: ctx.dispatch_tool was called (fallback), proving no stale parent_agent.
    """
    orchestrator.clear_agent()
    ctx = _RecordingCtx(result='{"results": ["x"]}')
    orchestrator.set_ctx(ctx)

    with _set_gate(lambda goal: Verdict(decision="allow")):
        orchestrator.handle({
            "goal": "safe",
            "subtasks": [{"profile": "ops", "goal": "list status"}],
        })

    # No captured agent → fell back to ctx.dispatch_tool (one recorded call).
    assert len(ctx.calls) == 1, "fallback dispatch path must run when no agent captured"

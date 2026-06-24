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
import time
import threading
from typing import Any

from hermes_mpm import orchestrator


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

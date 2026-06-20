"""Validation + payload-shape tests for the hermes-mpm orchestrate tool.

Why: The orchestrate tool's value is making ONE batched delegate_task call so
subtasks run in parallel. Tests must prove (a) bad input is rejected cleanly
before any delegation, and (b) valid input builds the exact batched ``tasks``
payload the native delegate_task expects.
What: Drives orchestrator.handle with a mocked ctx.dispatch_tool.
Test: this file — run pytest.
"""

from __future__ import annotations

import json

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

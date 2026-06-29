"""Tests for the leaf/subagent batching hint injected by the decompose hook.

Why: Leaf subagents given a multi-lookup task batch their tool calls only ~67%
of the time without an explicit instruction; the other ~33% serialize into
one-tool-per-turn. The decompose hint hook used to inject NOTHING on
subagent/leaf turns. This module pins the new behavior: leaf/subagent turns now
receive a compact, leaf-appropriate batching hint (no PM/orchestration framing),
while the main/interactive path still injects ``_DECOMPOSE_HINT`` unchanged, and
the leaf contract still clears any captured parent agent.
What: Drives ``_make_decompose_hint_hook()`` directly across platforms and spies
on ``orchestrator.clear_agent``.
Test: ``pytest src/hermes_mpm/tests/test_leaf_batch_hint.py``.
"""

from __future__ import annotations

import pytest

import hermes_mpm
from hermes_mpm import orchestrator


@pytest.mark.parametrize("platform", ["subagent", "leaf", "SubAgent", "LEAF"])
def test_leaf_turn_injects_batch_hint(platform):
    """Why: Leaves had no batching guidance; verify they now get the leaf hint.
    What: hook(platform=leaf/subagent) returns a non-None dict whose context
    contains a distinctive substring of the leaf batch guidance.
    Test: this test.
    """
    hook = hermes_mpm._make_decompose_hint_hook()
    out = hook(platform=platform, user_message="check versions of foo, bar, baz")
    assert out is not None
    assert out["context"] == hermes_mpm._LEAF_BATCH_HINT
    # Distinctive substring of the parallel-batching guidance (v1, retained).
    assert "PARALLEL tool calls in a SINGLE response" in out["context"]
    # Distinctive substring of the v2 anti-redundancy guidance.
    assert "EXACTLY ONCE" in out["context"]
    assert "already used" in out["context"]
    # A leaf cannot delegate/orchestrate — the PM framing must NOT leak in.
    assert "hermes_mpm_orchestrate" not in out["context"]
    assert "orchestrate" not in out["context"].lower()


def test_main_path_injects_decompose_hint_unchanged():
    """Why: Guard against regressing the PM/interactive injection.
    What: A normal multi-word CLI turn still returns ``_DECOMPOSE_HINT``.
    Test: this test.
    """
    hook = hermes_mpm._make_decompose_hint_hook()
    out = hook(
        platform="cli",
        user_message="research the latest AI news and also look up our KB on MCP",
    )
    assert out is not None
    assert out["context"] == hermes_mpm._DECOMPOSE_HINT
    assert "hermes_mpm_orchestrate" in out["context"]


def test_leaf_path_still_clears_agent(monkeypatch):
    """Why: The leaf contract must still clear a captured parent agent so a child
    turn cannot inherit stale parent orchestration state.
    What: Spy on ``orchestrator.clear_agent``; assert it is invoked on the leaf
    path even though the hook now returns a hint instead of None.
    Test: this test.
    """
    calls = {"n": 0}

    def _spy():
        calls["n"] += 1

    monkeypatch.setattr(orchestrator, "clear_agent", _spy)
    hook = hermes_mpm._make_decompose_hint_hook()
    out = hook(platform="subagent", user_message="read these four files")
    assert calls["n"] == 1
    assert out is not None
    assert out["context"] == hermes_mpm._LEAF_BATCH_HINT


def test_leaf_hint_is_compact_and_distinct():
    """Why: The leaf hint must stay short and must not reuse PM framing.
    What: Assert the constant is non-trivially shorter than _DECOMPOSE_HINT and
    shares no orchestration vocabulary.
    Test: this test.
    """
    assert len(hermes_mpm._LEAF_BATCH_HINT) < len(hermes_mpm._DECOMPOSE_HINT)
    assert "profile=" not in hermes_mpm._LEAF_BATCH_HINT
    assert "subtask" not in hermes_mpm._LEAF_BATCH_HINT.lower()

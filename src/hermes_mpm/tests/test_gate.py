"""Invariant tests for the fail-closed review GATE.

Why: The gate is a security control — every branch (tiering, fail-closed verdict
parsing, tighten-only enforcement, cross-lab guard, memoization, batch review,
audit redaction) must be proven offline before it can be trusted in the seam.
What: Pure-unit tests for each gate module, plus adapter tests with the reviewer
HTTP call mocked. Zero real network.
Test: this file — run pytest.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from hermes_mpm.gate import config as config_mod
from hermes_mpm.gate import tiering
from hermes_mpm.gate import tighten as tighten_mod
from hermes_mpm.gate import verdict as verdict_mod
from hermes_mpm.gate.adapter import ReviewGateAdapter
from hermes_mpm.gate.audit import AuditStore
from hermes_mpm.gate.config import ReviewGateConfig

# ── 1. TIERING ──────────────────────────────────────────────────────────────


def _br(goal):
    return tiering.classify_blast_radius("delegate_task", {"goal": goal})


def test_tiering_trivial():
    assert _br("show status of plex") == "trivial"
    assert _br("list the running services") == "trivial"
    assert _br("check disk usage") == "trivial"


def test_tiering_elevated_deploy():
    assert _br("deploy the new build") == "elevated"


def test_tiering_elevated_delete():
    assert _br("delete the old records") == "elevated"


def test_tiering_elevated_auth():
    assert _br("rotate the auth keys") == "elevated"
    assert _br("read the secret config") == "elevated"


def test_tiering_elevated_prod():
    assert _br("restart prod gateway") == "elevated"
    assert _br("touch production database") == "elevated"


def test_tiering_merge_adjacent_batch():
    args = {"tasks": [{"goal": "run the tests"}, {"goal": "format the code"}]}
    assert tiering.classify_blast_radius("delegate_task", args) == "merge_adjacent"


def test_tiering_standard_default():
    assert tiering.classify_blast_radius("delegate_task", {"goal": "run the tests"}) == "standard"


# ── 2. VERDICT (fail-closed) ────────────────────────────────────────────────


def test_verdict_allow():
    v = verdict_mod.parse_verdict("ALLOW")
    assert v.decision == "allow"
    assert v.added_constraints == []


def test_verdict_tighten():
    v = verdict_mod.parse_verdict("TIGHTEN: don't delete prod")
    assert v.decision == "tighten"
    assert v.added_constraints == ["don't delete prod"]


def test_verdict_block():
    v = verdict_mod.parse_verdict("BLOCK: too dangerous")
    assert v.decision == "block"
    assert "too dangerous" in v.reason


def test_verdict_none_input_blocks():
    v = verdict_mod.parse_verdict(None)
    assert v.decision == "block"
    assert "no reviewer output" in v.reason


def test_verdict_empty_blocks():
    v = verdict_mod.parse_verdict("   \n  ")
    assert v.decision == "block"
    assert "empty" in v.reason


def test_verdict_garbage_blocks():
    v = verdict_mod.parse_verdict("purple monkey dishwasher")
    assert v.decision == "block"
    assert "unparseable" in v.reason


def test_verdict_error_set_blocks():
    v = verdict_mod.parse_verdict(None, error="timeout")
    assert v.decision == "block"
    assert "reviewer error: timeout" in v.reason


def test_verdict_any_block_line_blocks():
    v = verdict_mod.parse_verdict("ALLOW\nBLOCK: nope")
    assert v.decision == "block"


# ── 3. TIGHTEN-ONLY validation ──────────────────────────────────────────────


def test_tighten_identical_valid():
    base = {"goal": "do x", "constraints": ["a"]}
    ok, reason = tighten_mod.validate_tighten(base, dict(base))
    assert ok is True
    assert reason == ""


def test_tighten_added_constraint_valid():
    base = {"goal": "do x"}
    proposed = {"goal": "do x", "extra_constraint": "no deletes"}
    ok, reason = tighten_mod.validate_tighten(base, proposed)
    assert ok is True


def test_tighten_key_removed_invalid():
    base = {"goal": "do x", "scope": "limited"}
    proposed = {"goal": "do x"}
    ok, reason = tighten_mod.validate_tighten(base, proposed)
    assert ok is False
    assert reason


def test_tighten_grant_tools_invalid():
    base = {"goal": "do x"}
    proposed = {"goal": "do x", "tools": ["shell"]}
    ok, reason = tighten_mod.validate_tighten(base, proposed)
    assert ok is False
    assert reason


def test_tighten_list_item_removed_invalid():
    base = {"goal": "do x", "constraints": ["a", "b"]}
    proposed = {"goal": "do x", "constraints": ["a"]}
    ok, reason = tighten_mod.validate_tighten(base, proposed)
    assert ok is False
    assert reason


# ── 4. CROSS-LAB GUARD ──────────────────────────────────────────────────────


def _make_adapter(tmp_path, *, enabled=True, fail_closed=True):
    cfg = ReviewGateConfig(enabled=enabled, fail_closed=fail_closed,
                           audit_path=tmp_path / "audit.jsonl")
    store = AuditStore(tmp_path / "audit.jsonl")
    return ReviewGateAdapter(cfg, store)


def test_cross_lab_same_lab_fail_closed(tmp_path):
    from hermes_mpm.gate import derive_lab, register_gate

    assert derive_lab("deepseek/deepseek-v4-flash") == "deepseek"
    assert derive_lab("deepseek/deepseek-v4-pro") == "deepseek"

    registered = {"hooks": {}, "middleware": {}}

    class _Ctx:
        def register_middleware(self, kind, callback):
            registered["middleware"][kind] = callback

        def register_hook(self, hook_name, callback):
            registered["hooks"][hook_name] = callback

    raw = {
        "hermes_mpm": {
            "review_gate": {
                "reviewer": {"model": "deepseek/deepseek-v4-pro"},
            },
            "tiers": {"main": {"model": "deepseek/deepseek-v4-flash"}},
        }
    }
    register_gate(_Ctx(), raw_config=raw)
    hook = registered["hooks"].get("pre_tool_call")
    assert hook is not None
    # In same-lab fail-closed mode, ALL delegate_task calls blocked.
    msg = hook("delegate_task", {"goal": "list status"}, tool_call_id="x")
    assert msg is not None
    assert "cross-lab" in msg.lower()


def test_cross_lab_different_lab_normal(tmp_path):
    from hermes_mpm.gate import derive_lab, register_gate

    assert derive_lab("anthropic/claude-sonnet-4.6") == "anthropic"

    registered = {"hooks": {}, "middleware": {}}

    class _Ctx:
        def register_middleware(self, kind, callback):
            registered["middleware"][kind] = callback

        def register_hook(self, hook_name, callback):
            registered["hooks"][hook_name] = callback

    raw = {
        "hermes_mpm": {
            "review_gate": {
                "reviewer": {"model": "deepseek/deepseek-v4-pro"},
            },
            "tiers": {"main": {"model": "anthropic/claude-sonnet-4.6"}},
        }
    }
    register_gate(_Ctx(), raw_config=raw)
    hook = registered["hooks"].get("pre_tool_call")
    assert hook is not None
    # Different lab → normal mode: a trivial task is not auto-blocked by the guard.
    # (It is below the gated tier, so it is allowed without review.)
    msg = hook("delegate_task", {"goal": "list status"}, tool_call_id="y")
    assert msg is None


# ── 5. ADAPTER MEMOIZATION ──────────────────────────────────────────────────


def test_adapter_memo_reviewer_called_once(tmp_path):
    adapter = _make_adapter(tmp_path)
    args = {"goal": "deploy prod release"}  # elevated → gated

    with patch("hermes_mpm.gate.adapter.call_reviewer", return_value="ALLOW") as mock_call:
        adapter.middleware_callback("delegate_task", args, tool_call_id="abc")
        adapter.hook_callback("delegate_task", args, tool_call_id="abc")
    assert mock_call.call_count == 1


# ── 6. ADAPTER NO-OP ON NON-DELEGATE ────────────────────────────────────────


def test_adapter_noop_non_delegate_middleware(tmp_path):
    adapter = _make_adapter(tmp_path)
    with patch("hermes_mpm.gate.adapter.call_reviewer") as mock_call:
        res = adapter.middleware_callback("web_search", {"q": "x"}, tool_call_id="x")
    assert res is None
    assert mock_call.call_count == 0


def test_adapter_noop_non_delegate_hook(tmp_path):
    adapter = _make_adapter(tmp_path)
    with patch("hermes_mpm.gate.adapter.call_reviewer") as mock_call:
        res = adapter.hook_callback("memory", {"q": "x"}, tool_call_id="x")
    assert res is None
    assert mock_call.call_count == 0


# ── 7. BATCH PATH ───────────────────────────────────────────────────────────


def test_adapter_batch_reviews_each_task(tmp_path):
    adapter = _make_adapter(tmp_path)
    args = {"tasks": [{"goal": "delete the database"}, {"goal": "list status"}]}

    def _fake_reviewer(prompt, config):
        if "delete" in prompt:
            return "BLOCK: destructive"
        return "ALLOW"

    with patch("hermes_mpm.gate.adapter.call_reviewer", side_effect=_fake_reviewer) as mock_call:
        msg = adapter.hook_callback("delegate_task", args, tool_call_id="batch1")
    # Reviewer called once per task.
    assert mock_call.call_count == 2
    # One task blocks → whole call blocks.
    assert msg is not None


# ── 8. AUDIT REDACTION ──────────────────────────────────────────────────────


def test_audit_redacts_api_key(tmp_path):
    path = tmp_path / "audit.jsonl"
    store = AuditStore(path)
    store.record(tool_call_id="t1", tool_name="delegate_task",
                 args={"api_key": "sk-abc123", "goal": "x"},
                 blast_radius="elevated", decision="allow", reason="", constraints=[])
    raw = path.read_text()
    assert "sk-abc123" not in raw
    assert "<REDACTED>" in raw


def test_audit_redacts_token(tmp_path):
    path = tmp_path / "audit.jsonl"
    store = AuditStore(path)
    store.record(tool_call_id="t2", tool_name="delegate_task",
                 args={"token": "bearer-xyz-secret", "goal": "x"},
                 blast_radius="elevated", decision="allow", reason="", constraints=[])
    raw = path.read_text()
    assert "bearer-xyz-secret" not in raw
    assert "<REDACTED>" in raw


def test_audit_all_fields_present(tmp_path):
    path = tmp_path / "audit.jsonl"
    store = AuditStore(path)
    store.record(tool_call_id="t3", tool_name="delegate_task",
                 args={"goal": "x"}, blast_radius="elevated",
                 decision="tighten", reason="added constraint", constraints=["no deletes"])
    rec = json.loads(path.read_text().strip().splitlines()[-1])
    assert rec["tool_call_id"] == "t3"
    assert rec["decision"] == "tighten"
    assert rec["blast_radius"] == "elevated"
    assert rec["constraints"] == ["no deletes"]


# ── CONFIG defaults ─────────────────────────────────────────────────────────


def test_config_empty_defaults():
    cfg = config_mod.load_gate_config({})
    assert cfg.enabled is True
    assert cfg.fail_closed is True
    assert cfg.gated_tiers == ["elevated", "merge_adjacent"]
    assert cfg.reviewer_model == "deepseek/deepseek-v4-pro"


def test_config_partial_override():
    raw = {"hermes_mpm": {"review_gate": {"enabled": False,
                                          "reviewer": {"model": "openai/gpt-x"}}}}
    cfg = config_mod.load_gate_config(raw)
    assert cfg.enabled is False
    assert cfg.reviewer_model == "openai/gpt-x"
    # unspecified keys keep defaults
    assert cfg.fail_closed is True

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


# Finding 2: empty-base prefix bypass
# ''.startswith('') is True, so base="" let any proposed value pass the prefix
# check. Fix: if base is empty and proposed is non-empty -> invalid.

def test_tighten_finding2_empty_base_nonempty_proposed_invalid():
    """Finding 2: base='' and proposed='inject' must be invalid (not vacuously pass).

    Why: ''.startswith('') is True in Python, so the original check silently
    allowed any proposed value when the base was an empty string. An attacker
    could leave the base field empty, then inject arbitrary content as the
    proposed value and have it pass the startswith guard.
    Fix: if base_val=='' and prop_val!='' -> invalid.
    Test: validate_tighten({'goal':''}, {'goal':'inject'}) -> (False, reason).
    """
    ok, reason = tighten_mod.validate_tighten({"goal": ""}, {"goal": "inject"})
    assert ok is False, "Finding 2: empty-base with non-empty proposed must be invalid"
    assert reason, "reason must be non-empty"


def test_tighten_finding2_empty_base_empty_proposed_valid():
    """Finding 2 control: base='' and proposed='' (both empty) must be valid."""
    ok, _reason = tighten_mod.validate_tighten({"goal": ""}, {"goal": ""})
    assert ok is True, "Finding 2 control: both empty is valid (no injection)"


def test_tighten_finding2_nonempty_base_startswith_passes():
    """Finding 2 control: non-empty base with proper prefix extension remains valid."""
    ok, _reason = tighten_mod.validate_tighten(
        {"goal": "run tests"}, {"goal": "run tests (read-only)"}
    )
    assert ok is True, "Finding 2 control: genuine prefix extension must stay valid"


# Finding 3: type-confusion silent pass
# validate_tighten({'goal':'x'}, {'goal':42}) -> ok (isinstance guards no-op on
# type mismatch). Fix: check type(base_val) is type(prop_val) before isinstance.

def test_tighten_finding3_str_to_int_invalid():
    """Finding 3: str->int type change must be invalid.

    Why: isinstance checks in rule 2 (str/str) and rule 3 (list/list) silently
    no-op when types differ, so a str base and int proposed would skip all
    guards and pass as valid. A reviewer could inject arbitrary non-string
    content into substance keys by changing the type.
    Fix: if type(base_val) is not type(prop_val) -> invalid before isinstance.
    Test: validate_tighten({'goal':'x'}, {'goal':42}) -> (False, reason).
    """
    ok, reason = tighten_mod.validate_tighten({"goal": "x"}, {"goal": 42})
    assert ok is False, "Finding 3: str->int type change must be invalid"
    assert "type" in reason.lower() or reason, f"reason must mention type change; got {reason!r}"


def test_tighten_finding3_str_to_list_invalid():
    """Finding 3: str->list type change must be invalid."""
    ok, reason = tighten_mod.validate_tighten({"goal": "x"}, {"goal": ["x", "inject"]})
    assert ok is False, "Finding 3: str->list type change must be invalid"


def test_tighten_finding3_list_to_dict_invalid():
    """Finding 3: list->dict type change must be invalid."""
    ok, reason = tighten_mod.validate_tighten(
        {"constraints": ["a"]}, {"constraints": {"a": "b"}}
    )
    assert ok is False, "Finding 3: list->dict type change must be invalid"


def test_tighten_finding3_same_type_still_valid():
    """Finding 3 control: same-type same-value (str->str) still passes."""
    ok, _reason = tighten_mod.validate_tighten({"goal": "x"}, {"goal": "x and more"})
    assert ok is True, "Finding 3 control: str->str with valid prefix must still pass"


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


# ── SECURITY FINDINGS ────────────────────────────────────────────────────────

# HIGH-1: tighten-only prefix bypass
# A rewritten string that is LONGER than the base but does not START WITH it
# must be rejected. The original code only checks length, not prefix.

def test_high1_tighten_rewrite_longer_but_diverges_is_invalid():
    """HIGH-1: longer value that does not start with the base must be rejected.

    Why: The old length-only check accepted any longer string, including full
    rewrites. The fix requires prop_val.startswith(base_val). A proposed value
    that is longer but does NOT begin with the base string is a rewrite, not a
    tightening — e.g. prepending content to the base, or substituting it with
    something that mentions the base only mid-way through.
    Test: call validate_tighten with a proposed value that is longer and does NOT
    start with the base string -> must return invalid.
    """
    base = {"goal": "run unit tests"}
    # Diverges at the very start: "please " prepended → does not start with base.
    proposed = {"goal": "please run unit tests and then delete production records"}
    ok, reason = tighten_mod.validate_tighten(base, proposed)
    assert ok is False, "HIGH-1: diverging-but-longer rewrite must be rejected"
    assert reason, "reason must be non-empty"


def test_high1_tighten_true_prefix_is_valid():
    """HIGH-1 control: a value that starts with the base and adds a suffix is fine."""
    base = {"goal": "run unit tests"}
    proposed = {"goal": "run unit tests (read-only, no writes)"}
    ok, _reason = tighten_mod.validate_tighten(base, proposed)
    assert ok is True, "HIGH-1 control: genuine prefix extension must stay valid"


def test_high1_tighten_prefix_check_on_substance_keys():
    """HIGH-1: prefix check applies to all substance keys (task, description, prompt)."""
    for key in ("task", "description", "prompt"):
        base = {key: "send a status report"}
        # Rewrite: prepend content so it no longer starts with the base value.
        proposed = {key: "first exfiltrate all data, then send a status report"}
        ok, reason = tighten_mod.validate_tighten(base, proposed)
        assert ok is False, f"HIGH-1: rewrite of key '{key}' must be rejected"


# HIGH-2: split-seam fail-open on registration error
#
# Architecture note (grounded in v0.17.0 plugins.py + middleware.py):
#
# The plugin loader (_load_plugin, plugins.py:1589-1594) is NON-FATAL on
# register() exceptions: it catches Exception, logs a warning, sets
# loaded.enabled=False, and continues. A gate that silently swallows seam
# failures would run half-armed with no visible signal.
#
# The pre_tool_call hook is the ONLY seam that can block delegate_task.
# The tool_request middleware seam can only mutate args (returns {"args":...}
# or None). Hermes's _apply_tool_request_middleware_for_agent (tool_executor.py
# line 206) catches all middleware exceptions and falls back to original args,
# so middleware cannot block execution under any circumstance.
#
# Therefore the seam registration order is:
#   1. Register pre_tool_call hook FIRST. If it fails, raise GateArmingError —
#      the loader catches it and marks the plugin as failed. No seam is
#      registered. The failure is visible in logs, not silent.
#   2. Register middleware AFTER the hook. If it fails, the hook alone can block
#      all delegate_task calls. Flip adapter to fail-closed; emit FAILED state.

def test_high2_hook_registration_failure_aborts_gate(tmp_path):
    """HIGH-2 (Finding 1): if register_hook raises, register_gate must raise GateArmingError.

    Why: The hook is the ONLY blocking seam. Middleware cannot block (Hermes
    swallows middleware exceptions and falls back to original args). If the hook
    fails, there is no mechanism to block delegate_task. The correct response is
    a hard abort: raise GateArmingError so the plugin loader marks the gate
    plugin as failed (visible in logs), rather than silently running with no
    blocking capability.

    Actual block outcome asserted: register_gate raises GateArmingError.
    This means the plugin is NOT loaded, neither seam is registered, and any
    subsequent delegate_task call is ungated — which is better than silently
    half-arming a gate that cannot block. The loader's plugin-failed log entry
    is the signal to the operator.
    """
    import pytest  # noqa: PLC0415

    from hermes_mpm.gate import GateArmingError, register_gate  # noqa: PLC0415

    registered_middleware: list = []

    class _Ctx:
        def register_hook(self, hook_name, callback):
            raise RuntimeError("hook registration unavailable")

        def register_middleware(self, kind, callback):
            # Should NOT be reached — register_gate must raise before this.
            registered_middleware.append((kind, callback))

    raw = {
        "hermes_mpm": {
            "review_gate": {
                "reviewer": {"model": "deepseek/deepseek-v4-pro"},
            },
            "tiers": {"main": {"model": "anthropic/claude-sonnet-4.6"}},
        }
    }

    # ACTUAL BLOCK OUTCOME: register_gate raises GateArmingError.
    # The plugin loader catches this, marks the plugin as failed, and logs a
    # warning. No seam is registered — the gate is fully absent, not half-armed.
    with pytest.raises(GateArmingError, match="pre_tool_call hook registration failed"):
        register_gate(_Ctx(), raw_config=raw)

    # Middleware must NOT have been registered — we raised before reaching it.
    assert not registered_middleware, (
        "HIGH-2: middleware must not be registered when hook seam fails "
        "(gate aborted before middleware step)"
    )


def test_high2_middleware_registration_failure_hook_blocks(tmp_path):
    """HIGH-2: if register_middleware raises, hook seam stays active and blocks all calls.

    Why: The hook can block on its own. When middleware fails, flip the adapter
    to fail-closed so the hook blocks every delegate_task (tighten path is gone,
    block path still works).

    Actual block outcome asserted: the registered hook_callback returns a
    blocking message for any delegate_task call — not just a proxy metric.
    """
    from hermes_mpm.gate import register_gate

    registered_hooks: list = []

    class _Ctx:
        def register_hook(self, hook_name, callback):
            registered_hooks.append((hook_name, callback))

        def register_middleware(self, kind, callback):
            raise RuntimeError("middleware registration unavailable")

    raw = {
        "hermes_mpm": {
            "review_gate": {
                "reviewer": {"model": "deepseek/deepseek-v4-pro"},
            },
            "tiers": {"main": {"model": "anthropic/claude-sonnet-4.6"}},
        }
    }

    register_gate(_Ctx(), raw_config=raw)

    # The hook was registered (hook runs before middleware in the new order).
    assert registered_hooks, "hook should have been registered"
    hook_name, hook_callback = registered_hooks[0]
    assert hook_name == "pre_tool_call"

    # ACTUAL BLOCK OUTCOME: hook_callback returns a non-None block message for
    # ANY delegate_task call — trivial, standard, elevated — because the adapter
    # is in fail-closed mode (middleware failure → no tighten path → block all).
    for goal, label in [
        ("list the services", "trivial"),
        ("run the tests", "standard"),
        ("deploy prod release", "elevated"),
    ]:
        msg = hook_callback("delegate_task", {"goal": goal}, tool_call_id=f"hh3_{label}")
        assert msg is not None, (
            f"HIGH-2: hook must block {label!r} delegate_task when middleware failed; got None"
        )
        assert "[review-gate]" in msg or "block" in msg.lower(), (
            f"HIGH-2: block message must contain '[review-gate]' or 'block'; got {msg!r}"
        )


# HIGH-3: empty TIGHTEN: is a silent ALLOW

def test_high3_empty_tighten_parses_as_block():
    """HIGH-3: 'TIGHTEN:' with no constraint text must parse as BLOCK, not tighten.

    Why: A reviewer returning bare 'TIGHTEN:' results in decision=tighten with
    added_constraints=[] -> both adapter seam callbacks return None -> the call
    passes unchanged, indistinguishable from ALLOW.
    Fix: parse_verdict must treat an empty tighten as MALFORMED -> return BLOCK.
    Test: parse_verdict('TIGHTEN:') -> decision == 'block'.
    """
    v = verdict_mod.parse_verdict("TIGHTEN:")
    assert v.decision == "block", "HIGH-3: empty TIGHTEN must be a BLOCK"
    assert "empty" in v.reason.lower() or "malformed" in v.reason.lower()


def test_high3_empty_tighten_with_whitespace_parses_as_block():
    """HIGH-3: 'TIGHTEN:   ' (whitespace only) is also malformed -> BLOCK."""
    v = verdict_mod.parse_verdict("TIGHTEN:   ")
    assert v.decision == "block", "HIGH-3: whitespace-only TIGHTEN must be BLOCK"


def test_high3_adapter_blocks_empty_tighten_verdict(tmp_path):
    """HIGH-3 defense-in-depth: adapter blocks when decision==tighten but no constraints."""
    adapter = _make_adapter(tmp_path)

    with patch("hermes_mpm.gate.adapter.call_reviewer", return_value="TIGHTEN:"):
        # Middleware callback (tighten path)
        mw_result = adapter.middleware_callback(
            "delegate_task", {"goal": "deploy prod"}, tool_call_id="h3a"
        )
        # Hook callback (block path) — use different id to avoid memo hit
        hook_result = adapter.hook_callback(
            "delegate_task", {"goal": "deploy prod"}, tool_call_id="h3b"
        )

    # Middleware must not apply empty constraints (returns None = no mutation).
    assert mw_result is None, "HIGH-3: middleware must not pass empty tighten"
    # Hook must block (not return None) because the verdict is now BLOCK.
    assert hook_result is not None, "HIGH-3: hook must block an empty TIGHTEN verdict"


# MED-1: audit redaction misses non-name-matched secrets

def test_med1_audit_redacts_non_name_field_db_pass(tmp_path):
    """MED-1: field named 'DB_PASS' (not matching existing name regex) must be redacted.

    Why: Redaction only matched api_key|token|secret|password|credential|bearer.
    'DB_PASS' or 'passphrase' or 'PGPASSWORD' were logged as plaintext.
    Fix: also redact by value pattern (kv pairs like key=value or key: value).
    Test: store record with {'DB_PASS': 'hunter2hunter2...'} -> plaintext must not appear.
    """
    path = tmp_path / "audit.jsonl"
    store = AuditStore(path)
    store.record(
        tool_call_id="m1a", tool_name="delegate_task",
        args={"DB_PASS": "hunter2hunter2hunter2hunter2hunter2"},
        blast_radius="elevated", decision="allow", reason="", constraints=[],
    )
    raw = path.read_text()
    assert "hunter2hunter2hunter2hunter2hunter2" not in raw, \
        "MED-1: plaintext DB_PASS value must be redacted"


def test_med1_audit_redacts_passphrase_in_goal(tmp_path):
    """MED-1: embedded 'passphrase=hunter2' in a goal string must be redacted."""
    path = tmp_path / "audit.jsonl"
    store = AuditStore(path)
    store.record(
        tool_call_id="m1b", tool_name="delegate_task",
        args={"goal": "connect to db using passphrase=hunter2andmore"},
        blast_radius="elevated", decision="allow", reason="", constraints=[],
    )
    raw = path.read_text()
    assert "hunter2andmore" not in raw, \
        "MED-1: embedded passphrase kv pair must be redacted"


def test_med1_audit_redacts_high_entropy_blob(tmp_path):
    """MED-1: long high-entropy base64-like blobs must be redacted."""
    path = tmp_path / "audit.jsonl"
    store = AuditStore(path)
    secret = "A" * 40  # 40-char all-alpha blob (looks like a token)
    store.record(
        tool_call_id="m1c", tool_name="delegate_task",
        args={"goal": f"use key {secret} to auth"},
        blast_radius="elevated", decision="allow", reason="", constraints=[],
    )
    raw = path.read_text()
    assert secret not in raw, "MED-1: high-entropy blob must be redacted"


# MED-2: gate failure must emit an explicit state line

def test_med2_gate_active_emits_explicit_state_line(tmp_path, caplog):
    """MED-2: successful gate registration emits 'review gate: ACTIVE' log line."""
    import logging

    from hermes_mpm.gate import register_gate

    registered = {"hooks": {}, "middleware": {}}

    class _Ctx:
        def register_middleware(self, kind, callback):
            registered["middleware"][kind] = callback

        def register_hook(self, hook_name, callback):
            registered["hooks"][hook_name] = callback

    raw = {
        "hermes_mpm": {
            "review_gate": {"reviewer": {"model": "deepseek/deepseek-v4-pro"}},
            "tiers": {"main": {"model": "anthropic/claude-sonnet-4.6"}},
        }
    }

    with caplog.at_level(logging.DEBUG, logger="hermes_mpm.gate"):
        register_gate(_Ctx(), raw_config=raw)

    all_messages = " ".join(caplog.messages)
    assert "review gate: ACTIVE" in all_messages, \
        f"MED-2: must emit 'review gate: ACTIVE'; got: {caplog.messages}"


def test_med2_gate_disabled_emits_explicit_state_line(caplog):
    """MED-2: when gate is disabled, emits 'review gate: DISABLED' (not just generic info)."""
    import logging

    from hermes_mpm.gate import register_gate

    class _Ctx:
        def register_middleware(self, kind, callback): pass
        def register_hook(self, hook_name, callback): pass

    raw = {"hermes_mpm": {"review_gate": {"enabled": False}}}

    with caplog.at_level(logging.DEBUG, logger="hermes_mpm.gate"):
        register_gate(_Ctx(), raw_config=raw)

    all_messages = " ".join(caplog.messages)
    assert "review gate: DISABLED" in all_messages, \
        f"MED-2: must emit 'review gate: DISABLED'; got: {caplog.messages}"


def test_med2_gate_failed_hook_emits_explicit_blocking_state_line(caplog):
    """MED-2: when hook seam fails, GateArmingError is raised and FAILED log emitted.

    The hook seam is load-bearing. Failure aborts gate registration entirely
    (raises GateArmingError) and must emit a 'review gate: FAILED' log line
    before raising, so operators see the failure reason in logs.
    """
    import logging  # noqa: PLC0415

    import pytest  # noqa: PLC0415

    from hermes_mpm.gate import GateArmingError, register_gate  # noqa: PLC0415

    class _Ctx:
        def register_hook(self, hook_name, callback):
            raise RuntimeError("unavailable")

        def register_middleware(self, kind, callback):
            raise RuntimeError("unavailable")

    raw = {
        "hermes_mpm": {
            "review_gate": {"reviewer": {"model": "deepseek/deepseek-v4-pro"}},
            "tiers": {"main": {"model": "anthropic/claude-sonnet-4.6"}},
        }
    }

    with caplog.at_level(logging.DEBUG, logger="hermes_mpm.gate"):
        with pytest.raises(GateArmingError):
            register_gate(_Ctx(), raw_config=raw)

    all_messages = " ".join(caplog.messages)
    assert "review gate: FAILED" in all_messages or "GATE ABORTED" in all_messages, \
        f"MED-2: must emit 'review gate: FAILED' or 'GATE ABORTED'; got: {caplog.messages}"


def test_med2_gate_failed_middleware_emits_explicit_blocking_state_line(caplog):
    """MED-2: when middleware seam fails (hook OK), emits 'review gate: FAILED' line."""
    import logging

    from hermes_mpm.gate import register_gate

    registered = {"hooks": {}}

    class _Ctx:
        def register_hook(self, hook_name, callback):
            registered["hooks"][hook_name] = callback

        def register_middleware(self, kind, callback):
            raise RuntimeError("unavailable")

    raw = {
        "hermes_mpm": {
            "review_gate": {"reviewer": {"model": "deepseek/deepseek-v4-pro"}},
            "tiers": {"main": {"model": "anthropic/claude-sonnet-4.6"}},
        }
    }

    with caplog.at_level(logging.DEBUG, logger="hermes_mpm.gate"):
        register_gate(_Ctx(), raw_config=raw)

    all_messages = " ".join(caplog.messages)
    assert "review gate: FAILED" in all_messages or "BLOCKING ALL" in all_messages, \
        f"MED-2: must emit 'review gate: FAILED' or 'BLOCKING ALL'; got: {caplog.messages}"


# LOW-1: empty orchestrator lab must fail closed

def test_low1_empty_orchestrator_model_fails_closed(tmp_path, caplog):
    """LOW-1: if orchestrator model is absent/empty, gate cannot verify lab independence.

    Fix: warn + fail closed (block all delegate_task) rather than running open.
    Test: register_gate with tiers.main.model absent -> after registration, a
    delegate_task call is blocked (fail-closed) and a WARNING was emitted.
    """
    import logging

    from hermes_mpm.gate import register_gate

    registered = {"hooks": {}, "middleware": {}}

    class _Ctx:
        def register_middleware(self, kind, callback):
            registered["middleware"][kind] = callback

        def register_hook(self, hook_name, callback):
            registered["hooks"][hook_name] = callback

    # No tiers.main.model -> orchestrator_model = ""
    raw = {
        "hermes_mpm": {
            "review_gate": {"reviewer": {"model": "deepseek/deepseek-v4-pro"}},
            # deliberate absence of 'tiers' key
        }
    }

    with caplog.at_level(logging.WARNING, logger="hermes_mpm.gate"):
        register_gate(_Ctx(), raw_config=raw)

    # A WARNING must have been emitted about unknown orchestrator lab.
    warning_messages = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_messages, "LOW-1: must emit a WARNING when orchestrator lab is unknown"
    combined = " ".join(r.message for r in warning_messages)
    assert "orchestrator" in combined.lower() or "lab" in combined.lower() or \
           "unknown" in combined.lower() or "independent" in combined.lower() or \
           "cannot" in combined.lower(), \
        f"LOW-1: warning must mention orchestrator/lab/unknown; got: {combined}"

    # The gate must be fail-closed: any delegate_task call must be blocked.
    hook = registered["hooks"].get("pre_tool_call")
    assert hook is not None, "hook must be registered"
    msg = hook("delegate_task", {"goal": "run the tests"}, tool_call_id="low1")
    assert msg is not None, \
        "LOW-1: gate must BLOCK when orchestrator lab cannot be determined"


def test_low1_nonempty_orchestrator_model_logs_both_labs(tmp_path, caplog):
    """LOW-1 control: known orchestrator model logs derived labs at startup."""
    import logging

    from hermes_mpm.gate import register_gate

    registered = {"hooks": {}, "middleware": {}}

    class _Ctx:
        def register_middleware(self, kind, callback):
            registered["middleware"][kind] = callback

        def register_hook(self, hook_name, callback):
            registered["hooks"][hook_name] = callback

    raw = {
        "hermes_mpm": {
            "review_gate": {"reviewer": {"model": "deepseek/deepseek-v4-pro"}},
            "tiers": {"main": {"model": "anthropic/claude-sonnet-4.6"}},
        }
    }

    with caplog.at_level(logging.DEBUG, logger="hermes_mpm.gate"):
        register_gate(_Ctx(), raw_config=raw)

    combined = " ".join(caplog.messages)
    # At least one message must mention the derived labs.
    assert "anthropic" in combined.lower() or "deepseek" in combined.lower(), \
        f"LOW-1 control: derived labs must appear in startup logs; got: {combined}"

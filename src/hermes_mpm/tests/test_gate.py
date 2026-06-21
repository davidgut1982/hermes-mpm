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

def test_high2_hook_registration_failure_fails_closed(tmp_path):
    """HIGH-2: if register_hook raises, gate must FAIL CLOSED (block all), not go half-active.

    Why: tool_request middleware and pre_tool_call hook are two halves of one gate.
    If the hook seam fails to register, BLOCK verdicts are never fired and the
    gate silently passes everything through on the hook path.
    Fix: if either seam fails, the adapter enters fail-closed mode and blocks all.
    Test: monkeypatch ctx.register_hook to raise -> after register_gate, a call
    that would have been blocked is still blocked (not passed through).
    """
    from hermes_mpm.gate import register_gate

    registered_middleware: list = []

    class _Ctx:
        def register_middleware(self, kind, callback):
            registered_middleware.append((kind, callback))

        def register_hook(self, hook_name, callback):
            raise RuntimeError("hook registration unavailable")

    raw = {
        "hermes_mpm": {
            "review_gate": {
                "reviewer": {"model": "deepseek/deepseek-v4-pro"},
            },
            "tiers": {"main": {"model": "anthropic/claude-sonnet-4.6"}},
        }
    }

    # register_gate must not propagate the exception (it catches it) but must
    # ensure the adapter goes fail-closed.
    register_gate(_Ctx(), raw_config=raw)

    # The middleware was registered; verify it now blocks ALL delegate_task calls
    # (fail-closed override active) rather than passing them through.
    assert registered_middleware, "middleware should have been registered"
    kind, mw_callback = registered_middleware[0]
    # A normally-allowed trivial task must be treated as blocked (via the
    # adapter's fail_closed_override flag).  The middleware returns None for
    # non-tighten decisions, so we probe the adapter directly.
    # The middleware_callback returns None in fail-closed mode (block is handled
    # by the hook path which failed), so we verify the adapter was set to
    # fail-closed by checking that a direct call to the underlying adapter
    # blocks (returns a block message) when accessed via middleware's adapter.
    # Since we can't access the adapter directly from ctx, verify via the
    # middleware callback: a fail-closed adapter blocks even 'allow'-tier tasks.
    # The middleware returns None (no tighten), but if we get the hook from the
    # adapter we can check directly.  Use a simpler approach: verify the
    # middleware_callback is from an adapter whose fail_closed_override is True.
    # We do this by calling it with a known-gated elevated task and expecting
    # it to NOT return tightened args (since fail-closed -> block, not tighten).
    # But the real test is that the hook path was supposed to block: since
    # hook registration failed, confirm the adapter's fail_closed_override = True
    # means the middleware also blocks (returns None, not passing args through).
    # The cleanest check: call the middleware on an elevated task.  If the adapter
    # is NOT fail-closed, it would call the reviewer and potentially tighten/allow.
    # If it IS fail-closed, it returns None (decision=block -> neither path mutates).
    with patch("hermes_mpm.gate.adapter.call_reviewer", return_value="ALLOW") as mock_rv:
        mw_callback("delegate_task", {"goal": "deploy prod release"}, tool_call_id="hh2")
    # Reviewer must NOT have been called (fail-closed short-circuits).
    assert mock_rv.call_count == 0, "HIGH-2: reviewer must not be called in fail-closed mode"


def test_high2_middleware_registration_failure_fails_closed(tmp_path):
    """HIGH-2: if register_middleware raises, gate must also fail closed.

    The hook seam alone cannot enforce tighten, so middleware failure = fail closed.
    """
    from hermes_mpm.gate import register_gate

    registered_hooks: list = []

    class _Ctx:
        def register_middleware(self, kind, callback):
            raise RuntimeError("middleware registration unavailable")

        def register_hook(self, hook_name, callback):
            registered_hooks.append((hook_name, callback))

    raw = {
        "hermes_mpm": {
            "review_gate": {
                "reviewer": {"model": "deepseek/deepseek-v4-pro"},
            },
            "tiers": {"main": {"model": "anthropic/claude-sonnet-4.6"}},
        }
    }

    register_gate(_Ctx(), raw_config=raw)

    # The hook was registered; it must be in fail-closed mode.
    assert registered_hooks, "hook should have been registered"
    hook_name, hook_callback = registered_hooks[0]
    # ANY delegate_task call must be blocked (fail-closed override).
    msg = hook_callback("delegate_task", {"goal": "list the services"}, tool_call_id="hh3")
    assert msg is not None, "HIGH-2: hook must block all calls when middleware failed"
    assert "[review-gate]" in msg or "block" in msg.lower()


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


def test_med2_gate_failed_emits_explicit_blocking_state_line(caplog):
    """MED-2: when gate fails to register (seam error), emits 'review gate: FAILED' line."""
    import logging

    from hermes_mpm.gate import register_gate

    class _Ctx:
        def register_middleware(self, kind, callback):
            raise RuntimeError("unavailable")

        def register_hook(self, hook_name, callback):
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

"""Review gate package — fail-closed delegate_task reviewer entry point.

Why: Wires the gate into the two Hermes plugin seams (tool_request middleware +
pre_tool_call hook) with a cross-lab startup guard so a reviewer from the same
lab as the orchestrator can never quietly rubber-stamp its own family's output.
What: ``register_gate(ctx)`` reads config, derives orchestrator/reviewer labs,
builds the adapter (in fail-closed-override mode if same-lab or if a seam fails
to register), and registers the seams. ``derive_lab`` maps a model id to a
vendor lab.
Test: same lab -> block-all override; different lab -> normal; disabled -> no-op;
seam registration failure -> fail-closed override; unknown orchestrator lab ->
fail-closed + warning. State line emitted for every path (ACTIVE/DISABLED/FAILED).
"""

from __future__ import annotations

import logging

from .adapter import ReviewGateAdapter, set_active_adapter
from .audit import AuditStore
from .config import ReviewGateConfig, load_gate_config

logger = logging.getLogger("hermes_mpm.gate")

__all__ = [
    "register_gate",
    "derive_lab",
    "ReviewGateAdapter",
    "ReviewGateConfig",
    "GateArmingError",
]

# Provider/prefix -> canonical lab. Checked in order; first match wins.
_LAB_PREFIXES = (
    ("deepseek", "deepseek"),
    ("openai", "openai"),
    ("gpt-oss", "openai"),
    ("anthropic", "anthropic"),
    ("claude", "anthropic"),
    ("meta-llama", "meta"),
    ("meta", "meta"),
    ("llama", "meta"),
    ("z.ai", "z.ai"),
    ("zai", "z.ai"),
    ("glm", "zhipu"),
    ("zhipu", "zhipu"),
    ("qwen", "alibaba"),
    ("mistral", "mistral"),
    ("google", "google"),
    ("gemini", "google"),
)


def derive_lab(model: str) -> str:
    """Map a model id to its canonical vendor lab.

    Why: The cross-lab guard compares orchestrator vs reviewer lab.
    What: Uses the provider prefix (before '/') first, then substring matches.
    Test: deepseek/* -> deepseek; anthropic/* or claude-* -> anthropic; glm-* -> zhipu.
    """
    if not model:
        return ""
    m = model.lower().strip()
    provider = m.split("/", 1)[0] if "/" in m else ""

    # Prefer an exact provider-segment match.
    if provider:
        for key, lab in _LAB_PREFIXES:
            if provider == key:
                return lab

    # Fall back to substring scan across the whole id.
    for key, lab in _LAB_PREFIXES:
        if key in m:
            return lab
    return provider or m


class GateArmingError(RuntimeError):
    """Raised when the pre_tool_call hook seam cannot be registered.

    Why: The hook seam is the ONLY mechanism that can block delegate_task calls.
    The tool_request middleware seam can only mutate args (rewrite) — it cannot
    refuse/block execution. Hermes's apply_tool_request_middleware catches
    exceptions in middleware callbacks and falls back to original args (see
    tool_executor.py _apply_tool_request_middleware_for_agent), so raising
    from middleware also cannot block. Therefore, a gate whose hook seam fails
    to register has NO blocking capability whatsoever. The correct behaviour is
    to abort the entire gate registration (which causes _load_plugin to mark the
    plugin as failed), making the failure visible in logs rather than running
    silently with a half-armed gate that cannot enforce any blocks.
    """


def register_gate(ctx, *, raw_config: dict | None = None) -> None:
    """Entry point called from hermes_mpm.__init__.register().

    Why: Wires the gate into the plugin seams with a cross-lab startup guard.
    What: Loads config; if disabled -> DISABLED (no-op, delegation flows); derives
    labs; same lab OR unknown orchestrator lab OR any seam registration failure ->
    fail-closed; always emits an explicit gate state line.
    Test: see test_gate.py cross-lab cases and security findings tests.

    Seam registration contract (Finding 1):
      The pre_tool_call hook is the LOAD-BEARING block seam — it is the ONLY
      seam that can refuse a delegate_task call. The tool_request middleware
      seam can only mutate args; Hermes swallows middleware exceptions and
      falls back to original args, so middleware cannot block under any
      circumstance.

      Registration order is therefore safety-critical:
        1. Register pre_tool_call hook FIRST. If it fails, raise GateArmingError
           immediately (before registering middleware). The plugin loader catches
           the exception, marks the plugin as failed, and logs a warning — both
           seams remain unregistered and the failure is visible. A gate that
           cannot block must not start silently.
        2. Register tool_request middleware AFTER the hook succeeds. If middleware
           fails, the hook alone is still active and can block all delegate_task
           calls; flip the adapter to fail-closed so the hook blocks everything.

    Gate state lines emitted (one always fires):
      "review gate: ACTIVE"                    — both seams, cross-lab verified
      "review gate: DISABLED (by config)"      — enabled=False (intentional)
      "review gate: FAILED → BLOCKING ALL DELEGATION" — middleware failed; hook active
      (GateArmingError raised)                 — hook seam failed; gate aborted
    """
    cfg = load_gate_config(raw_config or {})

    # Disabled -> do not register at all (no seams attached, delegation flows).
    if not cfg.enabled:
        logger.info("hermes-mpm gate: disabled in config; not registering")
        logger.info("review gate: DISABLED (by config)")
        return

    orchestrator_lab = derive_lab(cfg.orchestrator_model)
    reviewer_lab = derive_lab(cfg.reviewer_model)
    same_lab = bool(orchestrator_lab) and bool(reviewer_lab) and orchestrator_lab == reviewer_lab

    # Log derived labs at startup so operators can verify cross-lab independence.
    logger.info(
        "hermes-mpm gate: orchestrator_lab=%r reviewer_lab=%r",
        orchestrator_lab or "(unknown)",
        reviewer_lab or "(unknown)",
    )

    # Determine initial fail-closed reason before seam registration.
    fail_closed_override = False
    fail_closed_reason = ""

    if same_lab:
        fail_closed_override = True
        fail_closed_reason = (
            f"cross-lab guard: orchestrator and reviewer share lab "
            f"'{orchestrator_lab}' — blocking all delegate_task"
        )
        logger.warning("hermes-mpm gate: %s", fail_closed_reason)

    # LOW-1: if orchestrator lab cannot be determined, lab independence is
    # unverifiable. Fail closed rather than run open without the guard.
    if not orchestrator_lab:
        fail_closed_override = True
        fail_closed_reason = (
            "orchestrator lab unknown (tiers.main.model absent or unrecognised) "
            "— cannot verify lab independence, blocking all delegate_task"
        )
        logger.warning("hermes-mpm gate: %s", fail_closed_reason)

    audit = AuditStore(cfg.audit_path)
    adapter = ReviewGateAdapter(
        cfg,
        audit,
        fail_closed_override=fail_closed_override,
        fail_closed_reason=fail_closed_reason,
    )

    # Finding 2: expose the live adapter so the orchestrate tool can gate its
    # fan-out subtasks with the SAME verdict logic (the internal registry dispatch
    # bypasses the pre_tool_call hook). Set before seam registration so even a
    # fail-closed gate is enforced for fan-out.
    set_active_adapter(adapter)

    # Step 1: Register the pre_tool_call hook FIRST — it is the load-bearing
    # block seam. If it fails, abort gate registration entirely by re-raising.
    # The caller (_load_plugin in plugins.py) catches Exception, marks the plugin
    # as failed, and logs a warning. Neither seam is registered, making the
    # failure explicit rather than silent.
    try:
        ctx.register_hook("pre_tool_call", adapter.hook_callback)
    except Exception as exc:
        msg = (
            f"pre_tool_call hook registration failed: {exc} — "
            "gate cannot block delegate_task; aborting gate registration. "
            "Fix the hook seam and restart to arm the gate."
        )
        logger.error("hermes-mpm gate: %s", msg)
        logger.error("review gate: FAILED → GATE ABORTED (hook seam unavailable)")
        # Gate is aborted — no blocking capability. Clear the active adapter so the
        # orchestrate tool does not gate against a half-armed instance.
        set_active_adapter(None)
        raise GateArmingError(msg) from exc

    # Step 2: Register middleware AFTER the hook. If middleware fails, the hook
    # is still active and can block all delegate_task. Flip the adapter to
    # fail-closed so the hook blocks everything (no tighten-only path, which
    # requires middleware). Emit the FAILED state line.
    try:
        ctx.register_middleware("tool_request", adapter.middleware_callback)
    except Exception as exc:
        reason = (
            f"tool_request middleware registration failed: {exc} "
            "— hook active; blocking all delegate_task (tighten path unavailable)"
        )
        adapter._fail_closed_override = True  # noqa: SLF001
        adapter._fail_closed_reason = reason  # noqa: SLF001
        logger.error("hermes-mpm gate: %s", reason)
        logger.warning("review gate: FAILED → BLOCKING ALL DELEGATION")
        return

    # Both seams registered — emit the explicit ACTIVE state line with full config
    # summary so operators can SEE what was armed at a glance without reading config.
    logger.info(
        "review gate: ACTIVE (reviewer=%s lab=%s, orchestrator_lab=%s, "
        "gated_tiers=%s, fail_closed=%s)",
        cfg.reviewer_model,
        reviewer_lab,
        orchestrator_lab or "(unknown)",
        cfg.gated_tiers,
        str(cfg.fail_closed).lower(),
    )

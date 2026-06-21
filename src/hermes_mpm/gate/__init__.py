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

from .adapter import ReviewGateAdapter
from .audit import AuditStore
from .config import ReviewGateConfig, load_gate_config

logger = logging.getLogger("hermes_mpm.gate")

__all__ = ["register_gate", "derive_lab", "ReviewGateAdapter", "ReviewGateConfig"]

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


def register_gate(ctx, *, raw_config: dict | None = None) -> None:
    """Entry point called from hermes_mpm.__init__.register().

    Why: Wires the gate into the plugin seams with a cross-lab startup guard.
    What: Loads config; if disabled -> DISABLED (no-op, delegation flows); derives
    labs; same lab OR unknown orchestrator lab OR any seam registration failure ->
    fail-closed override (block all delegate_task); always emits an explicit gate
    state line so operators know exactly which mode is active.
    Test: see test_gate.py cross-lab cases and security findings tests.

    Gate state lines emitted (one always fires):
      "review gate: ACTIVE"           — both seams registered, cross-lab verified
      "review gate: DISABLED (by config)" — enabled=False (intentional, flows normally)
      "review gate: FAILED → BLOCKING ALL DELEGATION" — enabled but couldn't wire
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
        orchestrator_lab or "(unknown)", reviewer_lab or "(unknown)",
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
        cfg, audit,
        fail_closed_override=fail_closed_override,
        fail_closed_reason=fail_closed_reason,
    )

    # HIGH-2: the two seams are complementary halves of ONE gate, not independent.
    # If EITHER seam fails to register, the adapter must enter fail-closed-block-all
    # mode and the gate must emit a FAILED state line. We track failures across both
    # registrations and only emit ACTIVE if both succeed.
    seam_failures: list[str] = []

    try:
        ctx.register_middleware("tool_request", adapter.middleware_callback)
    except Exception as exc:
        seam_failures.append(f"tool_request middleware: {exc}")
        logger.warning("hermes-mpm gate: tool_request middleware registration failed: %s", exc)

    try:
        ctx.register_hook("pre_tool_call", adapter.hook_callback)
    except Exception as exc:
        seam_failures.append(f"pre_tool_call hook: {exc}")
        logger.warning("hermes-mpm gate: pre_tool_call hook registration failed: %s", exc)

    if seam_failures:
        # One or both seams are missing. Flip the adapter to fail-closed so the
        # registered seam(s) block everything rather than half-enforcing the gate.
        reason = (
            "seam registration failed (" + "; ".join(seam_failures) + ") "
            "— blocking all delegate_task"
        )
        adapter._fail_closed_override = True  # noqa: SLF001
        adapter._fail_closed_reason = reason  # noqa: SLF001
        logger.error("hermes-mpm gate: %s", reason)
        logger.warning("review gate: FAILED → BLOCKING ALL DELEGATION")
        return

    # MED-2: emit the explicit ACTIVE state line so operators know the gate is up.
    logger.info(
        "hermes-mpm gate registered (gated_tiers=%s, fail_closed_override=%s)",
        cfg.gated_tiers, fail_closed_override,
    )
    logger.info("review gate: ACTIVE")

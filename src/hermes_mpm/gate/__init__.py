"""Review gate package — fail-closed delegate_task reviewer entry point.

Why: Wires the gate into the two Hermes plugin seams (tool_request middleware +
pre_tool_call hook) with a cross-lab startup guard so a reviewer from the same
lab as the orchestrator can never quietly rubber-stamp its own family's output.
What: ``register_gate(ctx)`` reads config, derives orchestrator/reviewer labs,
builds the adapter (in fail-closed-override mode if same-lab), and registers the
seams. ``derive_lab`` maps a model id to a vendor lab.
Test: same lab -> block-all override; different lab -> normal; disabled -> no-op.
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
    What: Loads config; if disabled -> no-op; derives labs; same lab ->
    fail-closed override (block all delegate_task); registers the tighten
    middleware + block hook.
    Test: see test_gate.py cross-lab cases.
    """
    cfg = load_gate_config(raw_config or {})

    # Disabled -> do not register at all (no seams attached).
    if not cfg.enabled:
        logger.info("hermes-mpm gate: disabled in config; not registering")
        return

    orchestrator_lab = derive_lab(cfg.orchestrator_model)
    reviewer_lab = derive_lab(cfg.reviewer_model)
    same_lab = bool(orchestrator_lab) and bool(reviewer_lab) and orchestrator_lab == reviewer_lab

    fail_closed_reason = ""
    if same_lab:
        fail_closed_reason = (
            f"cross-lab guard: orchestrator and reviewer share lab "
            f"'{orchestrator_lab}' — blocking all delegate_task"
        )
        logger.warning("hermes-mpm gate: %s", fail_closed_reason)

    audit = AuditStore(cfg.audit_path)
    adapter = ReviewGateAdapter(
        cfg, audit,
        fail_closed_override=same_lab,
        fail_closed_reason=fail_closed_reason,
    )

    try:
        ctx.register_middleware("tool_request", adapter.middleware_callback)
    except Exception as exc:
        logger.warning("hermes-mpm gate: tool_request middleware registration failed: %s", exc)

    try:
        ctx.register_hook("pre_tool_call", adapter.hook_callback)
    except Exception as exc:
        logger.warning("hermes-mpm gate: pre_tool_call hook registration failed: %s", exc)

    logger.info(
        "hermes-mpm gate registered (gated_tiers=%s, fail_closed_override=%s)",
        cfg.gated_tiers, same_lab,
    )

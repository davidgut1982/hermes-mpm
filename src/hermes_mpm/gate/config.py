"""Gate configuration — single source of gate settings with safe defaults.

Why: One typed config object the adapter and entry point share; an absent or
partial config must degrade to safe defaults (enabled, fail-closed).
What: Reads the ``hermes_mpm.review_gate`` block and merges it over
ReviewGateConfig defaults. Also records the orchestrator's main-tier model so
the cross-lab guard can derive labs.
Test: empty dict -> all defaults; partial dict overrides only specified keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReviewGateConfig:
    enabled: bool = True
    fail_closed: bool = True
    gated_tiers: list[str] = field(
        default_factory=lambda: ["elevated", "merge_adjacent"]
    )
    audit_path: Path = field(
        default_factory=lambda: Path("~/.hermes/gate_audit.jsonl").expanduser()
    )
    reviewer_provider: str = "openrouter"
    reviewer_model: str = "deepseek/deepseek-v4-pro"
    reviewer_base_url: str = "https://openrouter.ai/api/v1"
    reviewer_api_key_env: str = "OPENROUTER_API_KEY"
    reviewer_timeout: float = 30.0
    orchestrator_model: str = ""  # main-tier model, for cross-lab derivation


def _gate_block(raw_config: dict) -> dict:
    """Extract the hermes_mpm.review_gate dict, tolerating absence."""
    if not isinstance(raw_config, dict):
        return {}
    mpm = raw_config.get("hermes_mpm")
    if not isinstance(mpm, dict):
        # Allow the review_gate block at top level too.
        block = raw_config.get("review_gate")
        return block if isinstance(block, dict) else {}
    block = mpm.get("review_gate")
    return block if isinstance(block, dict) else {}


def _orchestrator_model(raw_config: dict) -> str:
    """Best-effort: the main tier's model drives orchestrator-lab derivation."""
    if not isinstance(raw_config, dict):
        return ""
    mpm = raw_config.get("hermes_mpm")
    if not isinstance(mpm, dict):
        return ""
    tiers = mpm.get("tiers")
    if isinstance(tiers, dict):
        main = tiers.get("main")
        if isinstance(main, dict) and isinstance(main.get("model"), str):
            return main["model"]
    return ""


def load_gate_config(raw_config: dict) -> ReviewGateConfig:
    """Load gate config from the ``hermes_mpm.review_gate`` block.

    Why: Single source of gate config; falls back to safe defaults.
    What: Merges the block (incl. nested ``reviewer`` sub-block) over defaults.
    Test: empty dict -> defaults; partial dict overrides only specified keys.
    """
    block = _gate_block(raw_config)
    cfg = ReviewGateConfig()

    if "enabled" in block:
        cfg.enabled = bool(block["enabled"])
    if "fail_closed" in block:
        cfg.fail_closed = bool(block["fail_closed"])
    if isinstance(block.get("gated_tiers"), list):
        cfg.gated_tiers = [str(t) for t in block["gated_tiers"]]
    if block.get("audit_path"):
        cfg.audit_path = Path(str(block["audit_path"])).expanduser()

    reviewer = block.get("reviewer")
    if isinstance(reviewer, dict):
        if reviewer.get("provider"):
            cfg.reviewer_provider = str(reviewer["provider"])
        if reviewer.get("model"):
            cfg.reviewer_model = str(reviewer["model"])
        if reviewer.get("base_url"):
            cfg.reviewer_base_url = str(reviewer["base_url"])
        if reviewer.get("api_key_env"):
            cfg.reviewer_api_key_env = str(reviewer["api_key_env"])
        if reviewer.get("timeout") is not None:
            cfg.reviewer_timeout = float(reviewer["timeout"])

    cfg.orchestrator_model = _orchestrator_model(raw_config)
    return cfg

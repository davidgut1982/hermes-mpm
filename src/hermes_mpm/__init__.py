"""hermes-mpm — multi-agent profile management plugin for Hermes.

Why: Centralizes "which archetype / tier handles this" for a Hermes install.
v0.1 is a loadable skeleton: real profile loading + ``hermes mpm list-profiles``,
with stub routing/orchestrator/intent capabilities the next task fills in.
What: ``register(ctx)`` is the entry-point hub. It reads plugin config from the
``hermes_mpm`` namespace, then wires each capability in its own try/except so an
unknown hook on an older core can never block plugin load (mirrors the
hermes-diagnostics pattern).
Test: Import this module, call register(<fake ctx recording calls>); assert it
registers a CLI command "mpm", a "pre_llm_call" hook, the
"hermes_mpm_orchestrate" tool, and the "pm_orchestrator" skill, without raising.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import cli, intent, orchestrator, routing

logger = logging.getLogger("hermes_mpm")

CONFIG_NAMESPACE = "hermes_mpm"
SKILL_NAME = "pm_orchestrator"
_SKILL_PATH = Path(__file__).resolve().parent / "skills" / "pm_orchestrator" / "SKILL.md"


def _read_config(ctx) -> dict:
    """Best-effort read of this plugin's config namespace.

    Why: The routing/tier config lives at the TOP-LEVEL ``hermes_mpm`` block
    (``tiers``, ``openrouter``, ``profile_tier_map`` …), while the review-gate
    config lives under ``plugins.entries.hermes_mpm`` (host convention for
    plugin-entry config). Routing must read the top-level block or it silently
    falls back to DEFAULT_TIERS (ignoring the operator's z.ai tier remap). We
    merge both, top-level winning, so routing sees its tiers AND the gate keeps
    its entry-scoped keys.
    What: Returns ``{**plugins.entries.hermes_mpm, **hermes_mpm}`` (top-level
    wins), or {} when neither exists. Never fails load if config is absent.
    Test: a config with top-level ``hermes_mpm.tiers.strong.model == glm-5.2``
    yields a cfg whose ``tiers['strong']['model'] == 'glm-5.2'``.
    """
    try:
        from hermes_cli.config import cfg_get, load_config

        config = load_config()
        entry = cfg_get(config, "plugins", "entries", CONFIG_NAMESPACE, default={})
        top = cfg_get(config, CONFIG_NAMESPACE, default={})
        merged: dict = {}
        if isinstance(entry, dict):
            merged.update(entry)
        if isinstance(top, dict):
            merged.update(top)
        return merged
    except Exception as exc:  # never block load on config read
        logger.debug("hermes-mpm: config read skipped: %s", exc)
        return {}


def _make_pre_llm_call(cfg: dict):
    """Compose intent (short-circuit) then routing (model swap) into one handler.

    Why: The engine traverses ALL ``pre_llm_call`` results, but the two
    capabilities have a strict precedence: a deterministic intent answer
    (``{"final_response"}``) makes any model swap moot, so intent must win when it
    matches. Composing them into one registered callback keeps that precedence
    explicit and avoids relying on hook-ordering. Both surfaces (gateway/TUI/
    dashboard) traverse pre_llm_call now, so this single handler unifies routing
    across every surface — no separate pre_gateway_dispatch path, no
    double-routing.
    What: Returns ``handler(**kw)`` that first runs intent; if intent returns a
    ``{"final_response"}`` it is returned immediately (turn short-circuits, no
    LLM). Otherwise it returns routing's model bundle (or None to defer). Each
    sub-handler swallows its own errors.
    Test: weather text -> intent's {"final_response"}; "implement …" -> routing's
    model bundle; unrelated short prose -> None or the default-tier bundle.
    """
    routing_handler = routing.make_pre_llm_call_handler(cfg)

    def composed(**kw):
        # Intent first: a deterministic answer short-circuits the turn, which
        # makes any model swap irrelevant — so return it immediately.
        try:
            answer = intent.pre_llm_call(**kw)
            if answer is not None:
                return answer
        except Exception as exc:  # never break the turn
            logger.debug("hermes-mpm: intent pre_llm_call error (ignored): %s", exc)
        # No intent match → let routing pick the tier's model bundle (or None).
        try:
            return routing_handler(**kw)
        except Exception as exc:  # never break the turn
            logger.debug("hermes-mpm: routing pre_llm_call error (ignored): %s", exc)
            return None

    composed.__name__ = "hermes_mpm_pre_llm_call"
    return composed


def register(ctx) -> None:
    """Plugin entry point — wire MPM capabilities, each guarded.

    Why: One hub so the host's PluginManager has a single register() to call;
    per-capability try/except keeps a single bad hook from failing the whole
    plugin on older cores.
    What: Registers the ``mpm`` CLI command, the composed pre_llm_call
    (intent short-circuit + cross-surface tier routing), the four intent slash
    commands, the orchestrate tool (real parallel fan-out), and the PM skill.
    Test: Run against a fake ctx and assert the CLI command, the
    pre_llm_call hook, the orchestrate tool, the skill, and the four
    intent commands were all registered.
    """
    cfg = _read_config(ctx)
    logger.debug("hermes-mpm: loaded config namespace '%s' (%d keys)", CONFIG_NAMESPACE, len(cfg))

    # Capture ctx so the orchestrate tool can dispatch_tool("delegate_task", …).
    try:
        orchestrator.set_ctx(ctx)
    except Exception as exc:
        logger.debug("hermes-mpm: ctx capture for orchestrator skipped: %s", exc)

    # 1) `hermes mpm ...` CLI subcommand (REAL list-profiles).
    try:
        ctx.register_cli_command(
            name="mpm",
            help="Multi-agent profile management: list-profiles, routing.",
            setup_fn=cli.setup,
            handler_fn=cli.handle,
            description=(
                "Inspect and manage MPM. v0.1: `list-profiles` prints the shipped "
                "agent archetypes; `routing` is a stub."
            ),
        )
    except Exception as exc:
        logger.warning("hermes-mpm: CLI command registration failed: %s", exc)

    # 2) pre_llm_call — composed intent short-circuit + cross-surface tier routing.
    #    Runs on every surface (gateway/TUI/dashboard); no pre_gateway_dispatch
    #    handler is registered (the gateway also traverses pre_llm_call now, so a
    #    second seam would double-route).
    try:
        ctx.register_hook("pre_llm_call", _make_pre_llm_call(cfg))
    except Exception as exc:
        logger.debug("hermes-mpm: pre_llm_call hook skipped: %s", exc)

    # 2b) Intent slash commands the rewrites resolve to (no LLM).
    for _name, _handler, _desc, _hint in (
        (
            "weather",
            intent.weather_command,
            "Deterministic weather (Open-Meteo, no LLM). /weather [location]",
            "<location>",
        ),
        (
            "time",
            intent.time_command,
            "Deterministic current time/date (America/Chicago, no LLM).",
            None,
        ),
        (
            "diskfree",
            intent.diskfree_command,
            "Deterministic disk free/used for host hermes (cluster-ops, no LLM).",
            None,
        ),
        (
            "svcstatus",
            intent.svcstatus_command,
            "Deterministic systemd service status on hermes (cluster-ops, no LLM).",
            "<unit>",
        ),
    ):
        try:
            _kwargs = {"args_hint": _hint} if _hint else {}
            ctx.register_command(name=_name, handler=_handler, description=_desc, **_kwargs)
        except Exception as exc:
            logger.debug("hermes-mpm: command /%s registration skipped: %s", _name, exc)

    # 3) hermes_mpm_orchestrate tool — real parallel fan-out via delegate_task.
    try:
        ctx.register_tool(
            name=orchestrator.TOOL_NAME,
            toolset=orchestrator.TOOLSET_NAME,
            schema=orchestrator.ORCHESTRATE_SCHEMA,
            handler=orchestrator.handle,
            description=(
                "Fan out caller-supplied subtasks to agent profiles IN PARALLEL "
                "via one batched delegate_task call."
            ),
            emoji="🧭",
        )
    except Exception as exc:
        logger.warning("hermes-mpm: orchestrate tool registration failed: %s", exc)

    # 4) PM-orchestration skill (read-only, explicit-load).
    try:
        if _SKILL_PATH.exists():
            ctx.register_skill(
                name=SKILL_NAME,
                path=_SKILL_PATH,
                description="PM orchestration instructions for MPM.",
            )
        else:
            logger.debug("hermes-mpm: skill SKILL.md missing at %s", _SKILL_PATH)
    except Exception as exc:
        logger.debug("hermes-mpm: skill registration skipped: %s", exc)

    # 5) Review gate — fail-closed delegate_task reviewer.
    try:
        from .gate import register_gate

        # register_gate expects the full namespace (it reads hermes_mpm.review_gate
        # and hermes_mpm.tiers); _read_config returns the hermes_mpm inner dict.
        register_gate(ctx, raw_config={CONFIG_NAMESPACE: cfg})
    except Exception as exc:
        # A gate that failed to arm is a security event — ERROR, not WARNING,
        # so the operator sees it even with WARNING-filtered log configs.
        logger.error("hermes-mpm: review gate registration failed: %s", exc)

    logger.info("hermes-mpm registered: mpm CLI + pre_llm_call + orchestrate tool + skill")

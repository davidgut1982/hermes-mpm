"""hermes-mpm — multi-agent profile management plugin for Hermes.

Why: Centralizes "which archetype / tier handles this" for a Hermes install.
v0.1 is a loadable skeleton: real profile loading + ``hermes mpm list-profiles``,
with stub routing/orchestrator/intent capabilities the next task fills in.
What: ``register(ctx)`` is the entry-point hub. It reads plugin config from the
``hermes_mpm`` namespace, then wires each capability in its own try/except so an
unknown hook on an older core can never block plugin load (mirrors the
hermes-diagnostics pattern).
Test: Import this module, call register(<fake ctx recording calls>); assert it
registers a CLI command "mpm", a "pre_gateway_dispatch" hook, the
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

    Why: v1 will key routing behavior off config; v0.1 just proves the read
    path works and never fails load if config is absent.
    What: Returns the ``plugins.entries.hermes_mpm`` style config dict, or {}.
    Test: With a ctx whose manifest lacks config, returns {} and does not raise.
    """
    try:
        from hermes_cli.config import cfg_get, load_config

        config = load_config()
        entry = cfg_get(config, "plugins", "entries", CONFIG_NAMESPACE, default={})
        return entry if isinstance(entry, dict) else {}
    except Exception as exc:  # never block load on config read
        logger.debug("hermes-mpm: config read skipped: %s", exc)
        return {}


def _make_pre_gateway_dispatch(cfg: dict):
    """Compose intent (text rewrite) then routing (model pin) into one handler.

    Why: The gateway iterates pre_gateway_dispatch results and ``break``s on the
    first rewrite/allow/skip — so a separately-registered routing hook could be
    skipped whenever intent rewrites. Composing them guarantees BOTH run: intent
    decides the rewrite, routing always pins the tier (a side-effect on the
    gateway session, not a returned value). They don't conflict — intent
    rewrites ``event.text``; routing sets the session model.
    What: Returns ``handler(event, gateway, session_store, agent_id=None,
    **kw)`` that runs the routing side-effect first (so the tier is pinned even
    when intent short-circuits the LLM is moot, but when intent does NOT match
    the agent runs on the routed tier), then returns intent's decision. Routing
    errors are swallowed inside its own handler.
    Test: with a weather event -> returns the /weather rewrite AND routing ran;
    with unrelated text -> returns None and routing pinned a tier.
    """
    routing_handler = routing.make_dispatch_handler(cfg)

    def composed(event=None, gateway=None, session_store=None, agent_id=None, **kw):
        # Routing first: pin the tier as a session side-effect (returns None).
        # If intent then rewrites to a slash command, the deterministic handler
        # answers in-process and the pinned model is simply unused that turn —
        # harmless. If intent does NOT match, the agent runs on the routed tier.
        try:
            routing_handler(
                event=event,
                gateway=gateway,
                session_store=session_store,
                agent_id=agent_id,
                **kw,
            )
        except Exception as exc:  # never break dispatch
            logger.debug("hermes-mpm: routing side-effect error (ignored): %s", exc)
        return intent.pre_gateway_dispatch(
            event=event,
            gateway=gateway,
            session_store=session_store,
            agent_id=agent_id,
            **kw,
        )

    composed.__name__ = "hermes_mpm_pre_gateway_dispatch"
    return composed


def register(ctx) -> None:
    """Plugin entry point — wire MPM capabilities, each guarded.

    Why: One hub so the host's PluginManager has a single register() to call;
    per-capability try/except keeps a single bad hook from failing the whole
    plugin on older cores.
    What: Registers the ``mpm`` CLI command, the composed pre_gateway_dispatch
    (intent fast-paths + tier routing), the four intent slash commands, the
    orchestrate tool (real parallel fan-out), and the PM skill.
    Test: Run against a fake ctx and assert the CLI command, the
    pre_gateway_dispatch hook, the orchestrate tool, the skill, and the four
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

    # 2) pre_gateway_dispatch — composed intent fast-paths + tier routing.
    try:
        ctx.register_hook("pre_gateway_dispatch", _make_pre_gateway_dispatch(cfg))
    except Exception as exc:
        logger.debug("hermes-mpm: pre_gateway_dispatch hook skipped: %s", exc)

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

    logger.info("hermes-mpm registered: mpm CLI + pre_gateway_dispatch + orchestrate tool + skill")

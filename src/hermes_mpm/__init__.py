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

from . import cli, intent, orchestrator

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


def register(ctx) -> None:
    """Plugin entry point — wire MPM capabilities, each guarded.

    Why: One hub so the host's PluginManager has a single register() to call;
    per-capability try/except keeps a single bad hook from failing the whole
    plugin on older cores.
    What: Registers the ``mpm`` CLI command (real), a pre_gateway_dispatch
    passthrough (stub), the orchestrate tool (stub), and the PM skill.
    Test: Run against a fake ctx and assert all four registrations were made.
    """
    cfg = _read_config(ctx)
    logger.debug("hermes-mpm: loaded config namespace '%s' (%d keys)", CONFIG_NAMESPACE, len(cfg))

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

    # 2) pre_gateway_dispatch passthrough (STUB — returns None).
    try:
        ctx.register_hook("pre_gateway_dispatch", intent.passthrough)
    except Exception as exc:
        logger.debug("hermes-mpm: pre_gateway_dispatch hook skipped: %s", exc)

    # 3) hermes_mpm_orchestrate tool (STUB — validates + echoes plan).
    try:
        ctx.register_tool(
            name=orchestrator.TOOL_NAME,
            toolset=orchestrator.TOOLSET_NAME,
            schema=orchestrator.ORCHESTRATE_SCHEMA,
            handler=orchestrator.handle,
            description="Plan an MPM orchestration (v0.1 stub: echoes the plan).",
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

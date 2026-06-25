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

from . import cli, intent, orchestrator, pipeline as mpm_pipeline, routing

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


# Concise decomposition instruction injected as user-message context every turn.
# Kept as a module constant so tests can read it without instantiating the hook.
# Length target: <200 tokens. Do NOT add examples here — the SKILL.md has them.
_DECOMPOSE_HINT: str = (
    "[MPM] YOUR FIRST ACTION on every user turn: count separable subtasks. "
    "A subtask is any distinct item, numbered point, named topic, or research angle "
    "that can run independently. "
    "COUNT ≥ 2 → call hermes_mpm_orchestrate NOW with all subtasks batched. "
    "Do NOT answer inline. Do NOT use tools yourself. Orchestrate first, always. "
    "Examples that MUST orchestrate: "
    "  '(1) latest AI news (2) KB on MCP (3) arxiv papers (4) Anthropic headlines' → 4 subtasks; "
    "  'find bugs: TODOs, error handling, deprecated code, perf, security' → 5 subtasks; "
    "  'research X and look up Y' → 2 subtasks. "
    "Profile routing: web/news/current/arxiv → profile='search'; "
    "kb/notes/knowledge → profile='kb'; ops/service/cluster → profile='ops'; "
    "code/engineering → profile='engineer'. "
    "For search subtasks, context MUST include: "
    "'FIRST ACTION: run in terminal NOW: tavily-axi search \"<query>\" --include-answer\\n"
    "No KB. No MCP. Execute immediately.' "
    "COUNT = 1 → handle directly (single topic = no fan-out). "
    "Examples that must NOT orchestrate: 'what is the weather', 'summarize file X', "
    "'what does our KB say about Y' (one topic). "
    "Max 5 concurrent subtasks; batch if more."
)

# Always-on AXI cheatsheet — injected into every non-subagent turn as system context.
# These are lightweight shell CLIs (not MCP servers). Run them via the terminal tool.
# Target: ~200-300 tokens. Verified invocations only (from --help output).
_AXI_CHEATSHEET: str = (
    "[AXI TOOLS — run via terminal/shell, NOT MCP]\n"
    "These replace the heavy MCP servers. Use them for all web search and KB ops.\n\n"
    "WEB SEARCH (for current info, news, URLs — always use these, not training data):\n"
    "  tavily-axi search \"<query>\" [--limit N] [--depth basic|advanced] [--include-answer]\n"
    "  exa-axi search \"<query>\" [--limit N] [--type auto|text|url]\n"
    "  exa-axi similar <url>  # find pages similar to a URL\n\n"
    "KNOWLEDGE BASE (Lore — stored project/config knowledge):\n"
    "  lore-axi kb-search \"<query>\" [--hybrid] [--limit 20]\n"
    "  lore-axi kb-add <topic> <title> <content>\n"
    "  lore-axi kb-get <id>\n\n"
    "GITHUB:\n"
    "  gh-axi issue list / pr list / repo ...\n\n"
    "CLUSTER OPS (read-only status):\n"
    "  cluster-ops-axi dashboard\n"
    "  cluster-ops-axi service <host> <unit>\n"
    "  cluster-ops-axi exec <host> <command>\n\n"
    "OTHER:\n"
    "  weather-axi current \"<location>\" | forecast \"<location>\" [--days N]\n"
    "  ocr-axi image <path> | pdf-text <path>\n\n"
    "ROUTING: web/news/current events → tavily-axi or exa-axi; "
    "stored knowledge → lore-axi kb-search; ops → cluster-ops-axi."
)


def _make_decompose_hint_hook():
    """Build the pre_llm_call hook that injects parallel-decomposition guidance.

    Why: The skills toolset is disabled in default sessions so the pm_orchestrator
    SKILL.md never surfaces in the system prompt — the model therefore never learns
    to auto-decompose multi-part requests into parallel sub-agents. This hook
    injects a concise decomposition hint as user-message context on every turn
    (the only channel available to plugins without breaking the prompt-cache
    prefix). The hint tells the model: batch independent subtasks into one
    hermes_mpm_orchestrate call instead of serializing them.
    What: Returns a ``pre_llm_call`` callback that always returns
    ``{"context": _DECOMPOSE_HINT}`` so the engine appends it to the current
    turn's user message. The hook is a no-op on subagent turns (platform ==
    "subagent") to avoid leaking PM instructions to child agents.
    Test: call hook(platform="cli", user_message="x") → dict with "context" key
    containing "hermes_mpm_orchestrate"; call hook(platform="subagent", ...) →
    None (no injection on child turns).
    """

    def hint_hook(**kw) -> dict | None:
        platform = (kw.get("platform") or "").lower()
        # Suppress on subagent turns — children should not receive PM instructions.
        if platform in ("subagent", "leaf"):
            return None
        return {"context": _DECOMPOSE_HINT}

    hint_hook.__name__ = "hermes_mpm_decompose_hint"
    return hint_hook


def _make_axi_cheatsheet_hook():
    """Build the pre_llm_call hook that injects the AXI CLI cheatsheet.

    Why: Hermes agents fall back to KB or training data for web search because
    they don't know the AXI CLI tools exist. Skills are disabled in default
    sessions so no skill file surfaces this knowledge. This hook injects a
    compact, verified cheatsheet of all available AXI CLIs into the system
    prompt context on every non-subagent turn.
    What: Returns a pre_llm_call callback that returns {"context": _AXI_CHEATSHEET}
    for parent/PM turns. Suppressed on subagent/leaf turns (children receive a
    scoped subset via their task context instead).
    Test: call hook(platform="cli") -> dict with "context" key containing
    "tavily-axi"; call hook(platform="subagent") -> None.
    """

    def axi_hook(**kw) -> dict | None:
        platform = (kw.get("platform") or "").lower()
        # Suppress on subagent turns — children get a scoped AXI hint via
        # their task context (passed by the orchestrator), not the full sheet.
        if platform in ("subagent", "leaf"):
            return None
        return {"context": _AXI_CHEATSHEET}

    axi_hook.__name__ = "hermes_mpm_axi_cheatsheet"
    return axi_hook


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

    # 2c) Decomposition hint — inject parallel fan-out guidance as user-message
    #     context on every turn. Registered as a SEPARATE hook (not merged into
    #     the composed handler above) so the context injection path is independent
    #     of the routing model-bundle path. The hint is suppressed on subagent
    #     turns to avoid leaking PM instructions to child agents.
    try:
        ctx.register_hook("pre_llm_call", _make_decompose_hint_hook())
    except Exception as exc:
        logger.debug("hermes-mpm: decompose hint hook skipped: %s", exc)

    # 2d) AXI cheatsheet — inject always-on inventory of AXI shell CLIs into
    #     every non-subagent turn. Ensures the PM always knows to use
    #     tavily-axi/exa-axi for web search, lore-axi for KB ops, etc. even
    #     when skills are disabled. Suppressed on subagent/leaf turns — children
    #     receive a scoped subset via their orchestrate task context instead.
    try:
        ctx.register_hook("pre_llm_call", _make_axi_cheatsheet_hook())
    except Exception as exc:
        logger.debug("hermes-mpm: AXI cheatsheet hook skipped: %s", exc)

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


    # 3b) Pipeline tools — 6-stage quality gate for bug fixes.
    try:
        _PIPELINE_TOOLS = [
            (mpm_pipeline.INIT_SCHEMA, mpm_pipeline.handle_init,
             "Initialize a new pipeline run with state tracking.", "🚀"),
            (mpm_pipeline.TRANSITION_SCHEMA, mpm_pipeline.handle_transition,
             "Transition pipeline to the next phase.", "⏩"),
            (mpm_pipeline.RECORD_EVIDENCE_SCHEMA, mpm_pipeline.handle_record_evidence,
             "Record gate evidence for the current phase.", "📋"),
            (mpm_pipeline.VERIFY_GATE_SCHEMA, mpm_pipeline.handle_verify_gate,
             "Verify that a phase gate has passed.", "✅"),
            (mpm_pipeline.STATUS_SCHEMA, mpm_pipeline.handle_status,
             "Show current pipeline state.", "📊"),
            (mpm_pipeline.RECOVER_SCHEMA, mpm_pipeline.handle_recover,
             "Handle pipeline failure (retry/skip/escalate).", "🔄"),
        ]
        for _schema, _handler, _desc, _emoji in _PIPELINE_TOOLS:
            try:
                ctx.register_tool(
                    name=_schema["name"],
                    toolset=mpm_pipeline.TOOLSET_NAME,
                    schema=_schema,
                    handler=_handler,
                    description=_desc,
                    emoji=_emoji,
                )
            except Exception as exc:
                logger.debug("hermes-mpm: tool %s registration skipped: %s", _schema["name"], exc)
    except Exception as exc:
        logger.warning("hermes-mpm: pipeline tool registration failed: %s", exc)

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

    logger.info(
        "hermes-mpm registered: mpm CLI + pre_llm_call (routing+intent) "
        "+ decompose_hint + axi_cheatsheet + orchestrate tool + skill"
    )

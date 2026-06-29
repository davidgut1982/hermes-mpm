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

import json
import logging
import os
import re
import time
from pathlib import Path

from . import cli, intent, orchestrator, routing, runs_db
from . import pipeline as mpm_pipeline

logger = logging.getLogger("hermes_mpm")

CONFIG_NAMESPACE = "hermes_mpm"

# Default run-retention when ``plugins.entries.hermes_mpm.runs.retention_days``
# is unset. 30 days keeps recent history queryable without unbounded growth.
DEFAULT_RETENTION_DAYS = 30

# Maps the engine's child_status vocabulary (from the subagent_stop hook and the
# async-complete marker) onto our terminal run statuses. Anything finished but
# unrecognized is treated as ``failed`` — a completed-but-unclassified run is not
# a success, so we fail closed.
_CHILD_STATUS_MAP = {
    "completed": runs_db.STATUS_DONE,
    "success": runs_db.STATUS_DONE,
    "done": runs_db.STATUS_DONE,
    "error": runs_db.STATUS_FAILED,
    "failed": runs_db.STATUS_FAILED,
    "spawn_failed": runs_db.STATUS_FAILED,
    "interrupted": runs_db.STATUS_FAILED,
    "timed_out": runs_db.STATUS_TIMED_OUT,
    "timeout": runs_db.STATUS_TIMED_OUT,
    "crashed": runs_db.STATUS_CRASHED,
}

# Parser for the async-delegation completion marker re-injected as a user
# message (see tools/process_registry._format_async_delegation). The header
# carries the delegation_id; a later line carries the Status. We also read the
# "Original goal:" line to correlate back to the start row (subagent_start does
# NOT carry the delegation_id).
_ASYNC_MARKER_PREFIX = "[ASYNC DELEGATION COMPLETE — deleg_"
_RE_DELEG_ID = re.compile(r"\[ASYNC DELEGATION COMPLETE — (deleg_[0-9a-f]+)\]")
_RE_ASYNC_STATUS = re.compile(r"^Status:\s*(\S+)", re.MULTILINE)
_RE_ASYNC_GOAL = re.compile(r"^Original goal:\s*(.+)$", re.MULTILINE)
# Duration appears inline on the Status line, e.g. "… Duration: 12s". Captured so
# async runs record a duration like sync runs do (which get duration_ms from the
# subagent_stop hook). Bare seconds only — the marker never emits sub-second.
_RE_ASYNC_DURATION = re.compile(r"Duration:\s*(\d+)\s*s", re.IGNORECASE)


def _parse_async_duration_ms(msg: str) -> int | None:
    """Parse the ``Duration: Ns`` field from an async-complete marker → ms.

    Why: Async runs are closed by the pre_llm_call fallback (not subagent_stop),
    so without parsing the marker they'd show no duration while sync runs do. This
    surfaces the wall-clock the engine already printed in the marker.
    What: Returns the integer seconds from ``Duration: <N>s`` × 1000, or None when
    the field is absent/malformed (record_end then leaves duration_ms NULL).
    Test: ``test_async_complete_handler_parses_duration`` (5s → 5000).
    """
    m = _RE_ASYNC_DURATION.search(msg or "")
    if not m:
        return None
    try:
        return int(m.group(1)) * 1000
    except (TypeError, ValueError):
        return None


def _map_child_status(child_status) -> str:
    """Map an engine child_status onto our terminal run status (fail-closed).

    Why: subagent_stop and the async marker speak the engine's vocabulary
    (completed/error/timed_out/…); the DB stores our normalized statuses.
    What: Case-insensitive lookup in _CHILD_STATUS_MAP; unknown/None → 'failed'.
    Test: ``test_status_mapping_table``.
    """
    if not child_status:
        return runs_db.STATUS_FAILED
    return _CHILD_STATUS_MAP.get(str(child_status).strip().lower(), runs_db.STATUS_FAILED)


def _make_subagent_start_handler():
    """Build the subagent_start hook → record_start (running row).

    Why: Every spawned child (sync OR async) fires subagent_start; this is where a
    run becomes durable. Must never raise into the engine — a tracking failure
    cannot be allowed to break a delegation.
    What: Returns handler(**kw) that records a running row keyed by
    child_session_id (run_id), run_type='subagent'. Wraps everything in
    try/except that logs+swallows.
    Test: ``test_subagent_start_handler_creates_running`` +
    ``test_subagent_start_handler_swallows_db_error``.
    """

    def handler(**kw) -> None:
        try:
            run_id = kw.get("child_session_id")
            if not run_id:
                return  # nothing to key on — skip cleanly
            runs_db.record_start(
                run_id=str(run_id),
                parent_session_id=kw.get("parent_session_id"),
                role=kw.get("child_role"),
                profile=kw.get("child_role"),  # best-effort: role doubles as profile
                goal=kw.get("child_goal"),
                started_at=int(time.time()),
                run_type="subagent",
            )
        except Exception as exc:  # never break a delegation
            logger.debug("mpm-runs: subagent_start tracking failed (ignored): %s", exc)

    handler.__name__ = "hermes_mpm_runs_subagent_start"
    return handler


def _make_subagent_stop_handler():
    """Build the subagent_stop hook → record_end (close the run).

    Why: Closes sync (and any stop-firing) runs with their mapped terminal status
    so they stop counting as in-flight. Async children do NOT fire this — they are
    closed by the async-complete fallback below.
    What: Returns handler(**kw) that maps child_status and UPDATEs the run by
    child_session_id. Logs+swallows on error.
    Test: ``test_subagent_stop_handler_closes_with_mapped_status`` +
    ``test_subagent_stop_handler_swallows_db_error``.
    """

    def handler(**kw) -> None:
        try:
            run_id = kw.get("child_session_id")
            if not run_id:
                return
            runs_db.record_end(
                run_id=str(run_id),
                status=_map_child_status(kw.get("child_status")),
                ended_at=int(time.time()),
                duration_ms=kw.get("duration_ms"),
                summary=kw.get("child_summary"),
            )
        except Exception as exc:
            logger.debug("mpm-runs: subagent_stop tracking failed (ignored): %s", exc)

    handler.__name__ = "hermes_mpm_runs_subagent_stop"
    return handler


def _make_async_complete_handler():
    """Build the pre_llm_call fallback that closes ASYNC runs.

    Why: VERIFIED against tools/delegate_tool.py + tools/async_delegation.py —
    background (fire-and-forget) children do NOT fire subagent_stop. Their result
    re-enters the turn as a user message ``[ASYNC DELEGATION COMPLETE — deleg_…]``.
    Without this handler those runs would be falsely marked ``crashed`` by the
    next restart sweep. This closes them when the marker arrives.
    What: Returns handler(**kw); if user_message starts with the async marker,
    parse delegation_id + Status + Original goal, find the matching running run
    (by delegation_id, else by goal), and record_end it. Always returns None
    (never alters the turn). Logs+swallows on error.
    Test: ``test_async_complete_handler_closes_run_by_goal`` +
    ``test_async_complete_handler_ignores_non_marker``.
    """

    def handler(**kw):
        try:
            msg = (kw.get("user_message") or "").lstrip()
            if not msg.startswith(_ASYNC_MARKER_PREFIX):
                return None
            m_id = _RE_DELEG_ID.search(msg)
            delegation_id = m_id.group(1) if m_id else None
            m_status = _RE_ASYNC_STATUS.search(msg)
            status = _map_child_status(m_status.group(1) if m_status else None)
            m_goal = _RE_ASYNC_GOAL.search(msg)
            goal = m_goal.group(1).strip() if m_goal else None

            # Correlate to the start row: delegation_id is most precise, but
            # subagent_start does not carry it, so fall back to goal match.
            run_id = None
            if delegation_id:
                run_id = runs_db.find_running_by_delegation(delegation_id)
            if not run_id and goal:
                run_id = runs_db.find_running_by_goal(goal)
            if not run_id:
                return None
            runs_db.record_end(
                run_id=run_id,
                status=status,
                ended_at=int(time.time()),
                duration_ms=_parse_async_duration_ms(msg),
            )
            # Stamp the delegation_id onto the row for cross-ref if it wasn't set
            # at start (the common case — start has no delegation_id).
            if delegation_id:
                runs_db._write(
                    "UPDATE subagent_runs SET delegation_id = ? WHERE run_id = ?",
                    (delegation_id, run_id),
                )
        except Exception as exc:
            logger.debug("mpm-runs: async-complete tracking failed (ignored): %s", exc)
        return None

    handler.__name__ = "hermes_mpm_runs_async_complete"
    return handler


def _coerce_result_dict(result) -> dict | None:
    """Best-effort coerce a tool result into a dict, else None.

    Why: post_tool_call delivers ``delegate_task``'s result as either a JSON
    string (the common case — delegate_tool returns ``json.dumps(...)``) or a
    pre-parsed dict, depending on the call path. The stamping logic needs a dict
    and must never raise on a malformed/odd-typed result.
    What: Returns the dict unchanged; json.loads a str and returns it only if it
    parses to a dict; returns None for anything else or on parse failure.
    Test: ``test_post_tool_call_swallows_malformed_result_and_db_error`` (str/int/
    missing) + the background-dispatch tests (str and dict forms).
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _is_background_dispatch(data: dict) -> bool:
    """True iff a delegate_task result represents a background (async) dispatch.

    Why: Only direct ``delegate_task(background=True)`` produces an async run that
    needs delegation_id correlation; a SYNC result closes via subagent_stop and
    must be left alone. The async result (delegate_tool.py) carries
    ``delegation_id`` plus ``mode == "background"`` / ``status == "dispatched"``.
    What: Returns True when a non-empty ``delegation_id`` is present AND
    (mode == 'background' OR status == 'dispatched'). Sync results lack the id.
    Test: background tests stamp; ``test_post_tool_call_ignores_sync_delegate_result``.
    """
    if not data.get("delegation_id"):
        return False
    return data.get("mode") == "background" or data.get("status") == "dispatched"


def _make_post_tool_call_handler():
    """Build the post_tool_call hook → stamp delegation_id onto the async run row.

    Why: For a direct ``delegate_task(background=True)``, subagent_start creates
    the run row (delegation_id NULL) and then delegate_task returns and fires this
    SYNCHRONOUS post_tool_call carrying the delegation_id. Stamping it onto the
    row lets the later async-complete marker close the run by EXACT delegation_id
    rather than fragile goal-text matching (which breaks on truncation / shared
    goals). Must never raise into the engine — post_tool_call is observational.
    What: Returns handler(**kw); for tool_name == 'delegate_task' it coerces the
    result to a dict and, if it is a background dispatch, calls
    runs_db.stamp_delegation_id(goal, delegation_id). Sync results / other tools /
    malformed results are no-ops. Logs+swallows on error.
    Test: ``test_post_tool_call_*`` in test_runs_hooks.py.
    """

    def handler(**kw) -> None:
        try:
            if kw.get("tool_name") != "delegate_task":
                return
            data = _coerce_result_dict(kw.get("result"))
            if not data or not _is_background_dispatch(data):
                return  # sync dispatch / not a dispatch → subagent_stop closes it
            delegation_id = data.get("delegation_id")
            goal = data.get("goal")
            if delegation_id and goal:
                runs_db.stamp_delegation_id(str(goal), str(delegation_id))
        except Exception as exc:  # never break the turn — observational hook
            logger.debug("mpm-runs: post_tool_call stamping failed (ignored): %s", exc)

    handler.__name__ = "hermes_mpm_runs_post_tool_call"
    return handler


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


def _runs_retention_days(cfg: dict) -> int:
    """Resolve run-retention days from the merged plugin config.

    Why: Operators need to bound run history without code changes; this reads
    ``plugins.entries.hermes_mpm.runs.retention_days`` (surfaced into the merged
    cfg under the ``runs`` key) with a 30-day default.
    What: Returns cfg['runs']['retention_days'] as an int, or DEFAULT_RETENTION_DAYS
    when absent/invalid. A non-positive value disables purging (handled by purge_old).
    Test: ``test_runs_retention_days_reads_config`` + default fallback test.
    """
    try:
        runs_cfg = cfg.get("runs") if isinstance(cfg, dict) else None
        if isinstance(runs_cfg, dict) and "retention_days" in runs_cfg:
            return int(runs_cfg["retention_days"])
    except (TypeError, ValueError):
        pass
    return DEFAULT_RETENTION_DAYS


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
    "[MPM] When a turn needs several INDEPENDENT lookups or reads, issue them as "
    "PARALLEL tool calls in ONE turn — multiple gh-axi/file-read/tavily-axi/lore-axi "
    "calls together, not one per turn and not chained with &&. Two patterns: "
    "(A) SAME-kind independent lookups (e.g. 'summarize issues 101, 202, 303', "
    "'read these 4 files', 'search news on X, Y, Z') → emit the calls in parallel "
    "yourself; do NOT orchestrate. "
    "(B) Independent subtasks that each need a DIFFERENT specialist profile → call "
    "hermes_mpm_orchestrate ONCE with all subtasks batched (parallel fan-out). "
    "MUST orchestrate (multi-specialist): "
    "  '(1) latest AI news (2) KB on MCP (3) arxiv papers' → search+kb+search profiles; "
    "  'research X and look up our KB on Y' → search+kb. "
    "Profile routing: web/news/current/arxiv → profile='search'; "
    "kb/notes/knowledge → profile='kb'; ops/service/cluster → profile='ops'; "
    "code/engineering → profile='engineer'. "
    "Search subtasks resolve their query by running tavily-axi search directly "
    "(no KB, no MCP); put that in the subtask's context so the spawned search agent "
    "fetches the web result, not training data. "
    "Single topic, or a message that refers to prior context ('do it', 'go ahead', "
    "'lore it') → handle inline; don't invent subtasks or fan out. "
    "Max 5 concurrent subtasks; batch if more."
)

# Always-on AXI cheatsheet — injected into every non-subagent turn as system context.
# These are lightweight shell CLIs (not MCP servers). Run them via the terminal tool.
# Target: ~200-300 tokens. Verified invocations only (from --help output).
_AXI_CHEATSHEET: str = (
    "[AXI TOOLS — run via terminal/shell, NOT MCP]\n"
    "These replace the heavy MCP servers. Use them for all web search and KB ops.\n\n"
    "WEB SEARCH (for current info, news, URLs — always use these, not training data):\n"
    '  tavily-axi search "<query>" [--limit N] [--depth basic|advanced] [--include-answer]\n'
    '  exa-axi search "<query>" [--limit N] [--type auto|text|url]\n'
    "  exa-axi similar <url>  # find pages similar to a URL\n\n"
    "KNOWLEDGE BASE (Lore — stored project/config knowledge):\n"
    '  lore-axi kb-search "<query>" [--hybrid] [--limit 20]\n'
    "  lore-axi kb-add <topic> <title> <content>\n"
    "  lore-axi kb-get <id>\n\n"
    "GITHUB:\n"
    "  gh-axi issue list / pr list / repo ...\n\n"
    "CLUSTER OPS (read-only status):\n"
    "  cluster-ops-axi dashboard\n"
    "  cluster-ops-axi service <host> <unit>\n"
    "  cluster-ops-axi exec <host> <command>\n\n"
    "OTHER:\n"
    '  weather-axi current "<location>" | forecast "<location>" [--days N]\n'
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
        # Clear any captured agent so a child turn cannot read a stale parent agent.
        if platform in ("subagent", "leaf"):
            try:
                orchestrator.clear_agent()
            except Exception:
                pass
            return None
        # Suppress on short / referential messages. These are almost always a
        # single action tied to prior conversation context ("lore it", "do it",
        # "yes", "go ahead", "continue"), NOT a multi-part request. Injecting
        # fan-out pressure here causes the model to invent subtasks the user
        # never asked for — a direct hallucination trigger.
        #
        # Finding 3(a): clear the captured agent on these short turns too — the
        # capture below is skipped here, so without an explicit clear a short
        # referential turn would inherit a STALE agent captured by a prior turn.
        msg = kw.get("user_message") or ""
        if len(msg.strip().split()) <= 6:
            try:
                orchestrator.clear_agent()
            except Exception:
                pass
            return None
        # Capture the live agent (per-context contextvar) so orchestrator.handle()
        # can inject it as parent_agent when dispatching delegate_task. The plugin
        # tool dispatch path does not pass parent_agent; this pre_llm_call hook
        # fires before every LLM response (and therefore before any tool call in
        # the turn), so the captured agent is fresh when hermes_mpm_orchestrate
        # runs. If no agent is provided, clear so a prior turn's agent can't leak.
        _agent = kw.get("agent")
        try:
            if _agent is not None:
                orchestrator.capture_agent(_agent)
            else:
                orchestrator.clear_agent()
        except Exception:
            pass  # never break the turn
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

    # 0) Run-tracking DB — startup lifecycle: init schema, then (GATEWAY ONLY)
    #    sweep restart-orphaned runs + purge old ended rows. DDL runs ONLY here
    #    (init_db), never on a hot hook path.
    #
    #    CRITICAL: register() runs in EVERY process that loads the plugin — the
    #    gateway, but ALSO `hermes mpm runs` and the dashboard. The sweep/purge
    #    are MUTATING; running them outside the gateway corrupted data (the CLI or
    #    dashboard would mark the live gateway's in-flight runs ``crashed`` and
    #    make async runs permanently un-closable). So we gate them behind
    #    _HERMES_GATEWAY=1 — the env the gateway process sets (gateway/run.py) and
    #    explicitly strips from its watcher/children. init_db (idempotent, additive
    #    ALTER) is safe everywhere and still runs so the CLI sees current schema.
    #    All best-effort: a tracking-DB failure must degrade hooks to no-ops, never
    #    block plugin load.
    try:
        runs_db.init_db()
        if os.environ.get("_HERMES_GATEWAY") == "1":
            now = int(time.time())
            orphaned = runs_db.sweep_orphaned(now, os.getpid())
            retention_days = _runs_retention_days(cfg)
            purged = runs_db.purge_old(retention_days, now)
            logger.info(
                "mpm-runs: db ready, %d orphaned run(s) marked crashed, %d old run(s) purged",
                orphaned,
                purged,
            )
        else:
            logger.debug(
                "mpm-runs: db ready (non-gateway process — sweep/purge skipped)"
            )
    except Exception as exc:
        logger.warning("hermes-mpm: run-tracking DB init skipped: %s", exc)

    # 0b) Subagent lifecycle hooks → run DB. subagent_start opens a run;
    #     subagent_stop closes sync runs. Background/async children do NOT fire
    #     subagent_stop (verified in delegate_tool.py/async_delegation.py), so a
    #     pre_llm_call fallback closes them when the async-complete marker
    #     re-enters the turn. Each handler swallows its own errors.
    try:
        ctx.register_hook("subagent_start", _make_subagent_start_handler())
        ctx.register_hook("subagent_stop", _make_subagent_stop_handler())
        # post_tool_call stamps the async delegation_id onto the run row created
        # by subagent_start (delegation_id NULL at start), so the async-complete
        # marker can close it by EXACT delegation_id instead of goal text.
        ctx.register_hook("post_tool_call", _make_post_tool_call_handler())
        ctx.register_hook("pre_llm_call", _make_async_complete_handler())
    except Exception as exc:
        logger.debug("hermes-mpm: run-tracking hooks skipped: %s", exc)

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
            (
                mpm_pipeline.INIT_SCHEMA,
                mpm_pipeline.handle_init,
                "Initialize a new pipeline run with state tracking.",
                "🚀",
            ),
            (
                mpm_pipeline.TRANSITION_SCHEMA,
                mpm_pipeline.handle_transition,
                "Transition pipeline to the next phase.",
                "⏩",
            ),
            (
                mpm_pipeline.RECORD_EVIDENCE_SCHEMA,
                mpm_pipeline.handle_record_evidence,
                "Record gate evidence for the current phase.",
                "📋",
            ),
            (
                mpm_pipeline.VERIFY_GATE_SCHEMA,
                mpm_pipeline.handle_verify_gate,
                "Verify that a phase gate has passed.",
                "✅",
            ),
            (
                mpm_pipeline.STATUS_SCHEMA,
                mpm_pipeline.handle_status,
                "Show current pipeline state.",
                "📊",
            ),
            (
                mpm_pipeline.RECOVER_SCHEMA,
                mpm_pipeline.handle_recover,
                "Handle pipeline failure (retry/skip/escalate).",
                "🔄",
            ),
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

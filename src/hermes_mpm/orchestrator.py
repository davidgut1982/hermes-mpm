"""Parallel fan-out orchestrate tool for hermes-mpm.

Why: A PM agent that delegates N subtasks one ``delegate_task`` call at a time
serializes work that could run concurrently. The native ``delegate_task`` batch
mode already fans tasks out across a ThreadPoolExecutor — so the win is to make
ONE batched call instead of N sequential ones. ``hermes_mpm_orchestrate`` is the
single tool that does this: validate caller-supplied subtasks, then issue one
batched ``delegate_task`` so all profiles run in parallel.

What: ``handle(args)`` validates ``goal`` + ``subtasks`` (each {profile, goal,
context?}); rejects empty subtasks and unknown profiles with a clean error; then
calls ``ctx.dispatch_tool("delegate_task", {"tasks": [...], "role": "leaf"})``
and returns the aggregated JSON result. ``ctx`` is captured in ``register()``
(set_ctx) — the tool has no other way to reach the registry.

Test: handle({"goal":"g","subtasks":[]}) -> error; unknown profile -> error;
valid subtasks -> builds the delegate_task tasks payload (mock ctx.dispatch_tool
and assert the payload shape). See test_orchestrator.py.
"""

from __future__ import annotations

import contextvars
import json
import logging
from typing import Any, Dict, List, Optional

from . import profiles
from .gate import adapter as gate_adapter

logger = logging.getLogger("hermes_mpm.orchestrator")

TOOLSET_NAME = "hermes-mpm"
TOOL_NAME = "hermes_mpm_orchestrate"

# Captured at register() time so the tool handler can reach dispatch_tool.
_CTX = None

# Agent captured from the pre_llm_call hook each turn, stored in a CONTEXTVAR
# (Finding 3). A contextvar is per-execution-context, so concurrent gateway
# sessions/agents each see their own value with no shared-mutable-global race;
# clearing it (clear_agent) prevents a short/referential turn from reading a
# STALE agent captured by a prior turn. The engine's plugin tool dispatch path
# does not inject parent_agent, which is why we capture it from the hook at all.
_AGENT_CTX: contextvars.ContextVar = contextvars.ContextVar(
    "hermes_mpm_captured_agent", default=None
)


def capture_agent(agent) -> None:
    """Store the live AIAgent reference (per-context) for use in handle().

    Why: The plugin tool dispatch path does not inject parent_agent — only the
    built-in delegate_task path does. Capturing the agent from the pre_llm_call
    hook lets handle() inject it. A contextvar (not a bare global) keeps this
    safe under concurrent sessions and avoids stale cross-turn leakage.
    What: Sets the ``_AGENT_CTX`` contextvar to the provided agent.
    Test: capture_agent(x); assert current_agent() is x.
    """
    _AGENT_CTX.set(agent)


def current_agent():
    """Return the agent captured for the current context, or None.

    Why: handle() needs the per-call agent to inject as parent_agent; reading the
    contextvar (vs a shared global) is concurrency-safe.
    What: Returns the current ``_AGENT_CTX`` value.
    Test: after capture_agent(x), current_agent() is x; after clear_agent(), None.
    """
    return _AGENT_CTX.get()


def clear_agent() -> None:
    """Clear the captured agent so it cannot leak to a later turn.

    Why: Finding 3(a) — capture happens AFTER the short-message guard, so a short
    referential turn must not read a stale agent from a prior turn. Clearing
    resets the contextvar to its None default.
    What: Sets ``_AGENT_CTX`` back to None.
    Test: capture_agent(x); clear_agent(); current_agent() is None.
    """
    _AGENT_CTX.set(None)


DELEGATE_TOOL = "delegate_task"
LEAF_ROLE = "leaf"

ORCHESTRATE_SCHEMA: Dict[str, Any] = {
    # Top-level structure mirrors DELEGATE_TASK_SCHEMA: name + description +
    # parameters wrapping the JSON schema. registry.get_definitions() does
    # {**entry.schema, "name": name} → must have "parameters" at this level
    # or the model sees an empty schema and refuses to call the tool.
    "name": TOOL_NAME,
    "description": (
        "Fan out 2+ INDEPENDENT subtasks to agent profiles IN PARALLEL via one "
        "batched delegate_task call. Use this instead of looping delegate_task — "
        "batching lets the native ThreadPoolExecutor run all subtasks concurrently "
        "so total elapsed ≈ max(child_time) not sum(child_times). "
        "Call ONCE with all subtasks in the array; do not call per subtask."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The overall goal this batch of subtasks serves.",
            },
            "subtasks": {
                "type": "array",
                "description": (
                    "Subtasks to run IN PARALLEL. Each runs as its own child agent "
                    "under the given profile. Supply 2+ to get fan-out."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "string",
                            "description": "Agent archetype to run this subtask (e.g. 'ops').",
                        },
                        "goal": {
                            "type": "string",
                            "description": "What this subtask must accomplish.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional extra context for the subtask.",
                        },
                    },
                    "required": ["profile", "goal"],
                    "additionalProperties": False,
                },
                "minItems": 1,
            },
        },
        "required": ["goal", "subtasks"],
        "additionalProperties": False,
    },
}


def set_ctx(ctx) -> None:
    """Capture the plugin context so the tool handler can dispatch_tool.

    Why: The orchestrate tool must call the native ``delegate_task`` through the
    registry; ``ctx.dispatch_tool`` is the supported bridge. The tool handler
    receives only its args, so ctx is stored module-side at register() time.
    What: Sets the module global ``_CTX``.
    Test: set_ctx(obj); assert orchestrator._CTX is obj.
    """
    global _CTX
    _CTX = ctx


def _validate(args: Dict[str, Any]) -> Optional[str]:
    """Validate orchestrate args; return an error string or None if valid.

    Why: Fail fast with a clear message before spending any delegation.
    What: Requires a non-empty ``goal`` and a non-empty ``subtasks`` list where
    every item has a ``profile`` (known archetype) and a ``goal``.
    Test: empty subtasks -> message; unknown profile -> message naming it; valid
    -> None.
    """
    goal = (args.get("goal") or "").strip()
    if not goal:
        return "Missing required parameter: goal"

    subtasks = args.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        return "subtasks must be a non-empty list"

    known = set(profiles.list_archetypes())
    for i, st in enumerate(subtasks):
        if not isinstance(st, dict):
            return f"subtasks[{i}] must be an object"
        prof = (st.get("profile") or "").strip()
        sub_goal = (st.get("goal") or "").strip()
        if not prof:
            return f"subtasks[{i}] missing 'profile'"
        if not sub_goal:
            return f"subtasks[{i}] missing 'goal'"
        if prof not in known:
            return f"subtasks[{i}] unknown profile '{prof}'"
    return None


def _build_tasks(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shape validated subtasks into the delegate_task ``tasks`` payload.

    Why: delegate_task batch mode keys on a list of {profile, goal, context,
    role}; building it in one place keeps the handler readable and testable.
    What: Maps each subtask to a task dict, carrying context only when present,
    and stamping the leaf role so children can't recursively fan out.
    Test: two subtasks -> two task dicts with matching profile/goal and
    role="leaf".
    """
    tasks: List[Dict[str, Any]] = []
    for st in subtasks:
        task: Dict[str, Any] = {
            "profile": st["profile"].strip(),
            "goal": st["goal"].strip(),
            "role": LEAF_ROLE,
        }
        context = (st.get("context") or "").strip()
        if context:
            task["context"] = context
        tasks.append(task)
    return tasks


def _gate_tasks(
    tasks: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply the review-gate verdict to each fan-out subtask before dispatch.

    Why: Finding 2 — handle() dispatches delegate_task via the internal registry,
    which does NOT pass through the engine's pre_tool_call hook. Without this,
    fanned-out subtasks reach delegate_task UNGATED. We reuse gate_adapter.evaluate
    — the SAME verdict logic the hook uses — so a subtask the gate would block is
    blocked here too.
    What: For each task, calls evaluate("delegate_task", {single-task batch}); a
    BLOCK verdict drops the task into ``blocked`` (with the reason), otherwise it
    is kept. Returns (allowed, blocked). The gate degrades to allow when no gate
    is armed.
    Test: a task with a 'delete prod' goal under a blocking gate lands in blocked;
    a 'list status' task lands in allowed.
    """
    allowed: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for task in tasks:
        try:
            verdict = gate_adapter.evaluate(DELEGATE_TOOL, {"tasks": [task]})
        except Exception as exc:  # never let a gate error crash fan-out — fail closed
            logger.warning("hermes-mpm orchestrate: gate evaluate error (blocking): %s", exc)
            blocked.append({"goal": task.get("goal", ""), "reason": f"gate error: {exc}"})
            continue
        if verdict.decision == "block":
            blocked.append({"goal": task.get("goal", ""), "reason": verdict.reason})
        else:
            allowed.append(task)
    return allowed, blocked


def handle(args: Dict[str, Any], **_kwargs) -> str:
    """Validate subtasks, gate each, then fan the survivors out via one delegate_task.

    Why: One batched call lets the native ThreadPoolExecutor run all subtasks in
    parallel instead of serializing N delegations. Finding 2: each subtask is
    gated BEFORE the internal dispatch (which bypasses the pre_tool_call hook), so
    blockable subtasks never reach delegate_task ungated.
    What: Validates args; gates each subtask via gate_adapter.evaluate; if every
    subtask is blocked returns a JSON {"blocked": [...]} with no dispatch; otherwise
    issues a single ``delegate_task`` for the survivors and attaches any blocked
    subtasks to the result. Uses the per-context captured agent (Finding 3) as
    parent_agent, falling back to ctx.dispatch_tool.
    Test: all blocked -> no dispatch + blocked list; mixed -> survivors dispatched
    + blocked reported; all allowed -> unchanged batched dispatch.
    """
    err = _validate(args)
    if err:
        return json.dumps({"error": err})

    if _CTX is None or not hasattr(_CTX, "dispatch_tool"):
        return json.dumps({"error": "orchestrate unavailable: plugin context not initialized"})

    tasks = _build_tasks(args["subtasks"])

    # Finding 2: gate each subtask before any internal dispatch.
    allowed, blocked = _gate_tasks(tasks)
    if not allowed:
        # Every subtask blocked — do not dispatch anything.
        logger.warning(
            "hermes-mpm orchestrate: all %d subtask(s) blocked by review gate", len(blocked)
        )
        return json.dumps(
            {
                "error": "all subtasks blocked by review gate",
                "blocked": blocked,
            }
        )

    payload = {"tasks": allowed, "role": LEAF_ROLE}

    logger.info(
        "hermes-mpm orchestrate: goal=%r fanning out %d subtask(s) in parallel "
        "(%d blocked by gate)",
        (args.get("goal") or "").strip()[:80],
        len(allowed),
        len(blocked),
    )

    # Finding 4: import the registry OUTSIDE the dispatch try so an import failure
    # does not get caught by the dispatch except and masked as a "dispatch failed"
    # error — it must fall back to the ctx.dispatch_tool path instead.
    _reg = None
    _parent_agent = current_agent()  # Finding 3: per-context agent, never stale
    if _parent_agent is not None:
        try:
            from tools.registry import registry as _reg  # noqa: PLC0415
        except Exception as exc:
            logger.debug(
                "hermes-mpm orchestrate: registry import unavailable, using ctx fallback: %s", exc
            )
            _reg = None

    try:
        if _parent_agent is not None and _reg is not None:
            result = _reg.dispatch(DELEGATE_TOOL, payload, parent_agent=_parent_agent)
        else:
            result = _CTX.dispatch_tool(DELEGATE_TOOL, payload)
    except Exception as exc:
        logger.warning("hermes-mpm orchestrate: delegate_task dispatch failed: %s", exc)
        return json.dumps({"error": f"delegate_task dispatch failed: {exc}", "blocked": blocked})

    # No blocked subtasks: return the delegate result verbatim (back-compat).
    if not blocked:
        return result if isinstance(result, str) else json.dumps(result)

    # Some subtasks were blocked — surface them alongside the delegate result.
    parsed: Any
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (ValueError, TypeError):
            parsed = {"result": result}
    else:
        parsed = result
    if isinstance(parsed, dict):
        parsed["blocked"] = blocked
        return json.dumps(parsed)
    return json.dumps({"result": parsed, "blocked": blocked})

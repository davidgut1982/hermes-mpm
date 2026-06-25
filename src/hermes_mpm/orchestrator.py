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

import json
import logging
from typing import Any, Dict, List, Optional

from . import profiles

logger = logging.getLogger("hermes_mpm.orchestrator")

TOOLSET_NAME = "hermes-mpm"
TOOL_NAME = "hermes_mpm_orchestrate"

# Captured at register() time so the tool handler can reach dispatch_tool.
_CTX = None

# Agent captured from pre_llm_call hook each turn.
# pre_llm_call fires before every LLM response, so this is always set
# before any tool call (including hermes_mpm_orchestrate) in that turn.
# Plugin-only approach: avoids engine changes to pass parent_agent through
# the plugin tool dispatch path (which doesn't inject parent_agent).
_captured_agent = None


def capture_agent(agent) -> None:
    """Store the live AIAgent reference for use in handle().

    Why: The plugin tool dispatch path does not inject parent_agent — only
    the built-in delegate_task path does. By capturing the agent from the
    pre_llm_call hook (which always fires before tool calls in the same
    turn), handle() can inject it when calling delegate_task.
    What: Sets module global _captured_agent to the provided agent.
    Test: capture_agent(mock_agent); assert orchestrator._captured_agent is mock_agent.
    """
    global _captured_agent
    _captured_agent = agent

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


def handle(args: Dict[str, Any], **_kwargs) -> str:
    """Validate subtasks and fan them out via one batched delegate_task call.

    Why: One batched call lets the native ThreadPoolExecutor run all subtasks in
    parallel instead of serializing N delegations.
    What: Validates args; on success issues a single
    ``ctx.dispatch_tool("delegate_task", {"tasks": [...], "role": "leaf"})`` and
    returns its JSON result; on validation/ctx error returns a JSON error.
    Test: invalid args -> JSON with "error"; valid args with a mocked ctx ->
    dispatch_tool called once with the batched tasks payload.
    """
    err = _validate(args)
    if err:
        return json.dumps({"error": err})

    if _CTX is None or not hasattr(_CTX, "dispatch_tool"):
        return json.dumps({"error": "orchestrate unavailable: plugin context not initialized"})

    tasks = _build_tasks(args["subtasks"])
    payload = {"tasks": tasks, "role": LEAF_ROLE}

    logger.info(
        "hermes-mpm orchestrate: goal=%r fanning out %d subtask(s) in parallel",
        (args.get("goal") or "").strip()[:80],
        len(tasks),
    )
    try:
        # Resolve parent_agent from the agent captured in pre_llm_call.
        # pre_llm_call fires before every LLM response, which is always
        # before any tool call in the same turn, so _captured_agent is
        # always set when handle() runs.
        # Fallback chain: captured agent → dispatch_tool (which tries _cli_ref).
        _parent_agent = _captured_agent

        from tools.registry import registry as _reg
        if _parent_agent is not None:
            result = _reg.dispatch(DELEGATE_TOOL, payload, parent_agent=_parent_agent)
        else:
            result = _CTX.dispatch_tool(DELEGATE_TOOL, payload)
    except Exception as exc:
        logger.warning("hermes-mpm orchestrate: delegate_task dispatch failed: %s", exc)
        return json.dumps({"error": f"delegate_task dispatch failed: {exc}"})

    return result if isinstance(result, str) else json.dumps(result)

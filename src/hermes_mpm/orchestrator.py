"""Orchestrate tool stub for hermes-mpm.

Why: MPM exposes a single ``hermes_mpm_orchestrate`` tool the orchestrator
profile can call to fan work out to archetypes. v0.1 wires the tool with a
real schema and validation so the contract is stable; the actual delegation
fan-out lands in the next task.
What: ``handle()`` validates args against the schema, echoes back the proposed
plan (objective + chosen archetype), and returns a JSON string — the same
shape real tool handlers return.
Test: Call handle({"objective": "x", "archetype": "ops"}); assert the parsed
JSON has status="planned" and archetype="ops". Call handle({}) -> JSON error.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from . import profiles

TOOLSET_NAME = "hermes-mpm"
TOOL_NAME = "hermes_mpm_orchestrate"

ORCHESTRATE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "objective": {
            "type": "string",
            "description": "The goal to orchestrate. What the user wants accomplished.",
        },
        "archetype": {
            "type": "string",
            "description": (
                "Optional target archetype to route the objective to "
                "(e.g. 'ops', 'engineer'). If omitted, MPM will choose."
            ),
        },
    },
    "required": ["objective"],
    "additionalProperties": False,
}


def handle(args: Dict[str, Any], **_kwargs) -> str:
    """Validate args and echo the proposed plan (STUB — no delegation yet).

    Why: Locks the tool's input/output contract so callers and tests are valid
    before real fan-out exists.
    What: Requires a non-empty ``objective``; validates an optional
    ``archetype`` against the shipped profile set; returns a JSON plan.
    Test: handle({"objective":"deploy"}) -> status="planned"; handle({}) ->
    JSON with an "error" key; handle({"objective":"x","archetype":"nope"}) ->
    JSON error naming the unknown archetype.
    """
    objective = (args.get("objective") or "").strip()
    if not objective:
        return json.dumps({"error": "Missing required parameter: objective"})

    archetype = (args.get("archetype") or "").strip() or None
    if archetype is not None and archetype not in profiles.list_archetypes():
        return json.dumps(
            {
                "error": f"Unknown archetype '{archetype}'",
                "available": profiles.list_archetypes(),
            }
        )

    return json.dumps(
        {
            "status": "planned",
            "stub": True,
            "objective": objective,
            "archetype": archetype,
            "note": "hermes-mpm v0.1 scaffold: orchestration plan echoed, not executed.",
        }
    )

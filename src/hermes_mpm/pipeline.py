"""Pipeline state tracking tools for hermes-mpm — 6-stage quality gate.

Provides tools for managing the bug-fix pipeline:
- pipeline_init: Start a pipeline run
- pipeline_transition: Move to next phase
- pipeline_record_evidence: Record gate evidence
- pipeline_verify_gate: Check gate conditions
- pipeline_status: Show pipeline state
- pipeline_recover: Handle failures

State is tracked in-memory keyed by task_id.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes_mpm.pipeline")

TOOLSET_NAME = "hermes-mpm"

# Pipeline phases in order
PHASES = [
    "research",
    "code-analysis",
    "implementation",
    "code-critic",
    "qa",
    "documentation",
]

# In-memory pipeline state: {task_id: {phase, status, evidence, ...}}
_pipelines: Dict[str, Dict[str, Any]] = {}
_pipelines_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_phase(current: str) -> str | None:
    """Return the next phase after *current*, or None if already at the end."""
    try:
        idx = PHASES.index(current)
        if idx + 1 < len(PHASES):
            return PHASES[idx + 1]
        return None
    except ValueError:
        return None


def _phase_index(phase: str) -> int:
    try:
        return PHASES.index(phase)
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

INIT_SCHEMA: Dict[str, Any] = {
    "name": "pipeline_init",
    "description": "Initialize a new pipeline run for a bug fix or feature. Sets up state tracking and resets any prior run for the same task_id.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Unique identifier for this pipeline run (e.g., issue number or UUID).",
            },
            "description": {
                "type": "string",
                "description": "Description of the bug or feature being worked on.",
            },
            "skip_phases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of phase names to skip (e.g. ['research'] when requirements are clear).",
            },
        },
        "required": ["task_id", "description"],
    },
}

TRANSITION_SCHEMA: Dict[str, Any] = {
    "name": "pipeline_transition",
    "description": "Transition the pipeline to the next phase. Validates that the current phase's gate has passed before allowing the move.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Pipeline task ID.",
            },
            "phase": {
                "type": "string",
                "enum": PHASES,
                "description": "Target phase to transition to.",
            },
        },
        "required": ["task_id", "phase"],
    },
}

RECORD_EVIDENCE_SCHEMA: Dict[str, Any] = {
    "name": "pipeline_record_evidence",
    "description": "Record evidence at the current pipeline phase gate. Evidence is used to verify that a gate passes before transitioning to the next phase.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Pipeline task ID.",
            },
            "phase": {
                "type": "string",
                "enum": PHASES,
                "description": "Phase this evidence belongs to.",
            },
            "type": {
                "type": "string",
                "enum": ["file", "constraint", "approach", "verdict", "test", "lint", "review", "test-suite", "documentation"],
                "description": "Evidence type.",
            },
            "key": {
                "type": "string",
                "description": "Short identifier for this evidence (e.g. 'auth-service', 'lint-pass').",
            },
            "value": {
                "type": "string",
                "description": "Description of what was found or done.",
            },
            "status": {
                "type": "string",
                "enum": ["pass", "fail", "pending"],
                "description": "pass = gate met, fail = gate blocked, pending = not yet checked.",
            },
        },
        "required": ["task_id", "phase", "type", "key", "value", "status"],
    },
}

VERIFY_GATE_SCHEMA: Dict[str, Any] = {
    "name": "pipeline_verify_gate",
    "description": "Verify whether all conditions for the current phase's gate have been met. Returns pass/fail with details on missing evidence.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Pipeline task ID.",
            },
            "phase": {
                "type": "string",
                "enum": PHASES,
                "description": "Phase whose gate to verify.",
            },
        },
        "required": ["task_id", "phase"],
    },
}

STATUS_SCHEMA: Dict[str, Any] = {
    "name": "pipeline_status",
    "description": "Show the current pipeline state: which phase is active, which gates have passed, and evidence summary.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Pipeline task ID. If omitted, lists all active pipelines.",
            },
        },
    },
}

RECOVER_SCHEMA: Dict[str, Any] = {
    "name": "pipeline_recover",
    "description": "Handle a pipeline failure by retrying, skipping the failed phase, or escalating to the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Pipeline task ID.",
            },
            "strategy": {
                "type": "string",
                "enum": ["retry", "skip", "escalate"],
                "description": "retry = re-run the current phase, skip = skip this phase and proceed, escalate = surface to user.",
            },
            "error_message": {
                "type": "string",
                "description": "Description of the error encountered.",
            },
        },
        "required": ["task_id", "strategy"],
    },
}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_init(args: Dict[str, Any], **_ctx: Any) -> str:
    """Initialize a pipeline run."""
    task_id = args["task_id"]
    description = args.get("description", "")
    skip_phases = args.get("skip_phases", [])

    with _pipelines_lock:
        _pipelines[task_id] = {
            "task_id": task_id,
            "description": description,
            "current_phase": "research",
            "phases": {p: {"status": "pending", "evidence": []} for p in PHASES},
            "skip_phases": skip_phases,
            "created_at": time.time(),
            "updated_at": time.time(),
            "status": "active",
        }
        # Mark skip phases as done
        for sp in skip_phases:
            if sp in _pipelines[task_id]["phases"]:
                _pipelines[task_id]["phases"][sp]["status"] = "skipped"
        # Find first non-skipped phase
        for p in PHASES:
            if p not in skip_phases:
                _pipelines[task_id]["current_phase"] = p
                _pipelines[task_id]["phases"][p]["status"] = "active"
                break

    return json.dumps({
        "success": True,
        "task_id": task_id,
        "current_phase": _pipelines[task_id]["current_phase"],
        "message": f"Pipeline initialized for task {task_id}",
    })


def handle_transition(args: Dict[str, Any], **_ctx: Any) -> str:
    """Transition to the next phase."""
    task_id = args["task_id"]
    target = args["phase"]

    with _pipelines_lock:
        pipeline = _pipelines.get(task_id)
        if not pipeline:
            return json.dumps({"success": False, "error": f"Pipeline {task_id} not found. Call pipeline_init first."})

        current = pipeline["current_phase"]
        if pipeline["phases"].get(current, {}).get("status") not in ("done", "skipped"):
            return json.dumps({
                "success": False,
                "error": f"Cannot transition from '{current}' to '{target}': current phase gate not passed. Verify the gate first.",
                "current_phase": current,
            })

        # Validate phase order
        if _phase_index(target) <= _phase_index(current):
            return json.dumps({
                "success": False,
                "error": f"Cannot go backward from '{current}' to '{target}'.",
                "current_phase": current,
            })

        if pipeline["phases"].get(target, {}).get("status") == "skipped":
            return json.dumps({
                "success": False,
                "error": f"Phase '{target}' was skipped. Cannot transition to a skipped phase.",
                "current_phase": current,
            })

        pipeline["current_phase"] = target
        pipeline["phases"][target]["status"] = "active"
        pipeline["updated_at"] = time.time()

    return json.dumps({
        "success": True,
        "task_id": task_id,
        "previous_phase": current,
        "current_phase": target,
        "message": f"Transitioned to '{target}'",
    })


def handle_record_evidence(args: Dict[str, Any], **_ctx: Any) -> str:
    """Record evidence at a gate."""
    task_id = args["task_id"]
    phase = args["phase"]
    evidence = {
        "type": args["type"],
        "key": args["key"],
        "value": args["value"],
        "status": args["status"],
        "timestamp": time.time(),
    }

    with _pipelines_lock:
        pipeline = _pipelines.get(task_id)
        if not pipeline:
            return json.dumps({"success": False, "error": f"Pipeline {task_id} not found."})

        if phase not in pipeline["phases"]:
            return json.dumps({"success": False, "error": f"Unknown phase '{phase}'."})

        # Replace existing evidence with same key, or append
        ev_list = pipeline["phases"][phase]["evidence"]
        replaced = False
        for i, ev in enumerate(ev_list):
            if ev["key"] == evidence["key"] and ev["type"] == evidence["type"]:
                ev_list[i] = evidence
                replaced = True
                break
        if not replaced:
            ev_list.append(evidence)

        pipeline["updated_at"] = time.time()

    return json.dumps({
        "success": True,
        "task_id": task_id,
        "phase": phase,
        "evidence_key": args["key"],
        "status": args["status"],
    })


def handle_verify_gate(args: Dict[str, Any], **_ctx: Any) -> str:
    """Verify that a phase gate has passed."""
    task_id = args["task_id"]
    phase = args["phase"]

    with _pipelines_lock:
        pipeline = _pipelines.get(task_id)
        if not pipeline:
            return json.dumps({"success": False, "error": f"Pipeline {task_id} not found."})

        if phase not in pipeline["phases"]:
            return json.dumps({"success": False, "error": f"Unknown phase '{phase}'."})

        phase_data = pipeline["phases"][phase]

        if phase_data["status"] == "skipped":
            return json.dumps({
                "success": True,
                "task_id": task_id,
                "phase": phase,
                "passed": True,
                "message": f"Phase '{phase}' was skipped.",
                "evidence_count": 0,
            })

        # Evidence is required: at least one evidence with status="pass"
        evidence = phase_data.get("evidence", [])
        passed_items = [e for e in evidence if e.get("status") == "pass"]
        failed_items = [e for e in evidence if e.get("status") == "fail"]
        pending = [e for e in evidence if e.get("status") == "pending"]

        all_pass = len(evidence) > 0 and len(failed_items) == 0 and len(pending) == 0

        if all_pass:
            phase_data["status"] = "done"

        return json.dumps({
            "success": True,
            "task_id": task_id,
            "phase": phase,
            "passed": all_pass,
            "message": "All evidence checks passed" if all_pass else f"{len(failed_items)} failed, {len(pending)} pending, {len(passed_items)} passed",
            "evidence_count": len(evidence),
            "passed_count": len(passed_items),
            "failed_count": len(failed_items),
            "pending_count": len(pending),
        })


def handle_status(args: Dict[str, Any], **_ctx: Any) -> str:
    """Show pipeline status."""
    task_id = args.get("task_id", "")

    with _pipelines_lock:
        if task_id:
            pipeline = _pipelines.get(task_id)
            if not pipeline:
                return json.dumps({"success": False, "error": f"Pipeline {task_id} not found."})

            result = {
                "task_id": pipeline["task_id"],
                "description": pipeline["description"],
                "current_phase": pipeline["current_phase"],
                "status": pipeline["status"],
                "phases": {},
            }
            for p, data in pipeline["phases"].items():
                result["phases"][p] = {
                    "status": data["status"],
                    "evidence_count": len(data.get("evidence", [])),
                    "passed_count": len([e for e in data.get("evidence", []) if e.get("status") == "pass"]),
                }
            return json.dumps({"success": True, "pipeline": result})
        else:
            # List all pipelines
            summary = {}
            for tid, p in _pipelines.items():
                summary[tid] = {
                    "description": p["description"][:80],
                    "current_phase": p["current_phase"],
                    "status": p["status"],
                }
            return json.dumps({"success": True, "pipelines": summary, "count": len(summary)})


def handle_recover(args: Dict[str, Any], **_ctx: Any) -> str:
    """Handle a pipeline failure."""
    task_id = args["task_id"]
    strategy = args["strategy"]
    error_message = args.get("error_message", "")

    with _pipelines_lock:
        pipeline = _pipelines.get(task_id)
        if not pipeline:
            return json.dumps({"success": False, "error": f"Pipeline {task_id} not found."})

        current = pipeline["current_phase"]

        if strategy == "retry":
            pipeline["phases"][current]["status"] = "active"
            pipeline["phases"][current]["evidence"] = []
            msg = f"Retrying phase '{current}'"
        elif strategy == "skip":
            pipeline["phases"][current]["status"] = "skipped"
            next_p = _next_phase(current)
            if next_p:
                pipeline["current_phase"] = next_p
                pipeline["phases"][next_p]["status"] = "active"
            msg = f"Skipped phase '{current}'"
        else:  # escalate
            pipeline["status"] = "blocked"
            msg = f"Escalated — pipeline blocked at '{current}'"

        pipeline["updated_at"] = time.time()

    return json.dumps({
        "success": True,
        "task_id": task_id,
        "strategy": strategy,
        "phase": current,
        "message": msg,
    })

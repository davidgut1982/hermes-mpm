"""Blast-radius classifier — pure-Python review-cost tiering, no LLM.

Why: Determine how much review a delegate_task deserves BEFORE paying for a
reviewer call. Trivial reads get waved through; destructive/privileged work and
batches get gated.
What: Inspects ``tool_name`` and the delegate_task ``goal``/``tasks`` args with a
keyword heuristic. Deterministic and offline.
Test: deploy/migrate/delete/auth/secret/prod -> elevated; multi-task -> merge_adjacent;
status/list/show -> trivial; anything else -> standard.
"""

from __future__ import annotations

from typing import Literal

BlastRadius = Literal["trivial", "standard", "elevated", "merge_adjacent"]

# Keywords that escalate to `elevated` — destructive, privileged, or prod-touching.
_ELEVATED_KEYWORDS = (
    "deploy", "migrate", "delete", "drop", "remove", "auth", "secret",
    "prod", "production", "password", "token", "key", "credential",
    "database", "db",
)

# Keywords that downgrade to `trivial` — read-only / inspection verbs.
_TRIVIAL_KEYWORDS = (
    "status", "show", "list", "check", "format", "read", "get",
    "fetch", "display", "describe",
)


def _goal_text(task: dict) -> str:
    """Concatenate the reviewable text fields of a single task/goal dict."""
    parts = []
    for field in ("goal", "task", "description", "prompt"):
        val = task.get(field)
        if isinstance(val, str):
            parts.append(val)
    return " ".join(parts).lower()


def _classify_goal(text: str) -> BlastRadius:
    """Classify a single goal string by keyword precedence (elevated > trivial)."""
    # Elevated wins: a destructive verb anywhere dominates.
    if any(kw in text for kw in _ELEVATED_KEYWORDS):
        return "elevated"
    if any(kw in text for kw in _TRIVIAL_KEYWORDS):
        return "trivial"
    return "standard"


def classify_blast_radius(tool_name: str, args: dict) -> BlastRadius:
    """Heuristic blast-radius tier for a tool call.

    Why: Gate cost control — cheap deterministic tiering before any reviewer call.
    What: For delegate_task, a multi-task batch is ``merge_adjacent``; otherwise
    classify by the goal's keywords (elevated > trivial > standard). Non-delegate
    tools default to ``standard``.
    Test: see test_gate.py tiering cases.
    """
    args = args or {}
    tasks = args.get("tasks")
    if isinstance(tasks, list) and tasks:
        if len(tasks) > 1:
            return "merge_adjacent"
        # Single-element batch: classify by that task's goal.
        first = tasks[0]
        text = _goal_text(first) if isinstance(first, dict) else str(first).lower()
        return _classify_goal(text)

    return _classify_goal(_goal_text(args))

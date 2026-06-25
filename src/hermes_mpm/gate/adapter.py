"""Review-gate adapter — wires the two Hermes seams to the fail-closed reviewer.

Design credit: gate/adapter pattern inspired by MahdiHedhli/HermesUltraCode (MIT).

Why: delegate_task is the one tool that spawns autonomous sub-work, so it is the
single choke point worth gating. Two seams cover both decision paths: the
``tool_request`` middleware applies the TIGHTEN path (mutates args), and the
``pre_tool_call`` hook applies the BLOCK path (returns an error message).
What: Filters to delegate_task only; tiers each call; reviews gated tiers once
per tool_call_id (memoized); enforces tighten-only; audits every decision. Batch
calls (tasks=[...]) review each task and block the whole call if any task blocks.
Test: memoization (1 reviewer call across both seams); no-op on non-delegate;
batch reviews each task; fail-closed override blocks all.
"""

from __future__ import annotations

import logging

from .audit import AuditStore
from .config import ReviewGateConfig
from .reviewer import ReviewerConfig, build_prompt, call_reviewer
from .tiering import classify_blast_radius
from .tighten import validate_tighten
from .verdict import Verdict, parse_verdict

logger = logging.getLogger("hermes_mpm.gate")

DELEGATE_TOOL = "delegate_task"


class ReviewGateAdapter:
    """Hermes plugin wiring: tool_request middleware (tighten) + pre_tool_call hook (block)."""

    def __init__(self, gate_config: ReviewGateConfig, audit_store: AuditStore,
                 *, fail_closed_override: bool = False,
                 fail_closed_reason: str = "") -> None:
        self._cfg = gate_config
        self._audit = audit_store
        self._memo: dict[str, Verdict] = {}  # tool_call_id -> verdict
        self._fail_closed_override = fail_closed_override
        self._fail_closed_reason = fail_closed_reason or "cross-lab guard active"
        self._reviewer_cfg = ReviewerConfig(
            provider=gate_config.reviewer_provider,
            model=gate_config.reviewer_model,
            base_url=gate_config.reviewer_base_url,
            api_key_env=gate_config.reviewer_api_key_env,
            timeout=gate_config.reviewer_timeout,
        )

    # ── seam: tighten path ──────────────────────────────────────────────────

    def middleware_callback(self, tool_name: str, args: dict, **kwargs) -> dict | None:
        """tool_request middleware: tighten path. Returns {"args": tightened} or None."""
        if tool_name != DELEGATE_TOOL:
            return None
        args = args or {}
        tool_call_id = kwargs.get("tool_call_id")

        verdict = self._get_or_review(tool_call_id, tool_name, args)
        if verdict.decision == "tighten" and verdict.added_constraints:
            tightened = self._apply_constraints(args, verdict.added_constraints)
            ok, _reason = validate_tighten(args, tightened)
            if ok:
                return {"args": tightened}
            # A tighten that fails the append-only proof must not pass through.
            return None
        return None

    # ── seam: block path ────────────────────────────────────────────────────

    def hook_callback(self, tool_name: str = "", args: dict | None = None, **kwargs) -> str | None:
        """pre_tool_call hook: block path. Returns a block message, or None to allow.

        Why: The engine's invoke_hook("pre_tool_call", ...) dispatches with
        tool_name= and args= as kwargs (see hermes_cli/plugins.py
        get_pre_tool_call_block_message). The prior signature used function_name/
        function_args which never matched, so every call raised TypeError and the
        gate silently crashed on every tool call.
        What: Accept tool_name/args to match the engine's actual kwarg names.
        Test: Call hook_callback(tool_name="delegate_task", args={...}) — must not
        raise TypeError/NameError and must return a block/None verdict from the
        reviewer path.
        """
        if tool_name != DELEGATE_TOOL:
            return None
        tool_call_id = kwargs.get("tool_call_id")

        verdict = self._get_or_review(tool_call_id, tool_name, args or {})
        if verdict.decision == "block":
            return f"[review-gate] BLOCKED: {verdict.reason}"
        return None

    # ── core review (memoized) ──────────────────────────────────────────────

    def _get_or_review(self, tool_call_id, tool_name: str, args: dict) -> Verdict:
        """Memoized review. Calls the reviewer exactly once per tool_call_id."""
        if tool_call_id and tool_call_id in self._memo:
            return self._memo[tool_call_id]

        verdict = self._review(tool_call_id, tool_name, args)

        if tool_call_id:
            self._memo[tool_call_id] = verdict
        return verdict

    def _review(self, tool_call_id, tool_name: str, args: dict) -> Verdict:
        """Run the actual review for a (possibly batched) delegate_task call."""
        # Cross-lab fail-closed override: block everything.
        if self._fail_closed_override:
            verdict = Verdict(decision="block", added_constraints=[],
                              reason=self._fail_closed_reason)
            self._record(tool_call_id, tool_name, args, "n/a", verdict)
            return verdict

        blast = classify_blast_radius(tool_name, args)
        if blast not in self._cfg.gated_tiers:
            verdict = Verdict(decision="allow", added_constraints=[], reason="")
            self._record(tool_call_id, tool_name, args, blast, verdict)
            return verdict

        tasks = args.get("tasks")
        if isinstance(tasks, list) and tasks:
            verdict = self._review_batch(tasks)
        else:
            verdict = self._review_one(args)

        self._record(tool_call_id, tool_name, args, blast, verdict)
        return verdict

    def _review_one(self, task_args: dict) -> Verdict:
        """Review a single delegate_task payload. Reviewer errors -> fail-closed block."""
        try:
            output = call_reviewer(build_prompt(task_args), self._reviewer_cfg)
            return parse_verdict(output)
        except Exception as exc:  # transport/HTTP/parse failure -> block
            logger.warning("hermes-mpm gate: reviewer call failed: %s", exc)
            return parse_verdict(None, error=str(exc))

    def _review_batch(self, tasks: list) -> Verdict:
        """Review each task independently; any block -> whole call blocks."""
        all_constraints: list[str] = []
        block_reasons: list[str] = []
        saw_tighten = False

        for task in tasks:
            task_args = task if isinstance(task, dict) else {"goal": str(task)}
            v = self._review_one(task_args)
            if v.decision == "block":
                block_reasons.append(v.reason)
            elif v.decision == "tighten":
                saw_tighten = True
                all_constraints.extend(v.added_constraints)

        if block_reasons:
            return Verdict(decision="block", added_constraints=[],
                           reason="; ".join(block_reasons))
        if saw_tighten:
            return Verdict(decision="tighten", added_constraints=all_constraints,
                           reason="reviewer added constraints (batch)")
        return Verdict(decision="allow", added_constraints=[], reason="")

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_constraints(args: dict, constraints: list[str]) -> dict:
        """Append reviewer constraints to a copy of args (never mutate in place)."""
        tightened = dict(args)
        existing = list(tightened.get("gate_constraints") or [])
        tightened["gate_constraints"] = existing + list(constraints)
        return tightened

    def _record(self, tool_call_id, tool_name: str, args: dict,
                blast: str, verdict: Verdict) -> None:
        try:
            self._audit.record(
                tool_call_id=str(tool_call_id or ""),
                tool_name=tool_name,
                args=args,
                blast_radius=blast,
                decision=verdict.decision,
                reason=verdict.reason,
                constraints=verdict.added_constraints,
            )
        except Exception as exc:  # audit must never break the gate
            logger.debug("hermes-mpm gate: audit write skipped: %s", exc)

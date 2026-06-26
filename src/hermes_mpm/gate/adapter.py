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

    def hook_callback(self, tool_name: str = "", args: dict | None = None, **kwargs) -> dict | None:
        """pre_tool_call hook: block path. Returns a block directive dict, or None.

        Why: The engine's get_pre_tool_call_block_message (hermes_cli/plugins.py
        ~1942) ONLY honors a dict of shape {"action": "block", "message": <str>};
        ``if not isinstance(result, dict): continue`` silently drops anything else,
        then requires ``result.get("action") == "block"``. The prior bare-string
        return was therefore dropped on the floor — the gate never blocked. Also,
        the engine dispatches with tool_name=/args= as kwargs, so those are the
        accepted parameter names.
        What: On a BLOCK verdict return {"action": "block", "message": <reason>}
        (keeping the "[review-gate] BLOCKED: ..." text in the message). On
        allow/tighten return None (allow-through; tighten is applied by middleware).
        Test: hook_callback(tool_name="delegate_task", args={elevated}) with a
        BLOCK verdict → dict with action=="block" and the reason in "message";
        allow/tighten → None.
        """
        if tool_name != DELEGATE_TOOL:
            return None
        tool_call_id = kwargs.get("tool_call_id")

        verdict = self._get_or_review(tool_call_id, tool_name, args or {})
        if verdict.decision == "block":
            return {
                "action": "block",
                "message": f"[review-gate] BLOCKED: {verdict.reason}",
            }
        return None

    def evaluate(self, tool_name: str, args: dict, tool_call_id=None) -> Verdict:
        """Run the gate verdict for a tool call (shared by hook + orchestrator).

        Why: Finding 2 — the orchestrate tool fans out delegate_task via the
        internal registry, which bypasses the pre_tool_call hook. To gate those
        subtasks with the SAME logic the hook uses, both paths must call one
        function. This is that single source of truth.
        What: Delegates to the memoized review path and returns the Verdict.
        Test: evaluate("delegate_task", {elevated goal}) with a mocked BLOCK
        reviewer → Verdict(decision="block").
        """
        if tool_name != DELEGATE_TOOL:
            return Verdict(decision="allow", added_constraints=[], reason="")
        return self._get_or_review(tool_call_id, tool_name, args or {})

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


# ── module-level shared gate access (Finding 2: one source of truth) ─────────
# The live adapter instance is registered here by register_gate() so the
# orchestrate tool can apply the SAME verdict logic the pre_tool_call hook uses.
_ACTIVE_ADAPTER: ReviewGateAdapter | None = None


def set_active_adapter(adapter: ReviewGateAdapter | None) -> None:
    """Register (or clear) the live gate adapter for module-level evaluate().

    Why: The orchestrate tool dispatches delegate_task internally (bypassing the
    pre_tool_call hook), so it needs a reference to the armed gate to evaluate
    subtasks with identical logic. register_gate() sets this; tests clear it.
    What: Sets the module global ``_ACTIVE_ADAPTER``.
    Test: set_active_adapter(a); evaluate(...) uses a. set_active_adapter(None);
    evaluate(...) returns allow.
    """
    global _ACTIVE_ADAPTER
    _ACTIVE_ADAPTER = adapter


def evaluate(tool_name: str, args: dict, tool_call_id=None) -> Verdict:
    """Evaluate a tool call against the armed gate; ALLOW if no gate is armed.

    Why: Finding 2 — single entry point both hook_callback and the orchestrator
    use, so fan-out subtasks are gated exactly as direct delegate_task calls are.
    What: Delegates to the active adapter's ``evaluate``; with no armed gate it
    returns an ALLOW verdict (degrade to pre-gate behavior rather than hard-block
    all fan-out, mirroring the disabled-gate path).
    Test: with an active adapter and a BLOCK reviewer → block; with no adapter →
    allow.
    """
    if _ACTIVE_ADAPTER is None:
        return Verdict(decision="allow", added_constraints=[], reason="")
    return _ACTIVE_ADAPTER.evaluate(tool_name, args, tool_call_id=tool_call_id)

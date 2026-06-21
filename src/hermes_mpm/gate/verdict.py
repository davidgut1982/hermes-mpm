"""Verdict parsing — FAIL-CLOSED structured decision from reviewer output.

Why: The gate must never default to "allow" on ambiguity. Any missing, empty,
errored, or unparseable reviewer output is treated as a BLOCK.
What: Parses ALLOW / TIGHTEN: <c> / BLOCK: <reason> lines into a Verdict.
Test: ALLOW -> allow; TIGHTEN -> tighten w/ constraint; garbage/None/empty/error -> block;
any BLOCK line -> block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Verdict:
    decision: Literal["allow", "tighten", "block"]
    added_constraints: list[str] = field(default_factory=list)
    reason: str = ""


def _block(reason: str) -> Verdict:
    return Verdict(decision="block", added_constraints=[], reason=reason)


def parse_verdict(reviewer_output: str | None, *, error: str | None = None) -> Verdict:
    """Parse reviewer output into a structured verdict. FAIL-CLOSED.

    Why: Structured decisions for the gate adapter; unknown/empty/error -> block.
    What: Scans lines for ALLOW / TIGHTEN: / BLOCK:. Any BLOCK -> block; else any
    TIGHTEN -> tighten (collecting constraints); else only ALLOW -> allow. No
    recognizable line -> block.
    Test: see test_gate.py verdict cases.
    """
    if error is not None:
        return _block(f"reviewer error: {error}")
    if reviewer_output is None:
        return _block("no reviewer output")
    if not reviewer_output.strip():
        return _block("empty reviewer output")

    saw_allow = False
    saw_tighten = False
    constraints: list[str] = []
    block_reasons: list[str] = []

    for raw_line in reviewer_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("BLOCK"):
            reason = line[len("BLOCK"):].lstrip(":").strip()
            block_reasons.append(reason or "blocked by reviewer")
        elif upper.startswith("TIGHTEN"):
            constraint = line[len("TIGHTEN"):].lstrip(":").strip()
            if not constraint:
                # A bare "TIGHTEN:" with no constraint text is MALFORMED.
                # Fail closed: an empty tighten is indistinguishable from ALLOW
                # and must never silently pass the call through unchanged.
                block_reasons.append("malformed tighten: empty constraint text")
            else:
                constraints.append(constraint)
                saw_tighten = True
        elif upper.startswith("ALLOW"):
            saw_allow = True

    # Any BLOCK line dominates (fail-closed).
    if block_reasons:
        return _block("; ".join(block_reasons))
    if saw_tighten:
        return Verdict(decision="tighten", added_constraints=constraints,
                       reason="reviewer added constraints")
    if saw_allow:
        return Verdict(decision="allow", added_constraints=[], reason="")

    # Nothing recognizable -> fail closed.
    return _block("unparseable reviewer output")

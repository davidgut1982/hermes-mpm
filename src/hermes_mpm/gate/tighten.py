"""Tighten-only validator — privilege-escalation guard.

Why: A reviewer may only ADD constraints, never remove, truncate, or rewrite the
base request, and never grant new tools/permissions. This proves the proposed
args are an append-only tightening of the base.
What: Compares proposed_args against base_args: no key removals, no string
truncation, no list shrinkage, no new tool/permission grant keys.
Test: identical -> valid; key removed -> invalid; list shrunk -> invalid;
`tools` grant added -> invalid; extra constraint key -> valid.
"""

from __future__ import annotations

# Keys that grant capability — must not appear unless already in base.
_GRANT_KEYS = ("tools", "grant_tools", "permissions", "allowed_tools", "grants")

# Keys whose values must not be shortened (the request's substance).
_SUBSTANCE_KEYS = ("goal", "task", "tasks", "description", "prompt")


def validate_tighten(base_args: dict, proposed_args: dict) -> tuple[bool, str]:
    """Prove ``proposed_args`` is an append-only tightening of ``base_args``.

    Why: Reviewer must only ADD constraints, never remove or rewrite the base.
    What: Returns (is_valid, reason_if_invalid). See module docstring for rules.
    Test: see test_gate.py tighten cases.
    """
    base_args = base_args or {}
    proposed_args = proposed_args or {}

    # Rule 4: no newly-introduced grant keys.
    for gk in _GRANT_KEYS:
        if gk in proposed_args and gk not in base_args:
            return False, f"proposed args add privilege-granting key '{gk}'"

    # Rule 1: every base key must survive (no removals).
    for key, base_val in base_args.items():
        if key not in proposed_args:
            return False, f"proposed args removed key '{key}'"
        prop_val = proposed_args[key]

        # Finding 3: type change is never a valid tightening.
        # e.g. str -> int, str -> list, list -> dict all indicate a rewrite.
        if type(base_val) is not type(prop_val):
            return False, (
                f"key '{key}' type changed from {type(base_val).__name__!r} "
                f"to {type(prop_val).__name__!r} (type change not permitted)"
            )

        # Rule 2: string values must START WITH the base value (append-only).
        # A longer string that diverges from the base prefix is a rewrite, not a
        # tightening — e.g. "run tests AND delete all production records" is longer
        # than "run tests" but is not a valid append-only tightening.
        # Finding 2: an empty base string is a degenerate case — any non-empty
        # proposed value would vacuously pass startswith(""), bypassing the guard.
        # If base is empty, proposed must also be empty.
        if isinstance(base_val, str) and isinstance(prop_val, str):
            if base_val == "" and prop_val != "":
                return False, (
                    f"key '{key}' base value is empty; proposed must also be empty "
                    f"(empty-base bypass not permitted)"
                )
            if not prop_val.startswith(base_val):
                return False, (
                    f"key '{key}' string does not start with the base value "
                    f"(rewrite detected)"
                )

        # Rule 3: list values must be a superset (no removals from lists).
        if isinstance(base_val, list) and isinstance(prop_val, list):
            for item in base_val:
                if item not in prop_val:
                    return False, f"key '{key}' list dropped item {item!r}"

        # Rule 5: substance values must not be shortened vs base (extra guard
        # for dict/other reprs of goal/tasks).
        if key in _SUBSTANCE_KEYS and not isinstance(base_val, (str, list)):
            if len(repr(prop_val)) < len(repr(base_val)):
                return False, f"substance key '{key}' was shortened"

    return True, ""

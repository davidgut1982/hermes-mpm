"""`hermes mpm` CLI subcommand for hermes-mpm.

Why: Operators need a no-LLM way to inspect MPM state. v0.1 ships a REAL
``list-profiles`` (reads the shipped archetypes) plus a ``routing`` stub so the
subcommand tree is in place for the next task.
What: ``setup(parser)`` builds the argparse sub-subcommands; ``handle(args)``
dispatches to the matching action and returns an int exit code.
Test: Build a parser via setup(), parse ["list-profiles"], call handle(args);
assert it prints the 22 archetypes and returns 0. Parse ["routing"] -> prints
the not-implemented notice and returns 0.
"""

from __future__ import annotations

import argparse

from . import profiles


def setup(parser: argparse.ArgumentParser) -> None:
    """Populate the ``hermes mpm`` subparser.

    Why: The plugin host calls this with the ``mpm`` subparser; we attach our
    own action sub-subparsers here.
    What: Adds ``list-profiles`` and ``routing`` actions, each wiring back to
    handle() via the shared dispatcher.
    Test: After setup(), parser.parse_args(["list-profiles"]).mpm_action ==
    "list-profiles".
    """
    sub = parser.add_subparsers(dest="mpm_action", metavar="<action>")

    sub.add_parser(
        "list-profiles",
        help="List the shipped agent profile archetypes (no LLM).",
    )
    sub.add_parser(
        "routing",
        help="Show MPM tier-routing state (stub — not implemented in v0.1).",
    )

    # Default action when `hermes mpm` is called bare.
    parser.set_defaults(mpm_action=None)


def handle(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``hermes mpm`` invocation.

    Why: Single entry point the host calls via set_defaults(func=handle).
    What: Routes on args.mpm_action; returns 0 on success, 2 on unknown action.
    Test: handle(Namespace(mpm_action="list-profiles")) prints archetypes,
    returns 0; handle(Namespace(mpm_action="routing")) returns 0.
    """
    action = getattr(args, "mpm_action", None)

    if action in (None, "list-profiles"):
        return _list_profiles()
    if action == "routing":
        print("hermes mpm routing: not implemented in v0.1 (scaffold).")
        return 0

    print(f"Unknown mpm action: {action!r}")
    return 2


def _list_profiles() -> int:
    """Print the shipped archetypes with model + toolset count.

    Why: The REAL, demonstrable capability of the v0.1 scaffold.
    What: Loads profiles and prints a stable, sorted table.
    Test: Output contains "ops" and a trailing count line "22 archetype(s)".
    """
    profs = profiles.load_profiles()
    names = profiles.list_archetypes()
    print(f"hermes-mpm: {len(names)} shipped agent profile archetype(s)\n")
    for name in names:
        prof = profs.get(name, {})
        model = prof.get("model", "?")
        toolsets = prof.get("toolsets", []) or []
        print(f"  {name:<18} {len(toolsets):>2} toolset(s)  model={model}")
    print(f"\n{len(names)} archetype(s)")
    return 0

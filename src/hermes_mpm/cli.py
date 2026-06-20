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

from . import profiles, routing


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
    routing_p = sub.add_parser(
        "routing",
        help="Dry-run classify a sample message: print grouping + tier + model.",
    )
    routing_p.add_argument("message", nargs="?", default="", help="Sample message text.")
    routing_p.add_argument(
        "--platform",
        default="telegram",
        help="Origin platform (telegram/api/cron/cli). Default: telegram.",
    )
    routing_p.add_argument(
        "--profile",
        default=None,
        help="Optional agent profile influencing routing (e.g. engineer).",
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
        return _routing(args)

    print(f"Unknown mpm action: {action!r}")
    return 2


def _routing(args: argparse.Namespace) -> int:
    """Dry-run-classify a sample message and print grouping + tier + model.

    Why: A no-LLM way to see exactly which tier/model a message would route to,
    for debugging routing config without touching the gateway.
    What: Calls routing.classify on the message+platform(+profile), resolves the
    tier's default model, and prints the three values. CLI platform is reported
    as opted-out (routing won't fire there at runtime).
    Test: handle(Namespace(mpm_action="routing", message="is plex up?",
    platform="telegram")) prints "cheap_workhorse" and returns 0.
    """
    message = getattr(args, "message", "") or ""
    platform = getattr(args, "platform", "telegram") or "telegram"
    profile = getattr(args, "profile", None)

    if not message.strip():
        print('usage: hermes mpm routing "<message>" [--platform telegram] [--profile name]')
        return 2

    grouping, tier = routing.classify(message, platform=platform, profile=profile)
    tier_cfg = routing.DEFAULT_TIERS.get(tier, {})
    model = tier_cfg.get("model", "?")
    enabled = routing._platform_enabled({}, platform)

    print(f"message:   {message!r}")
    print(f"platform:  {platform}" + ("" if enabled else "  (routing opted-out at runtime)"))
    if profile:
        print(f"profile:   {profile}")
    print(f"grouping:  {grouping}")
    print(f"tier:      {tier}")
    print(f"model:     {model}")
    return 0


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

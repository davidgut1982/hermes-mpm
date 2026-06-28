"""`hermes mpm` CLI subcommand for hermes-mpm.

Why: Operators need a no-LLM way to inspect MPM state. v0.1 ships a REAL
``list-profiles`` (reads the shipped archetypes) plus a ``routing`` stub so the
subcommand tree is in place for the next task.
What: ``setup(parser)`` builds the argparse sub-subcommands; ``handle(args)``
dispatches to the matching action and returns an int exit code.
Test: Build a parser via setup(), parse ["list-profiles"], call handle(args);
assert it prints the 22 archetypes and returns 0. Parse ["routing"] -> prints
the not-implemented notice and returns 0. Parse ["gate-status"] -> prints gate
config summary and returns 0 (OK/DISABLED) or non-zero (WARN/misconfigured).
"""

from __future__ import annotations

import argparse
import time

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

    sub.add_parser(
        "gate-status",
        help=(
            "Print the configured review-gate state and static seam check. "
            "Exit 0 on OK or DISABLED; non-zero on misconfiguration (same-lab)."
        ),
    )

    runs_p = sub.add_parser(
        "runs",
        help="List tracked subagent runs (no LLM): status, age, duration, goal.",
    )
    runs_p.add_argument(
        "--status",
        default=None,
        help="Filter by status: running|done|failed|crashed|timed_out.",
    )
    runs_p.add_argument(
        "--session",
        default=None,
        help="Filter by parent session id.",
    )
    runs_p.add_argument(
        "--since",
        default=None,
        help="Only runs started within this window, e.g. 1h, 24h, 7d.",
    )
    runs_p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max rows to show (newest first). Default: 50.",
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
    if action == "gate-status":
        return _gate_status_from_live_config()
    if action == "runs":
        return _runs(args)

    print(f"Unknown mpm action: {action!r}")
    return 2


def _parse_since(since: str | None) -> int | None:
    """Convert a ``1h``/``24h``/``7d``/``30m`` window into an epoch cutoff.

    Why: Operators think in relative windows ("runs in the last 24h"), not epoch
    seconds; this maps the shorthand to ``now - delta`` for query_runs(since=).
    What: Parses ``<int><s|m|h|d>`` (or a bare int = seconds) and returns the
    epoch cutoff, or None when ``since`` is empty/unparseable (no filter).
    Test: ``test_runs_since_filter_parses_duration`` exercises 24h and 7d.
    """
    if not since:
        return None
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    raw = since.strip().lower()
    try:
        if raw[-1] in units:
            delta = int(raw[:-1]) * units[raw[-1]]
        else:
            delta = int(raw)
    except (ValueError, IndexError):
        return None
    return int(time.time()) - delta


def _fmt_age(seconds: float) -> str:
    """Render a compact age like ``5s``/``3m``/``2h``/``4d``.

    Why: A fixed-width relative age reads faster than absolute timestamps in a
    terminal table.
    What: Returns the largest single unit <= the duration.
    Test: Implicitly via ``test_runs_formats_rows`` (output is non-empty).
    """
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _runs(args: argparse.Namespace) -> int:
    """Print tracked subagent runs as a compact table (no LLM).

    Why: The operator-facing read surface over the run DB — answers "what ran /
    is running / crashed" across restarts without touching the gateway.
    What: Queries runs_db.query_runs with the parsed filters and prints a fixed
    -width table (short run id, status, profile/role, age, duration, goal). Prints
    a clean notice and returns 0 when there are no matching runs.
    Test: ``test_runs_empty`` + ``test_runs_formats_rows`` + ``test_runs_status_filter``.
    """
    from . import runs_db

    status = getattr(args, "status", None)
    session = getattr(args, "session", None)
    since = _parse_since(getattr(args, "since", None))
    limit = getattr(args, "limit", 50) or 50

    try:
        rows = runs_db.query_runs(status=status, session=session, since=since, limit=limit)
    except Exception as exc:  # DB unreadable — report, don't crash the CLI
        print(f"runs: could not read run DB: {exc}")
        return 1

    if not rows:
        print("No runs match.")
        return 0

    now = time.time()
    header = f"{'RUN':<10} {'STATUS':<9} {'PROFILE':<12} {'AGE':>5} {'DUR':>7}  GOAL"
    print(header)
    print("-" * len(header))
    for r in rows:
        run_id = (r.get("run_id") or "")[:8]
        rstatus = r.get("status") or "?"
        profile = (r.get("profile") or r.get("role") or "-")[:12]
        started = r.get("started_at") or now
        age = _fmt_age(now - started)
        dur_ms = r.get("duration_ms")
        dur = f"{dur_ms / 1000:.1f}s" if isinstance(dur_ms, (int, float)) else "-"
        goal = (r.get("goal") or "").replace("\n", " ")
        if len(goal) > 48:
            goal = goal[:47] + "…"
        print(f"{run_id:<10} {rstatus:<9} {profile:<12} {age:>5} {dur:>7}  {goal}")

    print(f"\n{len(rows)} run(s)")
    return 0


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


def _gate_status_from_live_config() -> int:
    """Load the live Hermes config and call _gate_status_handler.

    Why: The CLI subcommand needs a live config path for real operator use.
    What: Best-effort loads the hermes_mpm config namespace; falls back to {} on
    failure, so gate-status still runs (and reports DISABLED/defaults) even without
    a full Hermes install.
    Test: Indirectly tested via test_gate_status_cli_subcommand_dispatches; the
    live config load is expected to fall back gracefully in test environments.
    """
    try:
        from hermes_cli.config import cfg_get, load_config

        config = load_config()
        cfg_ns = cfg_get(config, "plugins", "entries", "hermes_mpm", default={})
        raw_config = {"hermes_mpm": cfg_ns if isinstance(cfg_ns, dict) else {}}
    except Exception:
        raw_config = {}
    return _gate_status_handler(raw_config)


def _gate_status_handler(raw_config: dict) -> int:
    """Print the configured gate state and return a scriptable exit code.

    Why: Operators need a queryable, no-LLM check that reports intended config
    state + static validation without running the gate or touching any live seams.
    What: Loads ReviewGateConfig from raw_config; derives orchestrator and reviewer
    labs; prints a summary including seam names; exits 0 on OK or DISABLED, non-zero
    on misconfiguration (same-lab reviewer → would fail-closed on arm).
    Test: cross-lab config -> OK + exit 0; same-lab -> WARN + non-zero;
    enabled=False -> DISABLED + exit 0. All tested in test_gate.py.
    """
    from .gate import derive_lab
    from .gate.config import load_gate_config

    cfg = load_gate_config(raw_config)

    # Seam names the gate registers — static, must match what register_gate uses.
    _SEAM_HOOK = "pre_tool_call"
    _SEAM_MIDDLEWARE = "tool_request"

    if not cfg.enabled:
        print("gate config: DISABLED")
        print("  enabled:     false")
        print(f"  seams used:  {_SEAM_HOOK} (hook), {_SEAM_MIDDLEWARE} (middleware)")
        print("  (seams must exist in the running core to arm, but gate is disabled)")
        return 0

    orchestrator_lab = derive_lab(cfg.orchestrator_model)
    reviewer_lab = derive_lab(cfg.reviewer_model)
    same_lab = bool(orchestrator_lab) and bool(reviewer_lab) and orchestrator_lab == reviewer_lab
    cross_lab_note = (
        "SAME (would fail-closed on arm)"
        if same_lab
        else "independent" if (orchestrator_lab and reviewer_lab)
        else f"UNKNOWN (orchestrator_lab={orchestrator_lab!r} — would fail-closed on arm)"
    )

    orch_lab_str = orchestrator_lab or "(unknown)"
    rev_lab_str = reviewer_lab or "(unknown)"
    orch_model_str = cfg.orchestrator_model or "(not set)"
    print("  enabled:          true")
    print(f"  reviewer:         {cfg.reviewer_model}  (lab={rev_lab_str})")
    print(f"  orchestrator:     {orch_model_str}  (lab={orch_lab_str})")
    print(f"  lab relationship: {cross_lab_note}")
    print(f"  gated_tiers:      {cfg.gated_tiers}")
    print(f"  fail_closed:      {str(cfg.fail_closed).lower()}")
    print(f"  audit_path:       {cfg.audit_path}")
    print(f"  seams used:       {_SEAM_HOOK} (hook, load-bearing block seam)")
    print(f"                    {_SEAM_MIDDLEWARE} (middleware, tighten-only path)")
    print("  note: seams must exist in the running core to arm")

    if same_lab:
        print(
            f"gate config: WARN (reviewer lab == orchestrator lab "
            f"'{orchestrator_lab}' — would fail-closed)"
        )
        return 1

    if not orchestrator_lab:
        print("gate config: WARN (orchestrator lab unknown — would fail-closed on arm)")
        return 1

    print("gate config: OK (independent reviewer, will arm)")
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

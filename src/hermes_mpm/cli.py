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
import sys
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
        help="Only runs started within this window; <int><s|m|h|d>, e.g. 30m, 24h, 7d.",
    )
    runs_p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max rows to show (newest first), 1-1000. Default: 50.",
    )

    parallelism_p = sub.add_parser(
        "parallelism",
        help=(
            "Show the tool-call batch rate (is parallelism working?, no LLM): "
            "overall + per-model, from per-turn tool-call counts."
        ),
    )
    parallelism_p.add_argument(
        "--since",
        default=None,
        help="Only turns within this window; <int><s|m|h|d>, e.g. 30m, 24h, 7d.",
    )
    parallelism_p.add_argument(
        "--model",
        default=None,
        help="Restrict the rate to a single model id.",
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
    if action == "parallelism":
        return _parallelism(args)

    print(f"Unknown mpm action: {action!r}")
    return 2


def _parse_since(since: str | None) -> int | None:
    """Convert a ``30m``/``1h``/``24h``/``7d`` window into an epoch cutoff.

    Why: Operators think in relative windows ("runs in the last 24h"), not epoch
    seconds; this maps the shorthand to ``now - delta`` for query_runs(since=). A
    non-empty value that we cannot parse must NOT silently mean "no filter" — that
    would dump the full list and mislead the operator — so it raises instead.
    What: Empty/None -> None (no filter). A valid ``<positive-int><s|m|h|d>`` ->
    the epoch cutoff. Anything else (bad unit, unit-less bare int like ``24``,
    junk) raises ValueError with a fix-it message.
    Test: ``test_runs_since_parses_to_correct_cutoff`` (cutoffs) +
    ``test_runs_since_unparseable_errors`` (5x/1hh/bare 24/abc/h raise).
    """
    if not since:
        return None
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    raw = since.strip().lower()
    unit = raw[-1:] if raw else ""
    if unit not in units:
        raise ValueError(
            f"invalid --since {since!r}; use forms like 30m, 24h, 7d "
            "(an integer followed by s/m/h/d)"
        )
    try:
        amount = int(raw[:-1])
    except ValueError as exc:
        raise ValueError(
            f"invalid --since {since!r}; use forms like 30m, 24h, 7d "
            "(an integer followed by s/m/h/d)"
        ) from exc
    if amount <= 0:
        raise ValueError(f"invalid --since {since!r}; the value must be a positive integer")
    return int(time.time()) - amount * units[unit]


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
    a clean notice and returns 0 when there are no matching runs. Rejects an
    unparseable ``--since`` or a ``--limit`` < 1 with a stderr error + exit 2,
    rather than silently dumping an unfiltered/over-broad list.
    Test: ``test_runs_empty`` + ``test_runs_formats_rows`` + ``test_runs_status_filter``
    + ``test_runs_since_unparseable_errors`` + ``test_runs_limit_zero_errors``.
    """
    from . import runs_db

    status = getattr(args, "status", None)
    session = getattr(args, "session", None)
    try:
        since = _parse_since(getattr(args, "since", None))
    except ValueError as exc:
        print(f"runs: {exc}", file=sys.stderr)
        return 2

    limit = getattr(args, "limit", 50)
    if limit is None or limit < 1:
        print(
            f"runs: invalid --limit {limit!r}; must be a positive integer (1-1000)",
            file=sys.stderr,
        )
        return 2
    limit = min(limit, 1000)

    try:
        rows = runs_db.query_runs(status=status, session=session, since=since, limit=limit)
    except Exception as exc:  # DB unreadable — report, don't crash the CLI
        print(f"runs: could not read run DB: {exc}")
        return 1

    if not rows:
        print("No runs match.")
        return 0

    now = time.time()
    header = (
        f"{'RUN':<10} {'STATUS':<9} {'PROFILE':<12} {'AGE':>5} {'DUR':>7} {'BATCH':>5}  GOAL"
    )
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
        # BATCH = the run's largest single-turn tool-call count (max_batch_size).
        # >1 means the run batched calls in parallel internally; "-" if never set.
        mbs = r.get("max_batch_size")
        batch = str(mbs) if isinstance(mbs, int) and mbs > 0 else "-"
        goal = (r.get("goal") or "").replace("\n", " ")
        if len(goal) > 48:
            goal = goal[:47] + "…"
        print(f"{run_id:<10} {rstatus:<9} {profile:<12} {age:>5} {dur:>7} {batch:>5}  {goal}")

    print(f"\n{len(rows)} run(s)")
    return 0


def _parallelism(args: argparse.Namespace) -> int:
    """Print the tool-call batch rate (overall + per model) — no LLM.

    Why: The operator-facing answer to "is parallelism working?" — the fraction
    of tool-emitting turns that batched more than one call. Computed from the
    durable per-turn tool-call counts (turn_batches), replacing the old buggy
    ad-hoc classifier with a trustworthy SELECT/GROUP BY number.
    What: Resolves an optional ``--since`` window (same shorthand as ``runs``) and
    ``--model`` filter, calls runs_db.batch_stats, and prints the overall rate +
    tool-turn count + a per-model breakdown. Prints a clean notice and returns 0
    when there is no data. Rejects an unparseable ``--since`` (stderr + exit 2)
    and reports a DB read failure (exit 1) instead of crashing.
    Test: ``test_parallelism_*`` in test_parallelism_cli.py.
    """
    from . import runs_db

    try:
        since = _parse_since(getattr(args, "since", None))
    except ValueError as exc:
        print(f"parallelism: {exc}", file=sys.stderr)
        return 2
    model = getattr(args, "model", None)

    try:
        stats = runs_db.batch_stats(since=since, model=model)
    except Exception as exc:  # DB unreadable — report, don't crash the CLI
        print(f"parallelism: could not read run DB: {exc}")
        return 1

    tool_turns = stats["tool_turns"]
    if not tool_turns:
        print("No tool-call turns recorded yet (no batch data).")
        return 0

    multi = stats["multi_tool_turns"]
    rate = stats["batch_rate"]
    print("MPM parallelism (tool-call batching)")
    print(f"  overall batch rate: {rate * 100:.1f}%  ({multi}/{tool_turns} tool-turns batched >1)")
    print("  by model:")
    header = f"    {'MODEL':<28} {'RATE':>7} {'BATCHED':>8} {'TOOL-TURNS':>11}"
    print(header)
    # model is nullable (the hook records model=kw.get("model"), which can be
    # None), so sort on a None-safe key — a bare sorted() raises TypeError when
    # the store mixes a NULL-model turn with named-model turns.
    for name, m in sorted(stats["by_model"].items(), key=lambda kv: kv[0] or ""):
        mrate = m["batch_rate"] * 100
        print(
            f"    {(name or '?'):<28} {mrate:>6.1f}% "
            f"{m['multi_tool_turns']:>8} {m['tool_turns']:>11}"
        )
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
        else "independent"
        if (orchestrator_lab and reviewer_lab)
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

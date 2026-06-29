# hermes-mpm

Multi-agent profile management (MPM) plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

**Status: v1 — capabilities live.** Tiered model routing, a parallel
orchestrator, and deterministic intent fast-paths, all on top of the shipped
profile set.

## Capabilities

- **Tiered model routing** (`routing.py`) — a pure-Python deterministic
  classifier (zero extra LLM calls) decides a tier per inbound request and pins
  that tier's model onto the gateway session before the agent is built. Cheap
  models answer cheap requests; strong models answer hard ones.
- **Parallel orchestrator** (`orchestrator.py`) — the `hermes_mpm_orchestrate`
  tool fans caller-supplied subtasks out to agent profiles **in parallel** via
  one batched `delegate_task` call (native ThreadPoolExecutor), instead of N
  serial delegations.
- **Intent fast-paths** (`intent.py`) — weather / time / disk / service-status
  phrasings are rewritten to deterministic slash commands (`/weather`, `/time`,
  `/diskfree`, `/svcstatus`) that answer in-process — no LLM for those turns.
- **Profiles** (`profiles.py`) — ships the default set of 22 agent archetypes.
- **Run tracking** (`runs_db.py`) — every subagent delegation (sync +
  async/background) is recorded to a durable SQLite DB so `hermes mpm runs` can
  answer "what ran / is running / crashed" across gateway restarts.
- **Batch telemetry** (`runs_db.py`) — a `post_api_request` hook records each
  LLM turn's assistant tool-call count (main agent + every subagent) so
  `hermes mpm parallelism` can answer "is parallelism working?" — the fraction
  of tool-emitting turns that batched more than one call — from a durable
  SELECT/GROUP BY, not a fragile classifier.
- **CLI** — `hermes mpm list-profiles`, `hermes mpm routing "<msg>"`
  (dry-run classifier), `hermes mpm runs` (tracked-run table), and
  `hermes mpm parallelism` (tool-call batch rate, overall + per-model).
- **Dashboard panel** (`dashboard/`) — an "MPM Runs" tab for the Hermes
  dashboard that surfaces the same run history in the browser via a
  **read-only** API, so an operator can watch live/finished subagent runs
  without the CLI.

## How routing steers the model

On `pre_gateway_dispatch`, the routing handler:

1. classifies the request → `(grouping, tier)` (pure function `classify()`),
2. resolves the tier's complete provider bundle
   `{model, provider, api_key, base_url, api_mode}` from config + the
   `OPENROUTER_API_KEY` env,
3. writes that bundle into the gateway's `_session_model_overrides[session_key]`
   (session key from `gateway._session_key_for_source(event.source)`),
4. calls `gateway._evict_cached_agent(session_key)` so the next build uses it,
5. returns `None` (never rewrites text).

Because the bundle includes a real `api_key`, the gateway's
`_resolve_session_agent_runtime` fast-path returns it directly — so a routed
profile's model **supersedes** the profile's own pinned model
(`profile_precedence: routing_wins`). Profiles not in `profile_tier_map` keep
their own model (routing writes nothing for them).

The routing handler respects an existing manual `/model` override (it never
stomps an operator's explicit pin) and is opt-out on CLI by default.

## Decision precedence (highest first)

1. explicit `/tier <name>` prefix
2. platform rule — `cron` / low-urgency → `free_background`
3. `profile_tier_map` (routing wins for listed profiles)
4. complexity up-route — any complexity keyword → `strong`
5. simplicity down-route — short + status/show/check/list (or "is X up?") →
   `cheap_workhorse`
6. default → `main`

## Install (into a Hermes venv)

```bash
pip install -e .
```

Enable + configure in `config.yaml` (see the schema below). Confirm with
`hermes plugins list` and `hermes mpm routing "is plex up?" --platform telegram`.

## Configuration schema (`hermes_mpm` block)

Also disable the core's built-in router so MPM is the single authority:

```yaml
smart_model_routing:
  enabled: false

hermes_mpm:
  profile_precedence: routing_wins          # routing supersedes profile model

  tiers:                                     # tier -> model
    free_background:
      model: "meta-llama/llama-3.3-70b-instruct:free"
      fallbacks: ["openai/gpt-oss-120b:free", "qwen/qwen3-235b-a22b-2507"]
    cheap_workhorse: { model: "qwen/qwen3-235b-a22b-2507" }
    main:            { model: "deepseek/deepseek-v4-flash" }
    strong:          { model: "deepseek/deepseek-v4-pro" }
    max:             { model: "anthropic/claude-sonnet-4.6" }

  profile_tier_map:                          # listed profiles get RE-ROUTED
    describer: free_background
    kb: free_background
    memory: free_background
    receipts: free_background
    ocr: free_background
    engineer: strong
    debugger: strong
    homelab: strong
    mcp-builder: strong

  bulk_profiles: [describer, kb, memory, receipts, ocr]

  complexity_keywords:                       # any present -> strong (up-route)
    [implement, debug, refactor, diagnose, migrate, architect, broken,
     failing, design, optimize]
  simple_keywords: [status, show, check, list]   # short + present -> cheap

  thresholds:                                # simplicity down-route gate
    max_chars: 600
    max_words: 120

  platforms:                                 # routing fires on these
    cli:        { enabled: false }           # CLI opt-out by default
    telegram:   { enabled: true }
    api_server: { enabled: true }
    cron:       { enabled: true }

  openrouter:                                # provider creds for the override
    provider: openrouter
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: OPENROUTER_API_KEY          # secret read from env, never hardcoded

  runs:                                      # run-tracking DB
    retention_days: 30                       # purge ended runs older than this; <=0 disables
```

A working example lives at `config.example.yaml`.

## Orchestrate tool

`hermes_mpm_orchestrate({goal, subtasks: [{profile, goal, context?}, ...]})`
validates the subtasks (non-empty; each needs a known `profile` + `goal`), then
runs each subtask through the review gate before dispatch — the internal fan-out
bypasses the engine's `pre_tool_call` hook, so the tool applies the same verdict
logic itself. Surviving subtasks go out in one batched `delegate_task` with
`role: leaf`; any gate-blocked subtasks are reported back in a `blocked` array
alongside the aggregated result. If every subtask is blocked, nothing is
dispatched and the tool returns `{"error": ..., "blocked": [...]}`. v1 is
caller-supplied subtasks (no LLM decomposition).

## Run tracking

Subagent lifecycle hooks (`subagent_start` / `subagent_stop`, plus a
`pre_llm_call` fallback that closes async/background runs when their
completion marker re-enters the turn) write every delegation to a durable
SQLite DB at `<hermes_home>/mpm_runs.db`. A `post_api_request` hook fires once
per LLM response (main agent + every subagent) and records that turn's
assistant tool-call count to the `turn_batches` table — the parallelism signal
(see [Parallelism telemetry](#parallelism-telemetry)) — folding it into the
matching subagent run's `max_batch_size` / `turn_count` columns. Schema creation happens at plugin
load (in every process, so the CLI and dashboard see the current schema). The
mutating restart-orphan sweep (`running` rows left by a dead process are marked
`crashed`) and retention purge are **deferred** to a lazy once-per-process pass
fired from the **first gateway hook** — not at plugin load, because `register()`
runs before the gateway sets `_HERMES_GATEWAY=1`, so a load-time guard would
always miss and the sweep would never fire. Run by hook, the env is set by then;
the pass is gated on `_HERMES_GATEWAY=1` at runtime and latched to run exactly
once, so the gateway sweeps once and the CLI and dashboard (which never set the
env) never sweep — preserving the cross-process fix that keeps them from
corrupting live runs.

Inspect runs with the no-LLM CLI:

```bash
hermes mpm runs                       # newest 50
hermes mpm runs --status crashed      # filter by status (running|done|failed|crashed|timed_out)
hermes mpm runs --session <id>        # filter by parent session
hermes mpm runs --since 24h           # relative window: <int><s|m|h|d>
hermes mpm runs --limit 200           # 1-1000, newest first (default 50)
```

It prints a fixed-width table (short run id, status, profile/role, age,
duration, BATCH, goal), where BATCH is the run's largest single-turn tool-call
count (`max_batch_size`; `-` when the run never emitted a tool call) — `>1`
means the run batched calls in parallel internally. An unparseable `--since` or
a `--limit` below 1 is rejected with a stderr error and exit code 2, rather than
silently dumping an over-broad list.

Retention defaults to 30 days; override it with `hermes_mpm.runs.retention_days`
(see the configuration schema). It purges both ended runs and old `turn_batches`
rows past the cutoff. A non-positive value disables purging.

### Parallelism telemetry

`hermes mpm parallelism` answers "is parallelism working?" — the batch rate,
the fraction of tool-emitting turns that batched more than one tool call,
overall and per model. It reads the durable per-turn tool-call counts
(`turn_batches`, written by the `post_api_request` hook) with no LLM:

```bash
hermes mpm parallelism                 # overall + per-model batch rate
hermes mpm parallelism --since 24h     # relative window: <int><s|m|h|d>
hermes mpm parallelism --model <id>    # restrict to a single model id
```

It prints the overall rate (`multi/tool-turns batched >1`) followed by a
per-model breakdown. With no tool-call turns recorded yet it prints a notice and
exits 0; an unparseable `--since` is rejected (stderr, exit 2) and a DB read
failure is reported (exit 1) rather than crashing.

### Dashboard panel

The same run history is also viewable in the Hermes dashboard. The plugin ships
a self-contained dashboard panel under `dashboard/` (declared in
`dashboard/manifest.json` as the **MPM Runs** tab, mounted after the kanban
tab): a prebuilt frontend bundle (`dashboard/dist/`) plus a backend
`dashboard/plugin_api.py` the host mounts at `/api/plugins/mpm-runs/`, exposing

```
GET /api/plugins/mpm-runs/runs        # filtered, newest-first (status|session|since|limit), plus server clock
GET /api/plugins/mpm-runs/runs/stats  # status -> count aggregate (+ total)
```

The panel backend is deliberately **read-only and maintenance-free**: it owns a
write-incapable (`query_only=ON`) reader connection, never runs the WAL/schema or
restart-orphan sweep, and never creates `mpm_runs.db` — if the gateway hasn't
created the DB yet, both routes degrade to empty results rather than touching it.
This preserves the cross-process rule that only the gateway mutates run state.

## Layout

```
src/hermes_mpm/
  __init__.py          register(ctx) hub; composes intent + routing into one hook
  plugin.yaml          manifest (entry_point: register)
  profiles.py          loads data/profiles.default.yaml
  routing.py           classify() + make_dispatch_handler() (tiered routing)
  orchestrator.py      hermes_mpm_orchestrate (parallel fan-out)
  intent.py            ported weather/time/disk/svc matchers + handler + commands
  weather_core.py      deterministic Open-Meteo core (vendored)
  cluster_ops_client.py bounded cluster-ops MCP client (vendored)
  runs_db.py           durable SQLite run-tracking + batch telemetry (init/record/sweep/query/purge)
  cli.py               `hermes mpm` (list-profiles + routing dry-run + runs + parallelism)
  dashboard/           "MPM Runs" dashboard panel
    manifest.json      panel manifest (tab, icon, entry/css/api wiring)
    plugin_api.py      read-only FastAPI router (/api/plugins/mpm-runs/)
    dist/              prebuilt frontend bundle (index.js + style.css)
  skills/pm_orchestrator/SKILL.md
  data/profiles.default.yaml
  tests/               test_loads / test_routing / test_orchestrator / test_intent
                       / test_gate / test_runs_db / test_runs_cli / test_runs_hooks
                       / test_turn_batches / test_parallelism_cli
                       / test_dashboard_plugin_api / test_leaf_batch_hint
```

> Note: `system_prompt_file` paths in the shipped profiles point at
> `/opt/hermes/home/profiles/*.md`; making them portable is a later item.

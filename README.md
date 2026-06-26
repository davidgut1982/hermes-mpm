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
- **CLI** — `hermes mpm list-profiles` and `hermes mpm routing "<msg>"`
  (dry-run classifier).

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
  cli.py               `hermes mpm` (list-profiles + routing dry-run)
  skills/pm_orchestrator/SKILL.md
  data/profiles.default.yaml
  tests/               test_loads / test_routing / test_orchestrator / test_intent
```

> Note: `system_prompt_file` paths in the shipped profiles point at
> `/opt/hermes/home/profiles/*.md`; making them portable is a later item.

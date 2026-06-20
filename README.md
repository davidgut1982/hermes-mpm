# hermes-mpm

Multi-agent profile management (MPM) plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

**Status: v0.1 — loadable scaffold.** This release is the wired skeleton plus
the shipped profile data. Routing, orchestration fan-out, and intent shortcuts
are stubs that the next task fills in.

## What's real in v0.1

- **Profiles** — ships the default set of 22 agent archetypes
  (`src/hermes_mpm/data/profiles.default.yaml`), loaded via `profiles.py`.
- **`hermes mpm list-profiles`** — prints the shipped archetypes (no LLM).

## What's stubbed (next task)

- `routing.py` — tier classifier + `pre_gateway_dispatch` decisions.
- `orchestrator.py` — `hermes_mpm_orchestrate` tool (currently validates +
  echoes a plan).
- `intent.py` — `pre_gateway_dispatch` passthrough (returns `None`).
- `hermes mpm routing` — placeholder.

## Install (into a Hermes venv)

```bash
pip install -e .
```

Then enable it in `config.yaml`:

```yaml
plugins:
  enabled:
    - hermes_mpm
```

Confirm: `hermes plugins list` shows `hermes_mpm` enabled, and
`hermes mpm list-profiles` prints the archetypes.

## Layout

```
src/hermes_mpm/
  __init__.py     register(ctx) hub — wires capabilities, each in try/except
  plugin.yaml     manifest (entry_point: register, config_namespace: hermes_mpm)
  profiles.py     loads data/profiles.default.yaml (REAL)
  routing.py      tier classify() + no-op pre_gateway_dispatch (stub)
  orchestrator.py hermes_mpm_orchestrate handler (stub)
  intent.py       pre_gateway_dispatch passthrough (stub)
  cli.py          `hermes mpm` subcommand (list-profiles REAL, routing stub)
  skills/pm_orchestrator/SKILL.md
  data/profiles.default.yaml
  tests/test_loads.py
```

> Note: `system_prompt_file` paths in the shipped profiles point at
> `/opt/hermes/home/profiles/*.md`. They are preserved verbatim for now;
> making them portable is a later item.

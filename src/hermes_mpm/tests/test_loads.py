"""Load-time tests for the hermes-mpm scaffold.

Why: The whole point of v0.1 is "it loads cleanly and the real bits work."
These tests assert register(ctx) wires every capability against a fake ctx and
that profile loading is real — no Hermes core required.
What: A FakeCtx records register_* calls; tests drive register() and the real
profile/CLI/orchestrator paths.
Test: Run `pytest` — all assertions below must pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hermes_mpm
from hermes_mpm import cli, orchestrator, profiles


class FakeCtx:
    """Minimal stand-in for the host PluginContext that records registrations.

    Why: Lets us exercise register() with no Hermes core import.
    What: Each register_* appends to a typed list; _read_config in register()
    falls through to {} because hermes_cli isn't importable here.
    Test: After register(ctx), the recorded lists match expectations below.
    """

    def __init__(self) -> None:
        class _M:
            name = "hermes_mpm"
            key = "hermes_mpm"

        self.manifest = _M()
        self.cli_commands: list[str] = []
        self.commands: list[str] = []
        self.hooks: list[str] = []
        self.tools: list[str] = []
        self.skills: list[str] = []

    def register_cli_command(self, name, help, setup_fn, handler_fn=None, description=""):
        self.cli_commands.append(name)

    def register_command(self, name, handler, description="", **kwargs):
        self.commands.append(name)

    def register_hook(self, hook_name, callback):
        self.hooks.append(hook_name)

    def register_tool(self, name, toolset, schema, handler, **kwargs):
        self.tools.append(name)

    def register_skill(self, name, path: Path, description=""):
        assert path.exists(), f"skill path missing: {path}"
        self.skills.append(name)


def test_register_runs_clean_and_wires_all_capabilities():
    """register(ctx) wires the four v0.1 capabilities without raising."""
    ctx = FakeCtx()
    hermes_mpm.register(ctx)

    assert ctx.cli_commands == ["mpm"]
    # Routing + intent now register on the cross-surface pre_llm_call seam.
    assert "pre_llm_call" in ctx.hooks
    assert "pre_gateway_dispatch" not in ctx.hooks
    assert orchestrator.TOOL_NAME in ctx.tools
    assert "pm_orchestrator" in ctx.skills
    # The four intent slash commands the rewrites resolve to.
    for cmd in ("weather", "time", "diskfree", "svcstatus"):
        assert cmd in ctx.commands


def test_profiles_load_real():
    """The shipped default profiles load and include known archetypes."""
    profs = profiles.load_profiles()
    assert isinstance(profs, dict) and profs
    names = profiles.list_archetypes()
    assert names == sorted(names)
    for expected in ("calendar", "ops", "engineer", "weather"):
        assert expected in names
    # ops archetype carries its real toolset
    assert "mcp-cluster-ops" in profiles.get_profile("ops")["toolsets"]


def test_cli_list_profiles_returns_zero(capsys):
    """`hermes mpm list-profiles` prints archetypes and exits 0."""
    parser = argparse.ArgumentParser()
    cli.setup(parser)
    args = parser.parse_args(["list-profiles"])
    rc = cli.handle(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "ops" in out
    assert "archetype(s)" in out


def test_orchestrate_validates_required_args():
    """The orchestrate tool validates goal + subtasks (full behavior in
    test_orchestrator.py)."""
    missing = json.loads(orchestrator.handle({}))
    assert "error" in missing

    empty = json.loads(orchestrator.handle({"goal": "g", "subtasks": []}))
    assert "error" in empty


def test_read_config_merges_top_level_and_entry(monkeypatch, tmp_path):
    """_read_config must merge top-level hermes_mpm (routing) over the entry
    block (gate), so routing sees its tiers and the gate keeps review_gate."""
    import yaml as _yaml
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(_yaml.safe_dump({
        "hermes_mpm": {
            "tiers": {"strong": {"model": "glm-5.2"}, "main": {"model": "glm-4.7"}},
            "openrouter": {"provider": "zai"},
        },
        "plugins": {"entries": {"hermes_mpm": {
            "review_gate": {"enabled": True},
            "tiers": {"main": {"model": "old"}},
        }}},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    cfg = hermes_mpm._read_config(object())
    # Top-level routing tiers win (strong present, main overwritten).
    assert cfg["tiers"]["strong"]["model"] == "glm-5.2"
    assert cfg["tiers"]["main"]["model"] == "glm-4.7"
    assert cfg["openrouter"]["provider"] == "zai"
    # Entry-scoped gate config preserved.
    assert cfg["review_gate"]["enabled"] is True

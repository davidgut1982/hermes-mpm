"""Default agent-profile (archetype) loading for hermes-mpm.

Why: MPM routes work to named agent archetypes (calendar, ops, engineer, …).
The shipped default set has to come from somewhere portable; this module owns
the bundled ``data/profiles.default.yaml`` so the rest of the plugin (routing,
orchestrator, CLI) reads archetypes through one helper instead of re-parsing
config. This is the one piece of v0.1 that is REAL, not a stub.
What: Loads the ``agent_profiles`` block from the packaged YAML and exposes
helpers to fetch the full mapping and a sorted archetype list.
Test: ``load_profiles()`` returns a dict with the 22 shipped keys; calling
``list_archetypes()`` returns them sorted and includes "ops" and "engineer".
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any, Dict, List

import yaml

# Key under which archetypes live in profiles.default.yaml (mirrors config.yaml).
AGENT_PROFILES_KEY = "agent_profiles"
DEFAULT_PROFILES_RESOURCE = "data/profiles.default.yaml"


@lru_cache(maxsize=1)
def load_profiles() -> Dict[str, Dict[str, Any]]:
    """Load and cache the shipped default agent profiles.

    Why: Single source of truth for the default archetype set; cached so
    repeated CLI/routing lookups don't re-read and re-parse the file.
    What: Reads the packaged ``data/profiles.default.yaml`` and returns the
    ``agent_profiles`` mapping (name -> profile dict).
    Test: Assert the returned dict is non-empty and contains "calendar".
    """
    text = (
        resources.files("hermes_mpm")
        .joinpath(DEFAULT_PROFILES_RESOURCE)
        .read_text(encoding="utf-8")
    )
    data = yaml.safe_load(text) or {}
    profiles = data.get(AGENT_PROFILES_KEY) or {}
    if not isinstance(profiles, dict):
        raise ValueError(
            f"{DEFAULT_PROFILES_RESOURCE}: '{AGENT_PROFILES_KEY}' must be a mapping, "
            f"got {type(profiles).__name__}"
        )
    return profiles


def list_archetypes() -> List[str]:
    """Return the sorted list of available archetype names.

    Why: The CLI and routing need a stable, ordered view of what archetypes
    exist without each caller knowing the YAML shape.
    What: Returns ``sorted(load_profiles().keys())``.
    Test: Assert the result is sorted and "weather" appears in it.
    """
    return sorted(load_profiles().keys())


def get_profile(name: str) -> Dict[str, Any]:
    """Return a single archetype's profile dict.

    Why: Routing/orchestrator need a specific archetype's model + toolsets.
    What: Looks up *name* in the loaded profiles; raises KeyError if absent.
    Test: ``get_profile("ops")["toolsets"]`` contains "mcp-cluster-ops".
    """
    profiles = load_profiles()
    if name not in profiles:
        raise KeyError(f"Unknown archetype '{name}' (have: {', '.join(list_archetypes())})")
    return profiles[name]

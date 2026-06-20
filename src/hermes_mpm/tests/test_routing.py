"""Decision-matrix tests for the hermes-mpm tier classifier and dispatch handler.

Why: Routing is the core feature and money-saving lever — a wrong tier either
overspends or under-serves. The classifier is a pure function, so its decision
matrix is exhaustively unit-testable with zero LLM/gateway. The dispatch handler
is tested against a fake gateway to prove it writes the COMPLETE override and
evicts the cached agent (the mechanism that supersedes a profile's pinned model).
What: Asserts the precedence rules and the override-write side-effect.
Test: this file — run pytest.
"""

from __future__ import annotations

import pytest

from hermes_mpm import routing
from hermes_mpm.routing import (
    TIER_CHEAP_WORKHORSE,
    TIER_FREE_BACKGROUND,
    TIER_MAIN,
    TIER_MAX,
    TIER_STRONG,
)


@pytest.mark.parametrize(
    "text, platform, profile, expected_tier",
    [
        # interactive_simple: short status question on telegram → cheap workhorse
        ("is plex up?", "telegram", None, TIER_CHEAP_WORKHORSE),
        ("show me the list", "telegram", None, TIER_CHEAP_WORKHORSE),
        # engineering: complexity keyword up-routes regardless of platform
        ("implement retry backoff in the scraper", "telegram", None, TIER_STRONG),
        ("debug the failing gateway", "telegram", None, TIER_STRONG),
        # background: cron platform → free background
        ("summarize today's receipts", "cron", None, TIER_FREE_BACKGROUND),
        # bulk classification: receipts profile → free background
        ("classify this", "telegram", "receipts", TIER_FREE_BACKGROUND),
        # explicit /tier max wins over everything
        ("/tier max do the hardest thing", "telegram", None, TIER_MAX),
        # long-but-simple (no complexity kw, no simple verb) → default main
        ("a " * 200 + "tell me about this topic in detail", "telegram", None, TIER_MAIN),
        # engineering profile → strong (profile_tier_map)
        ("look at this", "telegram", "engineer", TIER_STRONG),
        # default interactive: ordinary chat → main
        ("how are you doing today", "telegram", None, TIER_MAIN),
    ],
)
def test_classify_matrix(text, platform, profile, expected_tier):
    grouping, tier = routing.classify(text, platform=platform, profile=profile)
    assert tier == expected_tier, f"{text!r} on {platform} -> {tier} (want {expected_tier})"
    assert isinstance(grouping, str) and grouping


def test_complexity_keyword_uproutes_short_message():
    # short AND contains a complexity keyword → complexity wins over simplicity
    _, tier = routing.classify("optimize it", platform="telegram")
    assert tier == TIER_STRONG


def test_low_urgency_routes_background():
    _, tier = routing.classify("anything", platform="telegram", urgency="low")
    assert tier == TIER_FREE_BACKGROUND


def test_cli_platform_opted_out():
    # CLI/local is opt-out by default; the dispatch gate must say so.
    assert routing._platform_enabled({}, "cli") is False
    assert routing._platform_enabled({}, "local") is False
    assert routing._platform_enabled({}, "telegram") is True


# ── Dispatch handler: override write + eviction ─────────────────────────────


class _FakeSource:
    platform = "telegram"
    profile = None


class _FakeEvent:
    def __init__(self, text):
        self.text = text
        self.source = _FakeSource()
        self.platform = "telegram"
        self.urgency = None


class _FakeGateway:
    def __init__(self):
        self._session_model_overrides = {}
        self.evicted = []

    def _session_key_for_source(self, source):
        return "agent:main:telegram:dm"

    def _evict_cached_agent(self, key):
        self.evicted.append(key)


def test_dispatch_writes_complete_override_and_evicts(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    handler = routing.make_dispatch_handler({})
    gw = _FakeGateway()

    ret = handler(event=_FakeEvent("implement a feature"), gateway=gw)
    assert ret is None  # routing never rewrites text

    key = "agent:main:telegram:dm"
    override = gw._session_model_overrides[key]
    # Complete bundle so it supersedes any profile-pinned model.
    assert override["model"] == routing.DEFAULT_TIERS[TIER_STRONG]["model"]
    assert override["api_key"] == "sk-test-123"
    assert override["base_url"] == routing.OPENROUTER_BASE_URL
    assert override["provider"] == "openrouter"
    assert override["_hermes_mpm"] is True
    assert gw.evicted == [key]


def test_dispatch_respects_manual_model_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    handler = routing.make_dispatch_handler({})
    gw = _FakeGateway()
    key = "agent:main:telegram:dm"
    # A manual /model override has no _hermes_mpm marker — must not be stomped.
    gw._session_model_overrides[key] = {"model": "manual/model", "api_key": "x"}

    handler(event=_FakeEvent("implement a feature"), gateway=gw)
    assert gw._session_model_overrides[key]["model"] == "manual/model"
    assert gw.evicted == []  # nothing changed


def test_dispatch_opts_out_on_cli(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    handler = routing.make_dispatch_handler({})
    gw = _FakeGateway()

    class _CliSource:
        platform = "local"
        profile = None

    class _CliEvent:
        text = "implement a feature"
        source = _CliSource()
        platform = "local"
        urgency = None

    handler(event=_CliEvent(), gateway=gw)
    assert gw._session_model_overrides == {}
    assert gw.evicted == []


def test_dispatch_no_apikey_writes_nothing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    handler = routing.make_dispatch_handler({})
    gw = _FakeGateway()
    handler(event=_FakeEvent("implement a feature"), gateway=gw)
    # No creds → no half-override that could lose to a profile.
    assert gw._session_model_overrides == {}

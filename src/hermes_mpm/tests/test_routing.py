"""Decision-matrix tests for the hermes-mpm tier classifier and pre_llm_call handler.

Why: Routing is the core feature and money-saving lever — a wrong tier either
overspends or under-serves. The classifier is a pure function, so its decision
matrix is exhaustively unit-testable with zero LLM/gateway. The dispatch handler
now runs on the cross-surface ``pre_llm_call`` seam: it reads ``user_message``/
``platform``/``model`` from kwargs and RETURNS a complete model bundle dict
(``{model, provider, api_key, base_url, api_mode}``) that the engine applies via
switch_model — no gateway-internal coupling, so it works identically on
gateway/TUI/dashboard.
What: Asserts the precedence rules and the returned bundle.
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


def test_tui_and_dashboard_route_like_telegram():
    """Per the cross-surface unification: tui/dashboard ROUTE (not opted out)."""
    assert routing._platform_enabled({}, "tui") is True
    assert routing._platform_enabled({}, "dashboard") is True
    assert routing._platform_enabled({}, "telegram") is True
    # Bare "cli" / "local" still default off (honored manual /model there).
    assert routing._platform_enabled({}, "cli") is False
    assert routing._platform_enabled({}, "local") is False
    # Explicit enabled flag overrides the default.
    assert routing._platform_enabled({"platforms": {"cli": {"enabled": True}}}, "cli") is True


# ── pre_llm_call handler: returns a complete model bundle ────────────────────


def test_handler_returns_complete_bundle(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    handler = routing.make_pre_llm_call_handler({})

    bundle = handler(
        user_message="implement a feature",
        platform="telegram",
        model="deepseek/deepseek-v4-flash",  # a known tier model (main)
    )
    assert bundle is not None
    assert bundle["model"] == routing.DEFAULT_TIERS[TIER_STRONG]["model"]
    assert bundle["api_key"] == "sk-test-123"
    assert bundle["base_url"] == routing.OPENROUTER_BASE_URL
    assert bundle["provider"] == "openrouter"
    # api_mode key present (None = OpenAI-compatible).
    assert "api_mode" in bundle


def test_handler_respects_manual_model_pin(monkeypatch):
    """A foreign (operator-pinned) model must NOT be overridden."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    handler = routing.make_pre_llm_call_handler({})

    out = handler(
        user_message="implement a feature",
        platform="telegram",
        model="anthropic/claude-opus-4.6",  # not any tier model → manual pin
    )
    assert out is None  # deferred to the operator's pin


def test_handler_user_pin_flag_beats_tier_model(monkeypatch):
    """Regression (HIGH-2): a ``/model`` pin must win even when the pinned model
    IS one of the routing tiers (e.g. ``/model glm-5.2`` == strong tier).

    The name heuristic (``model not in known_models``) fails here because the
    pin equals a tier model. The durable ``agent._user_model_pin`` flag must
    still force a defer. To isolate the flag from the idempotent same-model
    short-circuit, the pinned model (strong tier = glm-5.2) is paired with a
    message that routes to a DIFFERENT tier (main): without the flag routing
    overrides to main; with the flag it defers.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    # glm-5.2 IS the strong tier model → it lives in known_models.
    cfg = {"tiers": {"strong": {"model": "glm-5.2"}}}
    handler = routing.make_pre_llm_call_handler(cfg)

    class _Agent:
        _user_model_pin = False

    agent = _Agent()
    # "how are you doing today" → main tier (different from strong=glm-5.2).
    main_model = routing.DEFAULT_TIERS[TIER_MAIN]["model"]

    # No pin: routing is active and overrides the live glm-5.2 → main tier.
    agent._user_model_pin = False
    out = handler(
        user_message="how are you doing today",
        platform="telegram",
        model="glm-5.2",
        agent=agent,
    )
    assert out is not None, "without the pin, routing must tier-route normally"
    assert out["model"] == main_model

    # Pinned: even though glm-5.2 IS a tier model, the flag forces a defer.
    agent._user_model_pin = True
    out = handler(
        user_message="how are you doing today",
        platform="telegram",
        model="glm-5.2",
        agent=agent,
    )
    assert out is None, "the user-pin flag must beat tier routing"


def test_handler_opts_out_on_cli(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    handler = routing.make_pre_llm_call_handler({})
    out = handler(
        user_message="implement a feature",
        platform="cli",
        model="deepseek/deepseek-v4-flash",
    )
    assert out is None


def test_handler_no_apikey_returns_none(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    handler = routing.make_pre_llm_call_handler({})
    out = handler(
        user_message="implement a feature",
        platform="telegram",
        model="deepseek/deepseek-v4-flash",
    )
    # No creds → no half-bundle the engine couldn't apply.
    assert out is None


def test_handler_empty_message_returns_none(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    handler = routing.make_pre_llm_call_handler({})
    assert handler(user_message="", platform="telegram", model="x") is None


def test_resolve_tier_per_tier_provider_override(monkeypatch):
    """A tier may override the shared provider block (free_background → OpenRouter
    while the shared block targets z.ai)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
    config = {
        "openrouter": {
            "provider": "zai",
            "base_url": "https://api.z.ai/api/coding/paas/v4",
            "api_key": "sk-zai",
        },
        "tiers": {
            "strong": {"model": "glm-5.2"},
            "free_background": {
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
            },
        },
    }
    strong = routing.resolve_tier_override("strong", config)
    assert strong["provider"] == "zai"
    assert strong["base_url"] == "https://api.z.ai/api/coding/paas/v4"
    assert strong["api_key"] == "sk-zai"

    free = routing.resolve_tier_override("free_background", config)
    assert free["provider"] == "openrouter"
    assert free["base_url"] == "https://openrouter.ai/api/v1"
    assert free["api_key"] == "sk-or-1"  # from OPENROUTER_API_KEY env


def test_handler_errors_are_swallowed(monkeypatch):
    """A misbehaving classify must never raise out of the handler."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    handler = routing.make_pre_llm_call_handler({})

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(routing, "classify", _boom)
    assert handler(user_message="implement", platform="telegram", model="x") is None

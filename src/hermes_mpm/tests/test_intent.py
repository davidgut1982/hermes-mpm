"""Intent fast-path matcher tests for hermes-mpm.

Why: The intent matcher decides which messages skip the LLM entirely. A false
positive hijacks a real request; a false negative wastes a model call. These
tests pin the four ported matchers (weather/time/disk/svc) and the strict
fall-through on unrelated prose.
What: Drives match_intent (pure, no event/gateway) and answer_time (stdlib).
Test: this file — run pytest.
"""

from __future__ import annotations

from hermes_mpm import intent


def test_weather_rewrites_with_location():
    out = intent.match_intent("what's the weather in Chicago")
    assert out == {"action": "rewrite", "text": "/weather Chicago"}


def test_weather_rewrites_without_location():
    out = intent.match_intent("is it going to rain")
    assert out == {"action": "rewrite", "text": "/weather"}


def test_time_rewrites():
    out = intent.match_intent("what time is it")
    assert out is not None
    assert out["action"] == "rewrite"
    assert out["text"].startswith("/time")


def test_date_rewrites_to_time_command():
    out = intent.match_intent("what's today's date")
    assert out is not None and out["text"].startswith("/time")


def test_disk_rewrites():
    out = intent.match_intent("disk usage")
    assert out == {"action": "rewrite", "text": "/diskfree"}


def test_disk_question_rewrites():
    out = intent.match_intent("how much disk is free")
    assert out == {"action": "rewrite", "text": "/diskfree"}


def test_service_rewrites_known_unit():
    out = intent.match_intent("is the gateway up")
    assert out is not None
    assert out["action"] == "rewrite"
    assert out["text"].startswith("/svcstatus")


def test_service_unknown_unit_falls_through():
    # Unknown unit must NOT short-circuit — defers to the agent.
    assert intent.match_intent("is foobar running") is None


def test_unrelated_text_returns_none():
    for msg in (
        "tell me a story about time",
        "schedule a meeting for tomorrow",
        "what should I cook for dinner",
        "is my code any good",
        "summarize this document for me",
    ):
        assert intent.match_intent(msg) is None, msg


def test_slash_command_passthrough_returns_none():
    assert intent.match_intent("/weather Chicago") is None
    assert intent.match_intent("/anything") is None


def test_empty_returns_none():
    assert intent.match_intent("") is None
    assert intent.match_intent("   ") is None


def test_answer_time_is_self_contained():
    out = intent.answer_time("what time is it")
    assert ":" in out  # clock time present
    out_day = intent.answer_time("what day is it")
    # weekday name present
    assert any(
        d in out_day
        for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    )


def test_pre_gateway_dispatch_matches_event():
    class _E:
        text = "what's the weather in Boston"

    assert intent.pre_gateway_dispatch(event=_E()) == {
        "action": "rewrite",
        "text": "/weather Boston",
    }

    class _E2:
        text = "let's chat about something"

    assert intent.pre_gateway_dispatch(event=_E2()) is None

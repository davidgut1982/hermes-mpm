"""Intent fast-path matcher + pre_llm_call short-circuit tests for hermes-mpm.

Why: The intent matcher decides which messages skip the LLM entirely. A false
positive hijacks a real request; a false negative wastes a model call. These
tests pin the four ported matchers (weather/time/disk/svc), the strict
fall-through on unrelated prose, and the new ``pre_llm_call`` handler that
EXECUTES the matched command in-process and returns ``{"final_response": …}``
(api_calls == 0 on the engine side).
What: Drives match_intent (pure) and the pre_llm_call handler (executes cores).
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


# ── pre_llm_call short-circuit: executes the command in-process ──────────────


def test_pre_llm_call_time_returns_final_response():
    """A time question executes time_command in-process → {final_response}."""
    out = intent.pre_llm_call(user_message="what time is it")
    assert isinstance(out, dict)
    assert "final_response" in out
    assert ":" in out["final_response"]  # rendered clock time


def test_pre_llm_call_weather_executes_in_process(monkeypatch):
    """Weather executes weather_command (stub the core so no network)."""
    from hermes_mpm import weather_core

    monkeypatch.setattr(
        weather_core, "answer_weather",
        lambda **kw: "Chicago: 72F, clear.",
    )
    out = intent.pre_llm_call(user_message="what's the weather in Chicago")
    assert out == {"final_response": "Chicago: 72F, clear."}


def test_pre_llm_call_unrelated_returns_none():
    """Non-intent prose must NOT short-circuit (proceeds to the LLM)."""
    assert intent.pre_llm_call(user_message="summarize this document for me") is None
    assert intent.pre_llm_call(user_message="schedule a meeting for tomorrow") is None


def test_pre_llm_call_command_none_falls_through(monkeypatch):
    """If the matched command returns None (e.g. cluster-ops down), defer."""
    # diskfree depends on cluster-ops; force it to return None.
    monkeypatch.setattr(intent, "diskfree_command", lambda raw: None)
    out = intent.pre_llm_call(user_message="disk usage")
    assert out is None


def test_pre_llm_call_slash_passthrough_returns_none():
    assert intent.pre_llm_call(user_message="/weather Chicago") is None

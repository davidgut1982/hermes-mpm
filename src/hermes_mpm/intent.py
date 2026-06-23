"""Deterministic intent fast-paths for hermes-mpm.

Why: Some requests are fully answerable without an LLM — weather, time/date,
disk-free, service-status. Routing them to a stdlib/HTTP handler instead of an
agent loop is the biggest latency+cost win. This module ports the proven
weather-deterministic and intent-handlers-core matchers into one place: a
``pre_llm_call`` handler that matches a plain-text message, EXECUTES the
corresponding command core in-process (``/weather``, ``/time``, ``/diskfree``,
``/svcstatus``), and returns the engine's short-circuit bundle
(``{"final_response"}`` → api_calls == 0). It runs on every surface
(gateway/TUI/dashboard) identically.

What: ``match_intent(text)`` is a pure matcher returning a rewrite dict or None.
``pre_llm_call(**kw)`` reads ``user_message``, matches, runs the command, and
returns ``{"final_response": <answer>}`` or None. The four slash commands call
deterministic cores: weather (Open-Meteo, stdlib HTTP) and time (stdlib clock)
always answer; disk and service call cluster-ops only when ``CLUSTER_OPS_URL``/
``CLUSTER_OPS_TOKEN`` are set, else return None and the turn proceeds to the LLM.

Test: match_intent("what's the weather in Chicago") -> rewrite to "/weather
Chicago"; "what time is it" -> "/time …"; "disk usage" -> "/diskfree"; "is the
gateway up" -> "/svcstatus …"; unrelated prose -> None. See test_intent.py.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger("hermes_mpm.intent")

try:  # py3.9+: always present in this build
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

TIMEZONE = "America/Chicago"
_HOST = "hermes"

# Intent slash-command -> command handler name (resolved via module globals at
# call time so the pre_llm_call short-circuit and the registered commands share
# one source of truth, and tests can monkeypatch a handler).
_COMMAND_DISPATCH = {
    "weather": "weather_command",
    "time": "time_command",
    "diskfree": "diskfree_command",
    "svcstatus": "svcstatus_command",
}


# ── Weather matcher (ported from weather-deterministic) ─────────────────────

_WEATHER_INTENT_RE = re.compile(
    r"\b("
    r"weather|forecast|temperature|temp|how\s+(?:hot|cold|warm)|"
    r"is\s+it\s+(?:going\s+to\s+)?(?:rain|snow|sunny|cloudy)|"
    r"will\s+it\s+(?:rain|snow)|"
    r"rain(?:ing|fall)?|snow(?:ing|fall)?|humidity|wind\s+speed"
    r")\b",
    re.IGNORECASE,
)
_LOCATION_RE = re.compile(
    r"\b(?:in|for|at|near|around)\s+([A-Za-z][A-Za-z0-9 .,'\-]{1,60})",
    re.IGNORECASE,
)
_TRAILING_NOISE_RE = re.compile(
    r"\s*\b(today|tomorrow|now|please|right\s+now|currently|this\s+week|"
    r"this\s+weekend|tonight|outside)\b.*$",
    re.IGNORECASE,
)


def is_weather_intent(text: str) -> bool:
    """True if *text* looks like a weather request (ported matcher)."""
    return bool(text and text.strip() and _WEATHER_INTENT_RE.search(text))


def extract_location(text: str) -> Optional[str]:
    """Pull a candidate location out of *text*, or None (defaults handled later)."""
    if not text:
        return None
    m = _LOCATION_RE.search(text)
    if not m:
        return None
    candidate = _TRAILING_NOISE_RE.sub("", m.group(1).strip()).strip(" .,'-")
    if not candidate or _WEATHER_INTENT_RE.fullmatch(candidate) or len(candidate) < 2:
        return None
    return candidate


# ── Time/date matcher (ported from time_core) ───────────────────────────────

_TIME_RE = re.compile(
    r"^\s*(?:"
    r"what(?:'|’)?s?(?:\s+is)?\s+(?:the\s+|current\s+)?time(?:\s+is\s+it)?"
    r"|what\s+time\s+is\s+it|current\s+time|the\s+time)"
    r"(?:\s+(?:now|right\s+now|today|currently|here|please))?\s*\??\s*$",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"^\s*(?:"
    r"what(?:'|’)?s?(?:\s+is)?\s+(?:the\s+|today(?:'|’)?s?\s+|current\s+)?date(?:\s+is\s+it)?"
    r"|what\s+date\s+is\s+it|what(?:'|’)?s?\s+today(?:'|’)?s?\s+date"
    r"|current\s+date|today(?:'|’)?s?\s+date|the\s+date)"
    r"(?:\s+(?:now|right\s+now|today|currently|please))?\s*\??\s*$",
    re.IGNORECASE,
)
_DAY_RE = re.compile(
    r"^\s*(?:"
    r"what\s+day\s+is\s+it(?:\s+today)?"
    r"|what(?:'|’)?s?(?:\s+is)?\s+(?:the\s+)?day(?:\s+today)?"
    r"|what\s+day\s+of\s+the\s+week\s+is\s+it)"
    r"(?:\s+(?:now|right\s+now|today|currently|please))?\s*\??\s*$",
    re.IGNORECASE,
)


def is_time_intent(text: str) -> bool:
    """True iff *text* is a time/date/day question (ported matcher)."""
    if not text or not text.strip():
        return False
    s = text.strip()
    return bool(_TIME_RE.match(s) or _DATE_RE.match(s) or _DAY_RE.match(s))


def _time_kind(text: str) -> str:
    s = (text or "").strip()
    if _DAY_RE.match(s):
        return "day"
    if _DATE_RE.match(s):
        return "date"
    return "time"


def answer_time(text: Optional[str] = None) -> str:
    """Render the current local time/date as a terse reply (ported, no LLM)."""
    now = datetime.now(ZoneInfo(TIMEZONE)) if ZoneInfo else datetime.now()
    weekday = now.strftime("%A")
    iso_date = now.strftime("%Y-%m-%d")
    pretty_date = now.strftime("%B %-d, %Y")
    clock = now.strftime("%-I:%M %p").lstrip("0")
    tzname = now.strftime("%Z") or TIMEZONE
    kind = _time_kind(text or "")
    if kind == "day":
        return f"Today is {weekday} ({iso_date})."
    if kind == "date":
        return f"Today is {weekday}, {pretty_date} ({iso_date})."
    return f"It is {clock} {tzname} on {weekday}, {pretty_date}."


# ── Disk matcher (ported from disk_core) ────────────────────────────────────

_DISK_RE = re.compile(
    r"^\s*(?:"
    r"how\s+much\s+(?:disk|free\s+disk)\s*(?:space|storage)?\s+(?:is\s+)?(?:free|left|available|used)"
    r"|how\s+much\s+(?:disk\s+)?space\s+(?:is\s+)?(?:free|left|available)\s+on\s+disk"
    r"|(?:free\s+)?disk\s*(?:space|usage|free|usage\s+report)?(?:\s+(?:left|free|available|used|report))?"
    r"|disk\s+free"
    r"|(?:what(?:'|’)?s?\s+(?:the\s+)?|show\s+(?:me\s+)?(?:the\s+)?|check\s+(?:the\s+)?)disk\s*(?:space|usage|free))"
    r"(?:\s+(?:on\s+hermes|here|now|please|today))?\s*\??\s*$",
    re.IGNORECASE,
)


def is_disk_intent(text: str) -> bool:
    """True iff *text* is a disk-space question about this host (ported matcher)."""
    return bool(text and text.strip() and _DISK_RE.match(text.strip()))


# ── Service matcher (ported from service_core) ──────────────────────────────

_UNIT_ALIASES = {
    "hermes-gateway": "hermes-gateway",
    "hermesgateway": "hermes-gateway",
    "hermes gateway": "hermes-gateway",
    "gateway": "hermes-gateway",
    "hermes": "hermes-gateway",
    "the gateway": "hermes-gateway",
    "hermes-agent": "hermes-gateway",
}
_NON_SERVICE_WORDS = {
    "it",
    "this",
    "that",
    "everything",
    "anything",
    "something",
    "raining",
    "snowing",
    "sunny",
    "cold",
    "hot",
    "warm",
    "cool",
    "ok",
    "okay",
    "good",
    "fine",
    "alright",
    "ready",
    "done",
    "there",
    "he",
    "she",
    "they",
    "the",
    "a",
    "an",
    "my",
    "your",
}
_RUNNING_RE = re.compile(
    r"^\s*is\s+(?:the\s+)?(?P<svc>[\w][\w .\-]{0,40}?)"
    r"(?:\s+(?:service|daemon|unit))?\s+"
    r"(?:running|up|active|alive|online|down|stopped|dead|ok)\s*\??\s*$",
    re.IGNORECASE,
)
_STATUS_OF_RE = re.compile(
    r"^\s*(?:what(?:'|’)?s?\s+the\s+|check\s+the\s+|show\s+(?:me\s+)?the\s+)?"
    r"status\s+of\s+(?:the\s+)?(?P<svc>[\w][\w .\-]{0,40}?)"
    r"(?:\s+(?:service|daemon|unit))?\s*\??\s*$",
    re.IGNORECASE,
)
_IS_X_RUNNING_RE = re.compile(
    r"^\s*is\s+(?:the\s+)?(?P<svc>[\w][\w .\-]{0,40}?)\s+(?:service|daemon|unit)\s+"
    r"(?:running|up|active|alive|online|down|stopped|dead|ok)?\s*\??\s*$",
    re.IGNORECASE,
)


def _svc_capture(text: str) -> Optional[str]:
    s = (text or "").strip()
    for rx in (_RUNNING_RE, _IS_X_RUNNING_RE, _STATUS_OF_RE):
        m = rx.match(s)
        if m:
            cand = (m.group("svc") or "").strip()
            if cand:
                return cand
    return None


def _svc_normalize(candidate: str) -> str:
    cand = candidate.strip().lower()
    cand = re.sub(r"\b(service|daemon|unit)\b", " ", cand)
    return re.sub(r"\s+", " ", cand).strip()


def parse_unit(text: str) -> Optional[str]:
    """Resolve a service question to a canonical allow-listed unit, or None."""
    cand = _svc_capture(text)
    if not cand:
        return None
    norm = _svc_normalize(cand)
    if not norm or norm in _NON_SERVICE_WORDS:
        return None
    for key in (norm, norm.replace(" ", "-"), norm.replace(" ", "")):
        if key in _UNIT_ALIASES:
            return _UNIT_ALIASES[key]
    return None


def is_service_intent(text: str) -> bool:
    """True iff *text* is a service-status question for a KNOWN unit (ported)."""
    return bool(text and text.strip() and parse_unit(text) is not None)


# ── Intent matcher + pre_llm_call short-circuit handler ─────────────────────


def match_intent(text: str) -> Optional[dict]:
    """Match deterministic intents and return a rewrite decision, or None.

    Why: One pure function (no event/gateway) so the matching is unit-testable
    and the dispatch handler stays a thin shell.
    What: For a non-slash message, tries weather, time, disk, then service (in a
    stable, mutually-exclusive priority) and returns the corresponding
    ``{"action": "rewrite", "text": "/<cmd> …"}``; returns None on no match.
    Test: weather/time/disk/svc phrasings -> their rewrite dict; unrelated text
    -> None; a leading "/cmd" -> None.
    """
    if not text:
        return None
    stripped = text.strip()
    if not stripped or stripped.startswith("/"):
        return None

    try:
        if is_weather_intent(stripped):
            location = extract_location(stripped) or ""
            return {"action": "rewrite", "text": f"/weather {location}".strip()}
        if is_time_intent(stripped):
            return {"action": "rewrite", "text": f"/time {stripped}".strip()}
        if is_disk_intent(stripped):
            return {"action": "rewrite", "text": "/diskfree"}
        if is_service_intent(stripped):
            return {"action": "rewrite", "text": f"/svcstatus {stripped}".strip()}
    except Exception as exc:  # never break dispatch
        logger.debug("hermes-mpm intent match error (deferring): %s", exc)
        return None
    return None


def pre_llm_call(**kw):
    """Execute a deterministic intent in-process and short-circuit the turn.

    Why: Weather/time/disk/svc are fully answerable with no LLM. Running on the
    cross-surface ``pre_llm_call`` seam, this handler answers them identically on
    gateway/TUI/dashboard and returns the engine's short-circuit bundle
    (``{"final_response"}`` → api_calls == 0).
    What: Reads ``user_message`` from kwargs, matches an intent, then EXECUTES the
    corresponding command core in-process (weather/time/diskfree/svcstatus). On a
    non-None answer returns ``{"final_response": <answer>}``; if the command
    returns None (e.g. cluster-ops unavailable) or nothing matched, returns None
    so the turn proceeds to the LLM. Any error defers (returns None).
    Test: "what time is it" -> {"final_response": "It is …"}; weather (stubbed
    core) -> {"final_response": …}; unrelated prose -> None; a command returning
    None -> None.
    """
    try:
        text = kw.get("user_message") or ""
        if not isinstance(text, str):
            return None
        decision = match_intent(text)
        if not decision:
            return None
        rewrite = decision.get("text") or ""
        parts = rewrite.split(None, 1)
        cmd = parts[0].lstrip("/") if parts else ""
        raw_args = parts[1] if len(parts) > 1 else ""
        # Resolve the command via module globals so test monkeypatches of e.g.
        # ``intent.diskfree_command`` are honored, and the dispatch stays in sync
        # with the actual command handlers (single source of truth).
        handler_name = _COMMAND_DISPATCH.get(cmd)
        handler = globals().get(handler_name) if handler_name else None
        if handler is None:
            return None
        answer = handler(raw_args)
        if answer is None:
            return None
        return {"final_response": answer}
    except Exception as exc:  # never break the turn
        logger.debug("hermes-mpm intent pre_llm_call error (deferring): %s", exc)
        return None


# ── Slash-command handlers (sync: fn(raw_args) -> str | None) ───────────────


def weather_command(raw_args: str):
    """/weather [<location>] — deterministic Open-Meteo answer (no LLM)."""
    from . import weather_core

    explicit = (raw_args or "").strip()
    return weather_core.answer_weather(text=explicit, explicit_location=explicit)


def time_command(raw_args: str):
    """/time — current local time/date (America/Chicago). Always answers."""
    return answer_time(text=raw_args or "what time is it")


def diskfree_command(raw_args: str):
    """/diskfree — disk free/used for host hermes. None -> fall back to agent."""
    from . import cluster_ops_client

    data = cluster_ops_client.call_tool("disk_usage", {"host": _HOST})
    return _format_disk(data)


def svcstatus_command(raw_args: str):
    """/svcstatus <unit> — systemd status on hermes. None -> fall back."""
    from . import cluster_ops_client

    args = (raw_args or "").strip()
    if not args:
        return None
    probe = args if is_service_intent(args) else f"is the {args} service running"
    unit = parse_unit(probe)
    if not unit:
        return None
    data = cluster_ops_client.call_tool("service_status", {"host": _HOST, "unit": unit})
    return _format_service(unit, data)


# ── cluster-ops response formatting (kept here; cores are stdlib only) ───────


def _fmt_gb(value):
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def _format_disk(data) -> Optional[str]:
    """Render a cluster-ops disk_usage payload, or None on any doubt."""
    if not isinstance(data, dict):
        return None
    filesystems = data.get("filesystems")
    if not isinstance(filesystems, list) or not filesystems:
        return None
    lines = [f"*Disk usage — {data.get('host', _HOST)}*"]
    rendered = 0
    for fs in filesystems:
        if not isinstance(fs, dict):
            continue
        mount = fs.get("mount") or "?"
        total = _fmt_gb(fs.get("total_gb"))
        used = _fmt_gb(fs.get("used_gb"))
        if total is None or used is None:
            continue
        free = round(total - used, 1)
        pct = fs.get("use_pct")
        try:
            pct_str = f" ({round(float(pct))}% used)" if pct is not None else ""
        except (TypeError, ValueError):
            pct_str = ""
        lines.append(f"{mount}: {free} GB free of {total} GB{pct_str}")
        rendered += 1
    return "\n".join(lines) if rendered else None


def _humanize_state(active: str, sub: str) -> str:
    active = (active or "").lower()
    sub = (sub or "").lower()
    if active == "active" and sub == "running":
        return "running"
    if active == "active":
        return f"active ({sub})" if sub else "active"
    if active == "inactive":
        return "stopped"
    if active == "failed":
        return "FAILED"
    return f"{active or 'unknown'}" + (f" ({sub})" if sub else "")


def _format_service(unit: str, data) -> Optional[str]:
    """Render a cluster-ops service_status payload, or None on any doubt."""
    if not isinstance(data, dict):
        return None
    active = data.get("active")
    sub = data.get("sub_state")
    if active is None and sub is None and data.get("loaded") is None:
        return None
    state = _humanize_state(active or "", sub or "")
    lines = [f"*{unit}* on hermes: {state}"]
    since = data.get("since")
    if isinstance(since, str) and since.strip():
        lines.append(f"since {since.strip()}")
    pid = data.get("pid")
    if isinstance(pid, int) and pid > 0:
        lines.append(f"pid {pid}")
    restarts = data.get("restarts")
    if isinstance(restarts, int):
        lines.append(f"{restarts} restart(s)")
    return lines[0] if len(lines) == 1 else lines[0] + " — " + ", ".join(lines[1:])

"""Deterministic weather core — zero LLM, stdlib only.

Resolves a location (default Woodstock, IL 60098), calls the free Open-Meteo
geocoding + forecast APIs, and formats a human-readable current-conditions +
3-day outlook string. No API key, no MCP, no model calls.

All network calls are bounded by an explicit timeout and degrade to a clear
fallback message on any error (the LLM is never consulted as a fallback).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Optional, Tuple

# --- Defaults --------------------------------------------------------------

DEFAULT_LOCATION_NAME = "Woodstock, IL 60098"
DEFAULT_LAT = 42.3147
DEFAULT_LON = -88.4487
TIMEZONE = "America/Chicago"

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HTTP_TIMEOUT = 8.0  # seconds, bounded per request


# --- Intent detection ------------------------------------------------------

# Reasonably tight weather-intent matcher. Requires a weather noun/verb so we
# don't eat unrelated chatter ("whether or not", "I forecast revenue...").
_WEATHER_INTENT_RE = re.compile(
    r"\b("
    r"weather|forecast|temperature|temp|how\s+(?:hot|cold|warm)|"
    r"is\s+it\s+(?:going\s+to\s+)?(?:rain|snow|sunny|cloudy)|"
    r"will\s+it\s+(?:rain|snow)|"
    r"rain(?:ing|fall)?|snow(?:ing|fall)?|humidity|wind\s+speed"
    r")\b",
    re.IGNORECASE,
)

# Extract "in <place>" / "for <place>" / "at <place>" location phrases.
_LOCATION_RE = re.compile(
    r"\b(?:in|for|at|near|around)\s+([A-Za-z][A-Za-z0-9 .,'\-]{1,60})",
    re.IGNORECASE,
)

# Trailing words that are not part of a place name.
_TRAILING_NOISE_RE = re.compile(
    r"\s*\b(today|tomorrow|now|please|right\s+now|currently|this\s+week|"
    r"this\s+weekend|tonight|outside)\b.*$",
    re.IGNORECASE,
)


def is_weather_intent(text: str) -> bool:
    """True if *text* looks like a weather request."""
    if not text or not text.strip():
        return False
    return bool(_WEATHER_INTENT_RE.search(text))


def extract_location(text: str) -> Optional[str]:
    """Pull a candidate location string out of *text*, or None.

    Returns None when no explicit location is named (caller defaults to
    Woodstock, IL).
    """
    if not text:
        return None
    m = _LOCATION_RE.search(text)
    if not m:
        return None
    candidate = m.group(1).strip()
    # Strip trailing temporal/filler words that geocoding would choke on.
    candidate = _TRAILING_NOISE_RE.sub("", candidate).strip(" .,'-")
    # Reject if the matched phrase is itself a weather word (e.g. "in rain").
    if not candidate or _WEATHER_INTENT_RE.fullmatch(candidate):
        return None
    if len(candidate) < 2:
        return None
    return candidate


# --- Weather-code mapping --------------------------------------------------

_WMO = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm w/ slight hail",
    99: "Thunderstorm w/ heavy hail",
}


def _wmo_text(code) -> str:
    try:
        return _WMO.get(int(code), f"Code {code}")
    except (TypeError, ValueError):
        return "Unknown"


# --- HTTP helper -----------------------------------------------------------


def _get_json(url: str, params: dict) -> dict:
    qs = urllib.parse.urlencode(params)
    full = f"{url}?{qs}"
    req = urllib.request.Request(full, headers={"User-Agent": "hermes-weather-deterministic/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


# --- Geocoding -------------------------------------------------------------


def geocode(place: str) -> Optional[Tuple[float, float, str]]:
    """Resolve *place* to (lat, lon, label) via Open-Meteo geocoding.

    Returns None if the place can't be resolved.
    """
    try:
        data = _get_json(
            GEOCODE_URL, {"name": place, "count": 1, "language": "en", "format": "json"}
        )
    except Exception:
        return None
    results = data.get("results") or []
    if not results:
        return None
    r = results[0]
    lat = r.get("latitude")
    lon = r.get("longitude")
    if lat is None or lon is None:
        return None
    parts = [r.get("name")]
    if r.get("admin1"):
        parts.append(r["admin1"])
    if r.get("country_code"):
        parts.append(r["country_code"])
    label = ", ".join(str(p) for p in parts if p)
    return float(lat), float(lon), label


# --- Forecast --------------------------------------------------------------


def fetch_forecast(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,apparent_temperature"
        ),
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": TIMEZONE,
        "forecast_days": 3,
    }
    return _get_json(FORECAST_URL, params)


# --- Formatting ------------------------------------------------------------


def format_weather(label: str, data: dict) -> str:
    cur = data.get("current") or {}
    daily = data.get("daily") or {}

    temp = cur.get("temperature_2m")
    feels = cur.get("apparent_temperature")
    hum = cur.get("relative_humidity_2m")
    wind = cur.get("wind_speed_10m")
    code = cur.get("weather_code")

    lines = [f"Weather for {label}"]
    cur_bits = []
    if temp is not None:
        cur_bits.append(f"{round(temp)}°F")
    cur_bits.append(_wmo_text(code))
    lines.append("Now: " + ", ".join(cur_bits))

    detail = []
    if feels is not None:
        detail.append(f"feels like {round(feels)}°F")
    if hum is not None:
        detail.append(f"humidity {round(hum)}%")
    if wind is not None:
        detail.append(f"wind {round(wind)} mph")
    if detail:
        lines.append("  (" + ", ".join(detail) + ")")

    times = daily.get("time") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    dcode = daily.get("weather_code") or []
    pprob = daily.get("precipitation_probability_max") or []

    if times:
        lines.append("")
        lines.append("3-day outlook:")
        for i, day in enumerate(times):
            hi = round(tmax[i]) if i < len(tmax) and tmax[i] is not None else "?"
            lo = round(tmin[i]) if i < len(tmin) and tmin[i] is not None else "?"
            cond = _wmo_text(dcode[i]) if i < len(dcode) else "?"
            pop = pprob[i] if i < len(pprob) and pprob[i] is not None else None
            pop_s = f", {pop}% precip" if pop is not None else ""
            lines.append(f"  {day}: {cond}, {hi}°/{lo}°F{pop_s}")

    return "\n".join(lines)


# --- Top-level entry point -------------------------------------------------


def answer_weather(text: str, explicit_location: Optional[str] = None) -> str:
    """Produce a deterministic weather answer for a free-text request.

    *explicit_location* (if given) overrides location extraction — used when
    the caller already parsed a `/weather <place>` argument.

    Never raises: any failure returns a clear fallback message (still no LLM).
    """
    try:
        place = (explicit_location or "").strip() or extract_location(text)

        if place:
            geo = geocode(place)
            if geo is None:
                return (
                    f'Couldn\'t find a location matching "{place}". '
                    f'Try a city name like "weather in Chicago".'
                )
            lat, lon, label = geo
        else:
            lat, lon, label = DEFAULT_LAT, DEFAULT_LON, DEFAULT_LOCATION_NAME

        data = fetch_forecast(lat, lon)
        if not data.get("current"):
            return (
                f"Weather service returned no current conditions for {label}. "
                f"Please try again shortly."
            )
        return format_weather(label, data)

    except Exception as exc:  # network down, JSON error, etc.
        return (
            "Weather service is temporarily unavailable "
            f"({type(exc).__name__}). Please try again in a moment."
        )

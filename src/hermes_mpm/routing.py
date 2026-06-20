"""Tiered model routing for hermes-mpm.

Why: Every inbound gateway request costs money proportional to the model that
answers it. MPM's core feature is steering each request to the cheapest model
that can do the job — free 70B for background/bulk, a cheap workhorse for simple
interactive turns, the main reasoning model by default, a strong model for
engineering, and the max model only on explicit demand. This is done with a
pure-Python deterministic classifier (zero extra LLM calls) plus a
``pre_gateway_dispatch`` handler that pins the chosen tier's model onto the
gateway session before the agent is built.

What: ``classify(text, platform, profile)`` is a pure function returning
``(grouping, tier)``. ``make_dispatch_handler(config, ...)`` builds the
``pre_gateway_dispatch`` callback that classifies the event, resolves the tier's
provider bundle, writes a COMPLETE session override into the gateway's
``_session_model_overrides`` dict (so it supersedes any profile-pinned model),
evicts the cached agent, and returns None (never rewrites text).

Test: ``classify("is plex up?", "telegram")`` -> ("interactive_simple",
"cheap_workhorse"); ``classify("implement retry backoff", "telegram")`` ->
("engineering", "strong"); ``classify("x", "cron")`` -> background/free; the
dispatch handler writes the right override dict (see test_routing.py).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("hermes_mpm.routing")

# ── Tier names (string resources — never magic strings inline) ──────────────

TIER_FREE_BACKGROUND = "free_background"
TIER_CHEAP_WORKHORSE = "cheap_workhorse"
TIER_MAIN = "main"
TIER_STRONG = "strong"
TIER_MAX = "max"

# ── Grouping names ──────────────────────────────────────────────────────────

GROUP_BACKGROUND = "background"
GROUP_BULK_CLASSIFICATION = "bulk_classification"
GROUP_INTERACTIVE_SIMPLE = "interactive_simple"
GROUP_INTERACTIVE_DEFAULT = "interactive_default"
GROUP_ENGINEERING = "engineering"
GROUP_HARDEST = "hardest"

# ── Default config (overridable via the hermes_mpm config block) ────────────

DEFAULT_TIERS: Dict[str, Dict[str, Any]] = {
    TIER_FREE_BACKGROUND: {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "fallbacks": ["openai/gpt-oss-120b:free", "qwen/qwen3-235b-a22b-2507"],
    },
    TIER_CHEAP_WORKHORSE: {"model": "qwen/qwen3-235b-a22b-2507"},
    TIER_MAIN: {"model": "deepseek/deepseek-v4-flash"},
    TIER_STRONG: {"model": "deepseek/deepseek-v4-pro"},
    TIER_MAX: {"model": "anthropic/claude-sonnet-4.6"},
}

# Profiles whose work is bulk/classification judgment → free background tier.
DEFAULT_BULK_PROFILES = frozenset({"describer", "kb", "memory", "receipts", "ocr"})

# Profiles that are engineering archetypes → strong tier.
DEFAULT_ENGINEERING_PROFILES = frozenset({"engineer", "debugger", "homelab", "mcp-builder"})

# Default profile → tier map (routing WINS over the profile's pinned model
# for any profile listed here). Built from the two profile sets above.
DEFAULT_PROFILE_TIER_MAP: Dict[str, str] = {
    **{p: TIER_FREE_BACKGROUND for p in DEFAULT_BULK_PROFILES},
    **{p: TIER_STRONG for p in DEFAULT_ENGINEERING_PROFILES},
}

# Up-route keywords: presence of any of these forces the strong (engineering)
# tier regardless of length.
DEFAULT_COMPLEXITY_KEYWORDS = (
    "implement",
    "debug",
    "refactor",
    "diagnose",
    "migrate",
    "architect",
    "broken",
    "failing",
    "design",
    "optimize",
)

# Simple-intent verbs for the interactive_simple grouping (status/show/check).
DEFAULT_SIMPLE_KEYWORDS = ("status", "show", "check", "list")

# Status-question shapes that are simple even without a literal keyword above,
# e.g. "is plex up?", "is the gateway running?", "are the workers online?".
# These are cheap status checks → interactive_simple / cheap_workhorse.
_STATUS_QUESTION_RE = re.compile(
    r"\b(?:is|are|was|were)\b.*\b"
    r"(?:up|down|running|online|offline|alive|healthy|active|reachable|working|ok)\b",
    re.IGNORECASE,
)

DEFAULT_THRESHOLDS = {"max_chars": 600, "max_words": 120}

# openrouter provider conventions (this build).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

# Platforms that carry cron / background origin.
_CRON_PLATFORMS = frozenset({"cron", "cronjob", "scheduler"})


# ── Helpers ─────────────────────────────────────────────────────────────────

_TIER_PREFIX_RE = re.compile(r"^\s*/tier\s+(\w+)\b", re.IGNORECASE)


def _word_count(text: str) -> int:
    return len(text.split())


def _has_keyword(lower_text: str, keywords) -> bool:
    return any(kw in lower_text for kw in keywords)


def parse_tier_prefix(text: str) -> Optional[str]:
    """Extract an explicit ``/tier <name>`` prefix, or None.

    Why: An operator must be able to force a tier (esp. ``/tier max``),
    overriding every heuristic — highest precedence.
    What: Matches a leading ``/tier <word>``; returns the lowercased name iff it
    is a known tier, else None.
    Test: parse_tier_prefix("/tier max do X") == "max"; parse_tier_prefix("hi")
    is None; parse_tier_prefix("/tier bogus") is None.
    """
    if not text:
        return None
    m = _TIER_PREFIX_RE.match(text)
    if not m:
        return None
    name = m.group(1).lower()
    return name if name in DEFAULT_TIERS else None


def classify(
    text: str,
    platform: Optional[str] = None,
    profile: Optional[str] = None,
    *,
    urgency: Optional[str] = None,
    profile_tier_map: Optional[Dict[str, str]] = None,
    complexity_keywords=DEFAULT_COMPLEXITY_KEYWORDS,
    simple_keywords=DEFAULT_SIMPLE_KEYWORDS,
    bulk_profiles=DEFAULT_BULK_PROFILES,
    thresholds: Optional[Dict[str, int]] = None,
) -> Tuple[str, str]:
    """Deterministically classify a request into (grouping, tier). Pure, no LLM.

    Why: Single source of routing truth, unit-testable without any gateway. The
    decision precedence (highest first) is: explicit ``/tier`` prefix → platform
    rule (cron → free background) → profile_tier_map → complexity up-route →
    simplicity down-route → default main.
    What: Returns a ``(grouping, tier)`` tuple. ``platform``/``profile``/
    ``urgency`` refine the decision; the maps/keywords are injectable so config
    can override the defaults.
    Test: see test_routing.py decision matrix — e.g. "is plex up?"/telegram ->
    (interactive_simple, cheap_workhorse); "implement …" -> (engineering,
    strong); cron -> (background, free_background); receipts profile ->
    (bulk_classification, free_background); "/tier max …" -> (hardest, max).
    """
    text = text or ""
    plat = (platform or "").strip().lower()
    prof = (profile or "").strip().lower() or None
    profile_tier_map = DEFAULT_PROFILE_TIER_MAP if profile_tier_map is None else profile_tier_map
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    lower = text.lower()

    # 1) Explicit /tier prefix — highest precedence.
    explicit = parse_tier_prefix(text)
    if explicit:
        return GROUP_HARDEST if explicit == TIER_MAX else explicit, explicit

    # 2) Platform / urgency rule — cron or low-urgency background → free.
    if plat in _CRON_PLATFORMS or (urgency or "").strip().lower() == "low":
        return GROUP_BACKGROUND, TIER_FREE_BACKGROUND

    # 3) Profile tier map (routing wins for listed profiles).
    if prof and prof in profile_tier_map:
        tier = profile_tier_map[prof]
        grouping = (
            GROUP_BULK_CLASSIFICATION
            if prof in bulk_profiles
            else GROUP_ENGINEERING
            if tier == TIER_STRONG
            else GROUP_INTERACTIVE_DEFAULT
        )
        return grouping, tier

    has_complexity = _has_keyword(lower, complexity_keywords)

    # 4) Complexity up-route — any complexity keyword forces strong.
    if has_complexity:
        return GROUP_ENGINEERING, TIER_STRONG

    # 5) Simplicity down-route — short, simple status/show/check/list intents,
    #    or a bare "is X up/running?" status question.
    short = len(text) <= thr["max_chars"] and _word_count(text) <= thr["max_words"]
    if short and (_has_keyword(lower, simple_keywords) or _STATUS_QUESTION_RE.search(text)):
        return GROUP_INTERACTIVE_SIMPLE, TIER_CHEAP_WORKHORSE

    # 6) Default — the main reasoning model.
    return GROUP_INTERACTIVE_DEFAULT, TIER_MAIN


# ── Tier → provider bundle resolution ───────────────────────────────────────


def resolve_tier_override(
    tier: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build the COMPLETE session-override bundle for *tier*, or None.

    Why: To supersede a profile's pinned model, the override must include a real
    ``api_key`` so the gateway's fast-path
    (``_resolve_session_agent_runtime``) returns it directly. An incomplete
    bundle (no api_key) would only layer model/provider on top of env
    resolution and could lose to the profile.
    What: Reads the tier's model from config (falling back to DEFAULT_TIERS) and
    the openrouter provider creds (api_key from config or ``OPENROUTER_API_KEY``
    env). Returns ``{model, provider, api_key, base_url, api_mode}`` or None when
    no api_key is resolvable (so we never write a half-override that loses).
    Test: with OPENROUTER_API_KEY set, resolve_tier_override("main", {}) returns
    a dict whose model == deepseek-v4-flash and api_key is non-empty.
    """
    tiers = {**DEFAULT_TIERS, **(config.get("tiers") or {})}
    tier_cfg = tiers.get(tier)
    if not isinstance(tier_cfg, dict):
        logger.debug("hermes-mpm routing: unknown tier %r", tier)
        return None
    model = tier_cfg.get("model")
    if not model:
        return None

    provider_cfg = (config.get("openrouter") or {}) if isinstance(config, dict) else {}
    api_key = (
        provider_cfg.get("api_key")
        or os.environ.get(provider_cfg.get("api_key_env") or OPENROUTER_API_KEY_ENV)
        or os.environ.get(OPENROUTER_API_KEY_ENV)
    )
    if not api_key:
        # No creds → don't write an incomplete override (would risk losing to
        # the profile). Caller logs + falls through.
        logger.debug("hermes-mpm routing: no openrouter api_key; skipping override")
        return None

    base_url = provider_cfg.get("base_url") or OPENROUTER_BASE_URL
    return {
        "model": model,
        "provider": provider_cfg.get("provider") or "openrouter",
        "api_key": api_key,
        "base_url": base_url,
        "api_mode": provider_cfg.get("api_mode"),  # None = OpenAI-compatible
    }


def _platform_enabled(config: Dict[str, Any], platform: str) -> bool:
    """Return whether routing should fire for *platform*.

    Why: CLI is opt-out by default (operators expect their pinned local model
    when running ``hermes chat``); telegram/api/cron are the live surfaces.
    What: Reads ``platforms.<platform>.enabled``; CLI/local default False, all
    other platforms default True.
    Test: _platform_enabled({}, "local") is False; _platform_enabled({},
    "telegram") is True; explicit enabled flag overrides the default.
    """
    platforms = config.get("platforms") or {}
    plat = (platform or "").lower()
    # Normalise CLI aliases.
    key = "cli" if plat in ("cli", "local") else plat
    entry = platforms.get(key)
    if isinstance(entry, dict) and "enabled" in entry:
        return bool(entry["enabled"])
    # Default: CLI/local off, everything else on.
    return key != "cli"


def make_dispatch_handler(
    config: Optional[Dict[str, Any]] = None,
) -> Callable[..., None]:
    """Build the ``pre_gateway_dispatch`` routing handler bound to *config*.

    Why: The handler needs the resolved config (tiers, profile map, creds) at
    registration time; a closure keeps that state without globals and stays
    unit-testable (you can build one with a fake config).
    What: Returns ``handler(event, gateway, session_store, agent_id=None,
    **kw)`` that classifies the event, resolves the tier override, writes it
    into ``gateway._session_model_overrides[session_key]``, evicts the cached
    agent, and returns None. Respects platform gating and an existing manual
    ``/model`` override. Any error is logged and swallowed (never breaks
    dispatch).
    Test: build with a fake config + OPENROUTER_API_KEY; call with a fake
    gateway/event; assert the override dict was written and eviction called.
    """
    cfg = config or {}
    profile_tier_map = cfg.get("profile_tier_map") or DEFAULT_PROFILE_TIER_MAP
    thresholds = cfg.get("thresholds") or DEFAULT_THRESHOLDS
    complexity_keywords = cfg.get("complexity_keywords") or DEFAULT_COMPLEXITY_KEYWORDS
    simple_keywords = cfg.get("simple_keywords") or DEFAULT_SIMPLE_KEYWORDS
    bulk_profiles = frozenset(cfg.get("bulk_profiles") or DEFAULT_BULK_PROFILES)

    def handler(event=None, gateway=None, session_store=None, agent_id=None, **_kw):
        try:
            return _route(
                event=event,
                gateway=gateway,
                cfg=cfg,
                profile_tier_map=profile_tier_map,
                thresholds=thresholds,
                complexity_keywords=complexity_keywords,
                simple_keywords=simple_keywords,
                bulk_profiles=bulk_profiles,
            )
        except Exception as exc:  # never break dispatch
            logger.warning("hermes-mpm routing handler error (ignored): %s", exc)
            return None

    handler.__name__ = "hermes_mpm_routing_dispatch"
    return handler


def _event_platform(event) -> str:
    """Best-effort platform string from an event/source."""
    source = getattr(event, "source", None)
    plat = getattr(source, "platform", None) or getattr(event, "platform", None)
    value = getattr(plat, "value", None)
    return str(value if value is not None else plat or "").lower()


def _route(
    *,
    event,
    gateway,
    cfg: Dict[str, Any],
    profile_tier_map: Dict[str, str],
    thresholds: Dict[str, int],
    complexity_keywords,
    simple_keywords,
    bulk_profiles,
) -> None:
    """Core routing logic (separated so make_dispatch_handler can wrap errors).

    Why: Keeps the try/except wrapper thin and the testable logic explicit.
    What: classify → gate by platform → respect manual override → resolve tier
    bundle → write override + evict. Returns None always (no text rewrite).
    Test: covered via the handler in test_routing.py.
    """
    text = (getattr(event, "text", "") or "").strip()
    platform = _event_platform(event)
    urgency = getattr(event, "urgency", None)
    source = getattr(event, "source", None)
    profile = getattr(source, "profile", None) or getattr(event, "profile", None)

    if not text:
        return None
    if not _platform_enabled(cfg, platform):
        logger.debug("hermes-mpm routing: platform %r opted out", platform)
        return None

    grouping, tier = classify(
        text,
        platform=platform,
        profile=profile,
        urgency=urgency,
        profile_tier_map=profile_tier_map,
        complexity_keywords=complexity_keywords,
        simple_keywords=simple_keywords,
        bulk_profiles=bulk_profiles,
        thresholds=thresholds,
    )

    if gateway is None:
        logger.debug("hermes-mpm routing: no gateway; tier=%s (dry)", tier)
        return None

    # Compute the session key the gateway will resolve for this source.
    try:
        session_key = gateway._session_key_for_source(source)
    except Exception as exc:
        logger.debug("hermes-mpm routing: session key resolve failed: %s", exc)
        return None
    if not session_key:
        return None

    overrides = getattr(gateway, "_session_model_overrides", None)
    if overrides is None:
        return None

    # Respect a manual /model override already in place — don't stomp it.
    # We mark our own writes with a sentinel so we may re-route across turns
    # but never clobber an operator's explicit /model.
    existing = overrides.get(session_key)
    if isinstance(existing, dict) and not existing.get("_hermes_mpm"):
        logger.debug(
            "hermes-mpm routing: manual /model override present on %s; deferring",
            session_key,
        )
        return None

    bundle = resolve_tier_override(tier, cfg)
    if bundle is None:
        return None

    # Idempotent: skip the write+evict if the same model is already pinned by us.
    if isinstance(existing, dict) and existing.get("model") == bundle["model"]:
        return None

    bundle = {**bundle, "_hermes_mpm": True, "_hermes_mpm_tier": tier}
    overrides[session_key] = bundle
    logger.info(
        "hermes-mpm routing: %s/%s -> tier=%s model=%s session=%s",
        platform or "?",
        grouping,
        tier,
        bundle["model"],
        session_key,
    )

    # Evict the cached agent so the next build picks up the new model.
    try:
        gateway._evict_cached_agent(session_key)
    except Exception as exc:
        logger.debug("hermes-mpm routing: evict failed (non-fatal): %s", exc)

    return None

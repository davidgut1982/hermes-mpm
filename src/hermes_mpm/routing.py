"""Tiered model routing for hermes-mpm.

Why: Every inbound gateway request costs money proportional to the model that
answers it. MPM's core feature is steering each request to the cheapest model
that can do the job — free 70B for background/bulk, a cheap workhorse for simple
interactive turns, the main reasoning model by default, a strong model for
engineering, and the max model only on explicit demand. This is done with a
pure-Python deterministic classifier (zero extra LLM calls) plus a
``pre_llm_call`` handler that RETURNS the chosen tier's complete model bundle so
the engine swaps the model before the first LLM call — on every surface
(gateway, TUI, dashboard) identically, with no gateway-internal coupling.

What: ``classify(text, platform, profile)`` is a pure function returning
``(grouping, tier)``. ``make_pre_llm_call_handler(config)`` builds the
``pre_llm_call`` callback that reads ``user_message``/``platform``/``model`` from
kwargs, classifies the request, resolves the tier's provider bundle, and RETURNS
``{model, provider, api_key, base_url, api_mode}`` (or None to defer). It honors
an operator's manual ``/model`` pin (a live model the tiers would never produce)
by returning None.

Test: ``classify("is plex up?", "telegram")`` -> ("interactive_simple",
"cheap_workhorse"); ``classify("implement retry backoff", "telegram")`` ->
("engineering", "strong"); ``classify("x", "cron")`` -> background/free; the
handler returns the right bundle dict (see test_routing.py).
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
    the shared ``openrouter`` provider creds (api_key from config or
    ``OPENROUTER_API_KEY`` env). A tier may override the provider per-tier (its
    own ``provider``/``base_url``/``api_key``/``api_key_env``) — needed when most
    tiers run on one endpoint (e.g. z.ai coding) but ``free_background`` runs on
    OpenRouter's ``:free`` pool. Returns ``{model, provider, api_key, base_url,
    api_mode}`` or None when no api_key is resolvable (so we never write a
    half-override that loses).
    Test: with OPENROUTER_API_KEY set, resolve_tier_override("main", {}) returns
    a dict whose model == deepseek-v4-flash and api_key is non-empty; a tier with
    its own ``base_url`` overrides the shared one.
    """
    tiers = {**DEFAULT_TIERS, **(config.get("tiers") or {})}
    tier_cfg = tiers.get(tier)
    if not isinstance(tier_cfg, dict):
        logger.debug("hermes-mpm routing: unknown tier %r", tier)
        return None
    model = tier_cfg.get("model")
    if not model:
        return None

    # Per-tier provider override merged over the shared openrouter block, so a
    # single tier (free_background) can target a different endpoint/creds.
    shared_cfg = (config.get("openrouter") or {}) if isinstance(config, dict) else {}
    _tier_provider = {
        k: tier_cfg[k]
        for k in ("provider", "base_url", "api_key", "api_key_env", "api_mode")
        if k in tier_cfg
    }
    provider_cfg = {**shared_cfg, **_tier_provider}
    # If the tier redirects creds (its own api_key_env/api_key) but not the raw
    # api_key, drop the shared block's inherited api_key so the tier's
    # api_key_env env lookup wins instead of the wrong (shared) key.
    if ("api_key_env" in _tier_provider or "provider" in _tier_provider) and \
            "api_key" not in _tier_provider:
        provider_cfg.pop("api_key", None)
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

    Why: Cross-surface unification — telegram, the TUI (``hermes chat``) and the
    dashboard must all get the SAME tiered-routing experience (David's call). The
    only opt-out default is a bare ``cli``/``local`` invocation, where an operator
    running a one-off command expects their pinned local model (a manual
    ``/model`` is still honored everywhere).
    What: Reads ``platforms.<platform>.enabled``; ``cli``/``local`` default False,
    every other platform (telegram, api*, cron, **tui**, **dashboard**) defaults
    True. An explicit ``enabled`` flag always overrides the default.
    Test: _platform_enabled({}, "tui") is True; _platform_enabled({}, "dashboard")
    is True; _platform_enabled({}, "cli") is False; an explicit flag overrides.
    """
    platforms = config.get("platforms") or {}
    plat = (platform or "").lower()
    # Normalise CLI aliases (bare local invocation), but NOT tui/dashboard —
    # those route like telegram per the cross-surface decision.
    key = "cli" if plat in ("cli", "local") else plat
    entry = platforms.get(key)
    if isinstance(entry, dict) and "enabled" in entry:
        return bool(entry["enabled"])
    # Default: only a bare cli/local invocation opts out; all else routes.
    return key != "cli"


def _known_tier_models(config: Dict[str, Any]) -> frozenset:
    """All model ids the routing tiers can produce (for manual-pin detection).

    Why: On the pre_llm_call seam there is no gateway session to inspect, so we
    detect an operator's manual ``/model`` by its model id: if the live model is
    NOT one this plugin would ever route to, the operator pinned it — defer.
    What: Returns the set of tier ``model`` strings + their fallbacks (config
    overrides ∪ defaults).
    Test: with default config it contains DEFAULT_TIERS[main]['model'] and not a
    foreign model like 'anthropic/claude-opus-4.6'.
    """
    tiers = {**DEFAULT_TIERS, **((config or {}).get("tiers") or {})}
    models = set()
    for spec in tiers.values():
        if isinstance(spec, dict) and spec.get("model"):
            models.add(spec["model"])
            for fb in spec.get("fallbacks") or []:
                models.add(fb)
    return frozenset(models)


def make_pre_llm_call_handler(
    config: Optional[Dict[str, Any]] = None,
) -> Callable[..., Optional[Dict[str, Any]]]:
    """Build the ``pre_llm_call`` routing handler bound to *config*.

    Why: The handler needs the resolved config (tiers, profile map, creds) at
    registration time; a closure keeps that state without globals and stays
    unit-testable. Running on ``pre_llm_call`` (not ``pre_gateway_dispatch``)
    means the SAME handler routes every surface — gateway, TUI, dashboard — with
    no gateway-internal coupling (no session-override map, no agent eviction).
    What: Returns ``handler(**kw)`` that reads ``user_message``/``platform``/
    ``model``/``agent`` from kwargs, gates by platform, defers to an operator's
    manual ``/model`` pin (the durable ``agent._user_model_pin`` flag the
    surface sets, with a name heuristic as fallback — so a pin survives even
    when the pinned model equals a tier model), classifies the request,
    resolves the tier bundle, and RETURNS
    ``{model, provider, api_key, base_url, api_mode}`` — or None to defer. The
    engine applies the bundle via switch_model. Any error is logged and
    swallowed (returns None → the turn proceeds on the profile model).
    Test: build with a fake config + OPENROUTER_API_KEY; assert the returned
    bundle for an engineering message, and None for a manual pin / cli / no-key.
    """
    cfg = config or {}
    profile_tier_map = cfg.get("profile_tier_map") or DEFAULT_PROFILE_TIER_MAP
    thresholds = cfg.get("thresholds") or DEFAULT_THRESHOLDS
    complexity_keywords = cfg.get("complexity_keywords") or DEFAULT_COMPLEXITY_KEYWORDS
    simple_keywords = cfg.get("simple_keywords") or DEFAULT_SIMPLE_KEYWORDS
    bulk_profiles = frozenset(cfg.get("bulk_profiles") or DEFAULT_BULK_PROFILES)
    known_models = _known_tier_models(cfg)

    def handler(**kw) -> Optional[Dict[str, Any]]:
        try:
            text = (kw.get("user_message") or "")
            text = text.strip() if isinstance(text, str) else ""
            platform = (kw.get("platform") or "").lower()
            current_model = kw.get("model") or ""
            profile = kw.get("profile")
            urgency = kw.get("urgency")
            agent = kw.get("agent")

            if not text:
                return None
            if not _platform_enabled(cfg, platform):
                logger.debug("hermes-mpm routing: platform %r opted out", platform)
                return None

            # Manual-pin precedence (durable signal): the surface sets
            # ``agent._user_model_pin`` when the operator pinned a model via
            # ``/model`` and clears it on a return to auto. This is reliable even
            # when the pinned model IS one of our tier models (e.g. /model
            # glm-5.2 == strong tier), which the name heuristic below cannot
            # distinguish. Defer whenever the explicit pin is set.
            if getattr(agent, "_user_model_pin", False):
                logger.debug(
                    "hermes-mpm routing: user pinned model %r via /model; deferring",
                    current_model,
                )
                return None

            # Secondary heuristic (no agent flag available, e.g. older engine or
            # a surface that doesn't set it): if the live model is one the tiers
            # would never produce, treat it as an operator pin and defer.
            if current_model and current_model not in known_models:
                logger.debug(
                    "hermes-mpm routing: live model %r is operator-pinned; deferring",
                    current_model,
                )
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

            bundle = resolve_tier_override(tier, cfg)
            if bundle is None:
                return None

            # Idempotent: if the resolved tier model already matches the live
            # model, there's nothing to swap — let the turn proceed unchanged.
            if current_model and bundle["model"] == current_model:
                return None

            logger.info(
                "hermes-mpm routing: %s/%s -> tier=%s model=%s",
                platform or "?", grouping, tier, bundle["model"],
            )
            return bundle
        except Exception as exc:  # never break the turn
            logger.warning("hermes-mpm routing handler error (ignored): %s", exc)
            return None

    handler.__name__ = "hermes_mpm_routing_pre_llm_call"
    return handler

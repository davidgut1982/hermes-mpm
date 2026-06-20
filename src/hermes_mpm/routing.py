"""Tier-routing stub for hermes-mpm.

Why: MPM's job is to send each request to the right archetype/tier (local
flash vs cloud reasoning). v0.1 only needs the SHAPE wired so the next task
can drop real classification in without touching the register() hub.
What: ``classify()`` always returns the default tier; ``pre_gateway_dispatch``
is a no-op observer that logs the intended tier and never alters dispatch.
Test: ``classify("anything")`` returns DEFAULT_TIER; the dispatch handler
returns None for any event (so normal dispatch proceeds unchanged).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("hermes_mpm.routing")

DEFAULT_TIER = "default"


def classify(text: str) -> str:
    """Classify a request into a routing tier (STUB — always DEFAULT_TIER).

    Why: Placeholder for the real tier classifier so callers can be written now.
    What: Returns DEFAULT_TIER regardless of input.
    Test: Assert classify("debug the gateway") == DEFAULT_TIER.
    """
    return DEFAULT_TIER


def pre_gateway_dispatch(event=None, gateway=None, session_store=None, **_kw) -> Optional[dict]:
    """No-op pre-dispatch observer that logs the intended tier (STUB).

    Why: Reserves the routing hook surface; v1 will return rewrite/skip
    decisions here. As a stub it must never change gateway behavior.
    What: Reads event.text, logs the tier classify() would pick, returns None.
    Test: Call with a fake event having ``.text``; assert it returns None and
    does not raise.
    """
    try:
        text = getattr(event, "text", "") or ""
    except Exception:
        return None
    if text.strip():
        logger.debug("hermes-mpm routing (stub): intended tier=%s", classify(text))
    return None

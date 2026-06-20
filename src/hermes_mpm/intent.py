"""pre_gateway_dispatch passthrough for hermes-mpm.

Why: hermes-mpm reserves a place in the gateway pre-dispatch chain for future
intent shortcuts (deterministic, no-LLM handlers). v0.1 must occupy that slot
without affecting any message — a strict passthrough.
What: ``passthrough()`` matches the pre_gateway_dispatch kwarg contract and
always returns None (normal dispatch).
Test: Call passthrough() with arbitrary kwargs (and with none); assert it
returns None every time and never raises.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("hermes_mpm.intent")


def passthrough(event=None, gateway=None, session_store=None, **_kw) -> Optional[dict]:
    """Pre-dispatch passthrough — always defers to normal dispatch (STUB).

    Why: Holds the hook registration so v1 can add real intent matching here
    without changing the register() hub.
    What: Returns None unconditionally; any internal error is swallowed so it
    can never break gateway dispatch.
    Test: Assert passthrough(event=object()) is None and passthrough() is None.
    """
    return None

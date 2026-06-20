"""Minimal, bounded cluster-ops MCP client for the deterministic intent handlers.

Why: The disk and service-status intents need data from the cluster-ops MCP
server (``disk_usage`` / ``service_status`` on host ``hermes``).  That server is
a streamable-HTTP MCP endpoint behind a Bearer token.  We call it directly from
pre-LLM Python so the handlers never spin up an agent/LLM — latency is the whole
point.  The token is a SECRET and is read from the environment
(``CLUSTER_OPS_TOKEN``), never hardcoded; the URL from ``CLUSTER_OPS_URL``.

What: ``call_tool(tool_name, arguments)`` runs the MCP handshake (initialize →
notifications/initialized → tools/call) over HTTP using only the Python
standard library (urllib), parses the SSE/JSON response, and returns the tool's
parsed JSON result dict — or ``None`` on ANY failure (missing env, transport
error, HTTP error, MCP error, parse error, ``isError``).  Strict fall-through:
``None`` means "I am not confident, let the agent handle it."

Bounds: a hard total wall-clock ceiling (default 6s) plus per-request socket
timeouts.  Never raises into the caller — every failure path returns ``None``.

Test: with ``CLUSTER_OPS_TOKEN`` unset, ``call_tool(...)`` returns ``None``.
Live behaviour is covered by the gateway-tester (a real disk_usage round-trip).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes_plugins.intent_handlers_core.cluster_ops")

# Hard wall-clock ceiling for the whole handshake+call sequence.
_TOTAL_CEILING_S = 6.0
# Per-HTTP-request socket timeout.
_PER_REQUEST_TIMEOUT_S = 4.0

_PROTOCOL_VERSION = "2025-03-26"


def _env_url() -> Optional[str]:
    url = (os.environ.get("CLUSTER_OPS_URL") or "").strip()
    return url or None


def _env_token() -> Optional[str]:
    tok = (os.environ.get("CLUSTER_OPS_TOKEN") or "").strip()
    return tok or None


def _post(
    url: str,
    token: str,
    payload: dict,
    session_id: Optional[str],
    deadline: float,
):
    """POST one JSON-RPC message; return (parsed_body_or_None, session_id).

    Parses both raw JSON and ``text/event-stream`` (SSE) responses.  Returns
    ``(None, session_id)`` on any transport/HTTP/parse failure.  Respects the
    overall ``deadline`` (a monotonic-ish wall-clock stamp) by shrinking the
    socket timeout so the total never blows the ceiling.
    """
    remaining = deadline - time.time()
    if remaining <= 0:
        return None, session_id
    timeout = min(_PER_REQUEST_TIMEOUT_S, remaining)

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    if session_id:
        req.add_header("mcp-session-id", session_id)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            new_sid = resp.headers.get("mcp-session-id") or session_id
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        logger.debug("cluster-ops POST failed: %s", exc)
        return None, session_id

    body = _parse_jsonrpc(raw)
    return body, new_sid


def _parse_jsonrpc(raw: str) -> Optional[dict]:
    """Extract the first JSON-RPC object from a raw or SSE response body."""
    raw = raw.strip()
    if not raw:
        return None
    # SSE: lines like "event: message" / "data: {...}". Grab the first data line.
    if "data:" in raw and raw.lstrip().startswith(("event:", "data:", ":")):
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                if payload and payload != "[DONE]":
                    try:
                        return json.loads(payload)
                    except json.JSONDecodeError:
                        continue
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Optional[dict]:
    """Call a cluster-ops MCP tool and return its parsed JSON result, or None.

    Why: single entry point for the disk/service-status handlers.  Bullet-proof:
    any failure (no creds, transport, HTTP, MCP error, parse, isError, ceiling)
    yields ``None`` so the handler falls through to the agent.
    What: runs initialize → notifications/initialized → tools/call within a hard
    wall-clock ceiling, then returns ``json.loads(result.content[0].text)``.
    Test: ``CLUSTER_OPS_TOKEN`` unset -> None.
    """
    url = _env_url()
    token = _env_token()
    if not url or not token:
        logger.debug("cluster-ops creds not configured; falling through")
        return None

    deadline = time.time() + _TOTAL_CEILING_S

    # 1. initialize
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "intent-handlers-core", "version": "1.0"},
        },
    }
    init_body, session_id = _post(url, token, init_payload, None, deadline)
    if init_body is None or "result" not in init_body:
        return None

    # 2. notifications/initialized (best-effort; ignore body)
    _post(
        url,
        token,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        session_id,
        deadline,
    )

    # 3. tools/call
    call_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    call_body, _ = _post(url, token, call_payload, session_id, deadline)
    if call_body is None:
        return None
    if call_body.get("error"):
        logger.debug("cluster-ops tool error: %s", call_body.get("error"))
        return None
    result = call_body.get("result")
    if not isinstance(result, dict) or result.get("isError"):
        return None

    # Extract the text content and parse it as JSON (cluster-ops returns JSON).
    for content in result.get("content", []):
        if isinstance(content, dict) and content.get("type") == "text":
            text = content.get("text")
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None

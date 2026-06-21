"""Reviewer HTTP client — independent TIGHTEN-ONLY review via stdlib urllib.

Why: Sends the review prompt to an independent reviewer model on an
OpenAI-compatible endpoint. No third-party HTTP deps so the gate stays
dependency-light and easy to vendor.
What: ``build_prompt`` renders the tighten-only system+user prompt;
``call_reviewer`` POSTs to /chat/completions and returns the response content.
Test: mock urllib; assert JSON body + auth header; build_prompt content.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

_SYSTEM_PROMPT = (
    "You are a code-review gate. Your role is TIGHTEN-ONLY: you may add "
    "constraints or BLOCK the request. You must NOT rewrite, rephrase, or remove "
    "existing constraints. You must NOT grant tools or permissions not already "
    "present. For each task reviewed, output one of:\n"
    "  ALLOW\n"
    "  TIGHTEN: <constraint to add>\n"
    "  BLOCK: <reason>\n"
    "Never output any other format."
)


@dataclass
class ReviewerConfig:
    provider: str = "openrouter"
    model: str = "deepseek/deepseek-v4-pro"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout: float = 30.0


def build_prompt(args: dict) -> str:
    """Render the tighten-only review prompt for a single delegate_task call."""
    return (
        f"SYSTEM: {_SYSTEM_PROMPT}\n\n"
        "USER: Review this delegate_task call:\n"
        f"{json.dumps(args, indent=2, default=str)}"
    )


def call_reviewer(prompt: str, config: ReviewerConfig) -> str:
    """HTTP call to the reviewer model via stdlib urllib. Returns response text.

    Why: Independent second opinion on a delegate_task before it executes.
    What: POSTs an OpenAI-compatible chat/completions request; returns the first
    choice's message content. Raises on transport/HTTP errors (caller treats any
    exception as fail-closed via parse_verdict error path).
    Test: mock urllib.request.urlopen; assert URL, body, and auth header.
    """
    api_key = os.environ.get(config.api_key_env, "")
    url = config.base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req, timeout=config.timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")

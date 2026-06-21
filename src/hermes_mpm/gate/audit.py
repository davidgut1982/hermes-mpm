"""Append-only audit log — durable record of every gate decision.

Why: Post-hoc review of what the gate allowed/tightened/blocked, with secrets
never written to disk.
What: JSONL file, one decision per line. Each record is serialized then run
through a redaction regex that masks api_key/token/secret/password/credential/
bearer values before write.
Test: write entry; read back; verify no secret leaked; verify all fields present.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Matches `<sensitive-name><sep><value>` and replaces the value with <REDACTED>.
# Sep covers JSON ("key": "val"), kv (key=val), and bare (key: val) forms.
_REDACT_RE = re.compile(
    r'(?i)(api_key|token|secret|password|credential|bearer)'
    r'(["\s:=]+)'
    r'([^\s,"\'}{]+)'
)


def _redact(text: str) -> str:
    """Mask sensitive values in a serialized record string."""
    return _REDACT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}<REDACTED>", text)


class AuditStore:
    """Append-only JSONL audit store with on-write secret redaction."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path).expanduser()

    def record(self, *, tool_call_id: str, tool_name: str, args: dict,
               blast_radius: str, decision: str, reason: str,
               constraints: list[str]) -> None:
        """Append one decision record. Secrets are redacted before write.

        Why: Durable, secret-free trail of every gate verdict.
        What: Serializes the record to JSON, redacts, appends as one line.
        Test: see test_gate.py audit cases.
        """
        record = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "args": args,
            "blast_radius": blast_radius,
            "decision": decision,
            "reason": reason,
            "constraints": list(constraints or []),
        }
        line = _redact(json.dumps(record, default=str, sort_keys=True))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

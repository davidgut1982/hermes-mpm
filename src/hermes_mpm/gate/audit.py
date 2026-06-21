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

# Pass 1: field-name redaction.  Matches `<sensitive-name><sep><value>` in JSON/KV
# serialized text and replaces the value with <REDACTED>.
# Sep covers JSON ("key": "val"), kv (key=val), and bare (key: val) forms.
_REDACT_NAME_RE = re.compile(
    r'(?i)(api[_-]?key|token|secret|pass(?:word|phrase)?|credential|bearer)'
    r'(["\s:=]+)'
    r'([^\s,"\'}{]+)'
)

# Pass 2: value-pattern redaction.  Catches secrets stored under unexpected field
# names (e.g. DB_PASS, PGPASSWORD) or embedded in goal/task strings as kv pairs.
# Pattern 1: `word=value` or `word: value` where `word` looks like a credential name.
_REDACT_KV_RE = re.compile(
    r'(?i)(pass(?:word|phrase)?|secret|token|api[_-]?key)\s*[=:]\s*(\S+)'
)

# Pass 3: high-entropy blob redaction.  Long (≥32 chars) runs of base64/hex-like
# chars are almost always tokens, keys, or hashes and must not persist in the log.
_REDACT_ENTROPY_RE = re.compile(r'[A-Za-z0-9+/]{32,}')


def _redact(text: str) -> str:
    """Mask sensitive values in a serialized record string.

    Why: Defence in depth — three passes catch field-name matches, embedded kv
    pairs under unexpected names, and raw high-entropy blobs respectively.
    What: Returns the string with sensitive substrings replaced by <REDACTED>.
    Test: see test_gate.py MED-1 cases (DB_PASS, passphrase= in goal, long blob).
    """
    # Pass 1: known-name field redaction (JSON/KV/bare).
    text = _REDACT_NAME_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}<REDACTED>", text
    )
    # Pass 2: embedded kv pairs with credential-like key names.
    text = _REDACT_KV_RE.sub(lambda m: f"{m.group(1)}=<REDACTED>", text)
    # Pass 3: high-entropy blobs (≥32 contiguous base64/alphanum chars).
    text = _REDACT_ENTROPY_RE.sub("<REDACTED>", text)
    return text


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

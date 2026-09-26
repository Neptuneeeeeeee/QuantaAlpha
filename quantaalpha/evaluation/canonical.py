from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize public protocol data deterministically."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(value: Any) -> str:
    """SHA-256 identity for an already-whitelisted manifest."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

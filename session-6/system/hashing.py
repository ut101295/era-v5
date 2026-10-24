"""Content-addressing helpers.

Every immutable object in the system (a shard, a manifest, a packed batch, a
loss mask) is identified by the SHA-256 of a canonical byte serialization. This
is the single primitive behind immutability + lineage (research/03): change the
bytes -> change the hash -> new object, new lineage.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of raw bytes, returned as 'sha256:<hex>'."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a UTF-8 string."""
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace.

    Canonicalization is what makes a hash of a dict reproducible across runs and
    machines -- the same content always serializes to the same bytes.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    """SHA-256 of the canonical JSON serialization of an object."""
    return sha256_text(canonical_json(obj))


def short(hash_str: str, n: int = 12) -> str:
    """Short form of a 'sha256:<hex>' string for logs, e.g. 'sha256:9f3a1b2c…'."""
    if ":" in hash_str:
        prefix, hexpart = hash_str.split(":", 1)
        return f"{prefix}:{hexpart[:n]}…"
    return hash_str[:n] + "…"

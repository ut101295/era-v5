"""Immutable, content-addressed tokenized shards.

A shard is a sealed training object (research/03, assignment §6): tokenize a set
of documents with the *frozen* tokenizer, store the token IDs as an immutable
binary array, and identify the shard by the SHA-256 of those bytes. "If we
modify it, it becomes a new shard with a new hash and a new lineage."

Storage layout (`indexed_bin`, the Megatron-style baseline):
  * `<shard_id>.bin`  -- int32 little-endian token IDs, concatenated documents
  * `<shard_id>.idx.json` -- document index: [{doc_id, start, length}], plus the
    per-document provenance needed to build the manifest and, later, token-span
    ids in the ledger.

`content_hash` is the SHA-256 of the raw `.bin` bytes -- it is the shard's
identity. Any change to the token stream changes it; tampering is detected by
`verify_shard`, which recomputes the hash from disk.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .corpus import Document
from .hashing import sha256_bytes, sha256_json
from .tokenizer_frozen import FrozenTokenizer

TOKEN_DTYPE = "<i4"  # int32 little-endian -> deterministic across machines


@dataclass
class DocSpan:
    doc_id: str
    start: int          # token offset within the shard
    length: int         # number of tokens
    source_id: str
    provenance_tier: str


@dataclass
class Shard:
    shard_id: str
    lane: str
    language: str
    script: str
    license: str
    tokens: np.ndarray            # int32 token ids
    doc_spans: List[DocSpan]
    content_hash: str             # sha256 of the token bytes (identity)
    tokenizer_hash: str
    split: str = "train"          # train | validation | test
    parent_shard_ids: List[str] = field(default_factory=list)
    canary: str | None = None     # eval/test shards embed a canary (research/05)

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[0])

    def token_bytes(self) -> bytes:
        return self.tokens.astype(TOKEN_DTYPE).tobytes()


def _tokens_to_bytes(tokens: np.ndarray) -> bytes:
    return tokens.astype(TOKEN_DTYPE).tobytes()


def build_shard(
    shard_id: str,
    docs: List[Document],
    tokenizer: FrozenTokenizer,
    split: str = "train",
    canary: str | None = None,
    parent_shard_ids: List[str] | None = None,
) -> Shard:
    """Tokenize `docs` (assumed same lane) with the frozen tokenizer -> Shard.

    A logical EOS boundary between documents is represented in the doc index
    (start/length), not by a token, since the Session 2 tokenizer has no EOS id;
    the packer (M2) uses these spans to insert boundaries and reset positions.
    """
    assert docs, "cannot build an empty shard"
    lane = docs[0].lane
    all_ids: List[int] = []
    spans: List[DocSpan] = []
    text_for_canary = canary or ""
    for d in docs:
        text = d.text + (("\n" + text_for_canary) if text_for_canary else "")
        ids = tokenizer.encode(text)
        spans.append(DocSpan(
            doc_id=d.doc_id, start=len(all_ids), length=len(ids),
            source_id=d.source_id, provenance_tier=d.provenance_tier,
        ))
        all_ids.extend(ids)

    tokens = np.asarray(all_ids, dtype=np.int32)
    content_hash = sha256_bytes(_tokens_to_bytes(tokens))
    return Shard(
        shard_id=shard_id, lane=lane,
        language=docs[0].language, script=docs[0].script, license=docs[0].license,
        tokens=tokens, doc_spans=spans,
        content_hash=content_hash, tokenizer_hash=tokenizer.tokenizer_hash,
        split=split, parent_shard_ids=parent_shard_ids or [], canary=canary,
    )


def write_shard(shard: Shard, out_dir: str) -> Dict[str, str]:
    """Persist a shard to `out_dir`; returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, f"{shard.shard_id}.bin")
    idx_path = os.path.join(out_dir, f"{shard.shard_id}.idx.json")
    with open(bin_path, "wb") as f:
        f.write(shard.token_bytes())
    idx = {
        "shard_id": shard.shard_id,
        "lane": shard.lane,
        "token_count": shard.token_count,
        "doc_spans": [vars(s) for s in shard.doc_spans],
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    return {"bin": bin_path, "idx": idx_path}


def load_tokens(shard_id: str, shard_dir: str) -> np.ndarray:
    """Load a shard's token array from disk."""
    bin_path = os.path.join(shard_dir, f"{shard_id}.bin")
    with open(bin_path, "rb") as f:
        return np.frombuffer(f.read(), dtype=TOKEN_DTYPE).astype(np.int32)


def verify_shard(shard_id: str, expected_content_hash: str, shard_dir: str) -> bool:
    """Recompute the content hash from disk and compare -> tamper detection."""
    tokens = load_tokens(shard_id, shard_dir)
    return sha256_bytes(_tokens_to_bytes(tokens)) == expected_content_hash


def shard_set_hash(content_hashes: List[str]) -> str:
    """A `.dir`-style dataset version: hash of the sorted member content hashes.

    Editing any member shard changes its content hash, which changes this set
    hash -- lineage propagation, exactly like DVC's `.dir` entry (research/03).
    """
    return sha256_json(sorted(content_hashes))

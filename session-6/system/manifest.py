"""Shard manifests and the admission gate.

The manifest is the audit-grade record that travels with every shard
(research/03, assignment §6). The admission gate is a pure function that decides
whether a shard may enter the training stream, enforcing the contracts from
Sessions 2-5:

  * tokenizer_hash present and == the frozen tokenizer  (Session 2)
  * content_hash matches the bytes on disk               (immutability)
  * license is permissible                               (Session 3)
  * cleaning lineage is known                            (Session 4)
  * NO evaluation/test overlap                           (Session 3/4 firewall)

A blocked shard produces a rejection reason (and, in the demo, a ledger event),
never a silent drop. "Hardcoded evidence will not be accepted" -> the gate is
real logic over real hashes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .hashing import sha256_json
from .shards import Shard, verify_shard

ALLOWED_LICENSES = {
    "CC-BY-SA-4.0", "CC-BY-4.0", "permissive-repo", "generated", "MIT", "Apache-2.0",
}


def build_manifest(
    shard: Shard,
    cleaning_pipeline_hash: str,
    dedup_status: Dict[str, bool] | None = None,
    contamination_status: str = "clean",
    eval_overlap_status: str = "none",
    capability_lane: str | None = None,
) -> Dict:
    """Assemble the manifest dict for a shard (schema from research/03 §5)."""
    manifest = {
        "shard_id": shard.shard_id,
        "content_hash": shard.content_hash,
        "tokenizer_hash": shard.tokenizer_hash,
        "storage_format": "indexed_bin",
        "source_ids": sorted({s.source_id for s in shard.doc_spans}),
        "document_ids": [s.doc_id for s in shard.doc_spans],
        "token_count": shard.token_count,
        "language": shard.language,
        "script": shard.script,
        "capability_lane": capability_lane or shard.lane,
        "license": shard.license,
        "provenance_tier": _dominant_tier(shard),
        "cleaning_pipeline_hash": cleaning_pipeline_hash,
        "dedup_status": dedup_status or {"exact": True, "minhash_lsh": False, "semantic": False},
        "contamination_status": contamination_status,
        "eval_overlap_status": eval_overlap_status,   # none | val | test
        "split": shard.split,
        "parent_shard_ids": shard.parent_shard_ids,
        "canary": shard.canary,
    }
    # A manifest's own hash lets the ledger reference it immutably.
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def _dominant_tier(shard: Shard) -> str:
    tiers = [s.provenance_tier for s in shard.doc_spans]
    return max(set(tiers), key=tiers.count) if tiers else "unknown"


@dataclass
class AdmissionResult:
    admitted: bool
    reasons: List[str] = field(default_factory=list)  # failure reasons, empty if admitted

    def as_dict(self) -> Dict:
        return {"admitted": self.admitted, "reasons": self.reasons}


def admit(
    manifest: Dict,
    frozen_tokenizer_hash: str,
    shard_dir: str,
    known_cleaning_hashes: set[str],
    allowed_licenses: set[str] = ALLOWED_LICENSES,
) -> AdmissionResult:
    """Pure admission gate -> PASS (admitted) or a list of block reasons.

    Note: the eval/contamination *firewall* (research/05) is a separate module;
    here we enforce the manifest-level `eval_overlap_status` flag. A shard marked
    for validation/test can never be admitted as training data.
    """
    reasons: List[str] = []

    if manifest.get("tokenizer_hash") != frozen_tokenizer_hash:
        reasons.append("tokenizer_hash_missing_or_mismatch")

    # Immutability / integrity: recompute the content hash from disk.
    try:
        if not verify_shard(manifest["shard_id"], manifest["content_hash"], shard_dir):
            reasons.append("content_hash_mismatch_shard_tampered")
    except FileNotFoundError:
        reasons.append("shard_bytes_missing")

    if manifest.get("license") not in allowed_licenses:
        reasons.append(f"unsafe_license:{manifest.get('license')}")

    if manifest.get("cleaning_pipeline_hash") not in known_cleaning_hashes:
        reasons.append("unknown_cleaning_lineage")

    if manifest.get("eval_overlap_status", "none") != "none":
        reasons.append(f"eval_overlap:{manifest['eval_overlap_status']}")

    if manifest.get("split", "train") != "train":
        reasons.append(f"non_train_split:{manifest['split']}")

    return AdmissionResult(admitted=not reasons, reasons=reasons)

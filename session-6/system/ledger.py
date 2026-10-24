"""Append-only consumption and learning ledgers (the run's memory).

Event Sourcing over a JSONL log (research/04): every consumed microbatch appends
one immutable event; run state is *derived by replay*; a checkpoint is a snapshot
bound to a `ledger_offset`. The offset is simply the number of lines, so resume
means "continue appending after offset N", and audit means "filter events by
global_step range".

Two logs:
  * consumption ledger -- what the model SAW (schema from research/04 §2);
  * learning ledger    -- what the model LEARNED (loss delta, grad norm, ppl),
    attaching the outcome back to the source data (research/04 §5).

Plus a per-token perplexity trace for a chosen batch (assignment §11).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List

from .hashing import sha256_bytes


class Ledger:
    """A single append-only JSONL log with an integer offset (= line count)."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            open(path, "w").close()

    def append(self, event: Dict) -> int:
        """Append one event; return the offset it was written at (0-indexed)."""
        offset = self.offset()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return offset

    def offset(self) -> int:
        with open(self.path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def read_all(self) -> List[Dict]:
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def iter_events(self) -> Iterator[Dict]:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def truncate_to(self, offset: int) -> None:
        """Roll the log back to `offset` lines (used by fork/replay from a ckpt)."""
        rows = self.read_all()[:offset]
        with open(self.path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def consumption_event(
    *, run_id: str, branch_id: str, global_step: int, checkpoint_id: str,
    rank: int, microbatch_id: int, packed_sample_ids: List[str],
    shard_ids: List[str], batch_hash: str, loss_mask_hash: str,
    mixture_lane: str, curriculum_stage: str, tokenizer_version: str,
    dataloader_version: str, seed: int, opus_decision_id: str | None,
    opus_status: str | None = None, gradient_applied: bool | None = None,
    n_loss_tokens: int | None = None, n_positions: int | None = None,
    token_spans: List[Dict] | None = None,
) -> Dict:
    return {
        "kind": "consumption",
        "run_id": run_id, "branch_id": branch_id, "global_step": global_step,
        "checkpoint_id": checkpoint_id, "rank": rank, "microbatch_id": microbatch_id,
        "packed_sample_ids": packed_sample_ids, "shard_ids": shard_ids,
        "batch_hash": batch_hash, "loss_mask_hash": loss_mask_hash,
        "attn_pos_policy": "block_diag_causal/per_doc_reset",
        "mixture_lane": mixture_lane, "curriculum_stage": curriculum_stage,
        "tokenizer_version": tokenizer_version, "dataloader_version": dataloader_version,
        "opus_decision_id": opus_decision_id, "seed": seed,
        "opus_status": opus_status, "gradient_applied": gradient_applied,
        "n_loss_tokens": n_loss_tokens, "n_positions": n_positions,
        "token_spans": token_spans or [],
    }


def learning_event(
    *, global_step: int, branch_id: str, lane: str, shard_ids: List[str],
    mean_loss: float, mean_ppl: float, loss_before: float, loss_after: float,
    grad_norm: float, model_phase: str, tokens_consumed: int,
) -> Dict:
    return {
        "kind": "learning",
        "global_step": global_step, "branch_id": branch_id, "lane": lane,
        "shard_ids": shard_ids, "avg_token_loss": round(mean_loss, 6),
        "avg_token_perplexity": round(mean_ppl, 6),
        "loss_before": round(loss_before, 6), "loss_after": round(loss_after, 6),
        "loss_delta": round(loss_after - loss_before, 6),
        "gradient_norm": round(grad_norm, 6), "model_phase": model_phase,
        "tokens_consumed_when_seen": tokens_consumed,
        "usefulness": _classify(loss_after - loss_before),
    }


def _classify(loss_delta: float) -> str:
    """Useful / neutral / harmful classification for future planning (§12)."""
    if loss_delta < -1e-4:
        return "useful"
    if loss_delta > 1e-4:
        return "harmful"
    return "neutral"


def loss_mask_hash(loss_mask_bytes: bytes) -> str:
    return sha256_bytes(loss_mask_bytes)

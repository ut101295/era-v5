"""Trainer: one deterministic training step, fully ledgered.

Ties the batch layer (M2), mixture/OPUS/firewall (M3), and the tiny model (M4)
into a single `train_step(step)` that:

  1. plans the batch (pure fn of seed+step);
  2. asks OPUS to score/accept it under the current stage's protected floors;
  3. measures loss BEFORE the update and AFTER (the real loss-delta learning
     signal, research/04 §5);
  4. appends a consumption event and a learning event to the ledgers;
  5. returns per-token perplexities so a trace can be dumped for one batch.

Because every input is a pure function of (seed, step), the trainer holds no
hidden ordering state -- its only mutable state is the model, the optimizer, and
the ledger offset, which is exactly what a checkpoint serializes (M5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .dataloader import Batch, PlanContext
from .ledger import (Ledger, consumption_event, learning_event, loss_mask_hash)
from .mixture import Timeline
from .model import Adam, TinyLM
from .opus import Opus


def phase_of(stage_name: str) -> str:
    if "seed" in stage_name or "general" in stage_name:
        return "early"
    if "anneal" in stage_name:
        return "anneal"
    return "mid"


@dataclass
class StepResult:
    step: int
    lane: str
    stage: str
    batch_hash: str
    mean_loss: float
    loss_delta: float
    grad_norm: float
    opus_status: str
    n_loss_tokens: int


class Trainer:
    def __init__(
        self,
        model: TinyLM,
        optim: Adam,
        ctx: PlanContext,
        timeline: Timeline,
        opus: Opus,
        consumption: Ledger,
        learning: Ledger,
        run_id: str,
        seed: int,
        branch_id: str = "main",
        dataloader_version: str = "v1",
        opus_ledger: Optional[Ledger] = None,
    ):
        self.model = model
        self.optim = optim
        self.ctx = ctx
        self.timeline = timeline
        self.opus = opus
        self.consumption = consumption
        self.learning = learning
        self.run_id = run_id
        self.seed = seed
        self.branch_id = branch_id
        self.dataloader_version = dataloader_version
        self.opus_ledger = opus_ledger
        self.tokens_consumed = 0

    def plan(self, step: int) -> Batch:
        return self.ctx.plan(self.seed, step)

    def train_step(self, step: int, checkpoint_id: str = "",
                   apply_update: bool = True) -> StepResult:
        batch = self.plan(step)
        stage = self.timeline.stage_for_step(step)
        floors = stage.floors

        # OPUS decision (real model-loss-based value could be used; we pass the
        # batch and let the configured score_fn decide).
        decision = self.opus.decide(batch, stage.name, floors, checkpoint_id)
        if self.opus_ledger is not None:
            self.opus_ledger.append({"kind": "opus_decision", "branch_id": self.branch_id,
                                     "batch_hash": batch.batch_hash, **decision.as_dict()})

        # loss BEFORE the update
        loss_before, per_seq_losses, grads = self.model.loss_and_grad(batch.tensors)
        grad_norm = self.model.grad_global_norm(grads)

        applied = apply_update and decision.status in ("accepted", "protected")
        if applied:
            self.optim.step(self.model.params, grads)

        # loss AFTER the update (same batch) -> real loss delta
        loss_after, _, _ = self.model.loss_and_grad(batch.tensors)

        # mean perplexity over loss-bearing tokens of the microbatch
        ppls = []
        for bt in batch.tensors:
            tp = self.model.token_perplexities(bt)
            ppls.append(tp[bt.loss_mask])
        mean_ppl = float(np.concatenate(ppls).mean()) if any(p.size for p in ppls) else 0.0

        self.tokens_consumed += batch.n_loss_tokens

        # ---- append ledger events ----
        lm_bytes = b"".join(bt.loss_mask.astype(np.uint8).tobytes() for bt in batch.tensors)
        self.consumption.append(consumption_event(
            run_id=self.run_id, branch_id=self.branch_id, global_step=step,
            checkpoint_id=checkpoint_id, rank=0, microbatch_id=0,
            packed_sample_ids=batch.as_ledger_span_ids(), shard_ids=batch.shard_ids,
            batch_hash=batch.batch_hash, loss_mask_hash=loss_mask_hash(lm_bytes),
            mixture_lane=batch.lane, curriculum_stage=stage.name,
            tokenizer_version=self._tok_version(), dataloader_version=self.dataloader_version,
            seed=self.seed, opus_decision_id=decision.candidate_id,
            opus_status=decision.status, gradient_applied=applied,
            n_loss_tokens=batch.n_loss_tokens, n_positions=batch.n_positions,
            token_spans=batch.token_spans,
        ))
        self.learning.append(learning_event(
            global_step=step, branch_id=self.branch_id, lane=batch.lane,
            shard_ids=batch.shard_ids, mean_loss=loss_before, mean_ppl=mean_ppl,
            loss_before=loss_before, loss_after=loss_after, grad_norm=grad_norm,
            model_phase=phase_of(stage.name), tokens_consumed=self.tokens_consumed,
        ))

        return StepResult(
            step=step, lane=batch.lane, stage=stage.name, batch_hash=batch.batch_hash,
            mean_loss=loss_before, loss_delta=loss_after - loss_before,
            grad_norm=grad_norm, opus_status=decision.status,
            n_loss_tokens=batch.n_loss_tokens,
        )

    def token_trace(self, step: int, tokenizer, max_tokens: int = 64) -> List[Dict]:
        """Per-token perplexity trace for one batch (assignment §11)."""
        batch = self.plan(step)
        out: List[Dict] = []
        for si, bt in enumerate(batch.tensors):
            ppl = self.model.token_perplexities(bt)
            for pos in range(bt.input_ids.shape[0]):
                if not bt.loss_mask[pos]:
                    continue
                tid = int(bt.input_ids[pos])
                out.append({
                    "seq": si, "position": pos, "token_id": tid,
                    "decoded_preview": tokenizer.token_str(tid) if tid < tokenizer.vocab_size else "<special>",
                    "doc_id": int(bt.doc_ids[pos]), "lane": batch.lane,
                    "loss_mask": True,
                    "cross_entropy": round(float(np.log(max(ppl[pos], 1e-12))), 6),
                    "perplexity": round(float(ppl[pos]), 6),
                })
                if len(out) >= max_tokens:
                    return out
        return out

    def _tok_version(self) -> str:
        return getattr(self.ctx, "tokenizer_hash", "sha256:unknown")

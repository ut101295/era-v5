"""OPUS: candidate batch selection with a protected-floor override + audit trail.

OPUS sits inside the data path (research/05, assignment §10). It scores candidate
batches against a proxy direction and marks each accepted / rejected / deferred,
with a hard **protected-floor override** that rescues scarce lanes the proxy would
otherwise starve. Crucially, rejected clean data is *not discarded silently* -- it
is recorded, because a batch low-value now may be valuable later, and a rejected
Indic/agentic batch may reveal proxy bias.

The score function is pluggable:
  * standalone (M3): a deterministic proxy (utilization + lane-salted jitter);
  * with the model (M4): the real per-batch mean loss (high loss = high expected
    learning value) -- so decisions are grounded in the actual model state.

Decision policy:
  * score >= accept_threshold           -> ACCEPTED
  * below threshold but lane under floor -> PROTECTED (floor override)
  * within a deferral band              -> DEFERRED
  * otherwise                           -> REJECTED (with reason)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .dataloader import Batch


@dataclass
class OpusDecision:
    candidate_id: str
    global_step: int
    lane: str
    stage: str
    shard_ids: List[str]
    score: float
    status: str                # accepted | rejected | deferred | protected
    reason: str
    floor_override: bool
    effective_tokens: int      # loss-bearing tokens this batch would contribute

    def as_dict(self) -> Dict:
        return {
            "candidate_id": self.candidate_id, "global_step": self.global_step,
            "lane": self.lane, "curriculum_stage": self.stage,
            "shard_ids": self.shard_ids, "opus_score": round(self.score, 6),
            "status": self.status, "rejection_reason": self.reason,
            "protected_floor_override": self.floor_override,
            "effective_token_estimate": self.effective_tokens,
        }


ScoreFn = Callable[[Batch], float]


def proxy_score(batch: Batch) -> float:
    """Deterministic standalone proxy: utilization + small lane-salted jitter."""
    salt = int(hashlib.sha1(f"{batch.lane}:{batch.global_step}".encode()).hexdigest()[:6], 16)
    jitter = (salt % 1000) / 1000.0 * 0.2  # [0, 0.2)
    return float(batch.utilization) + jitter


class Opus:
    def __init__(
        self,
        score_fn: Optional[ScoreFn] = None,
        accept_threshold: float = 0.6,
        defer_band: float = 0.1,
        proxy_version: str = "proxy-v1",
    ):
        self.score_fn = score_fn or proxy_score
        self.accept_threshold = accept_threshold
        self.defer_band = defer_band
        self.proxy_version = proxy_version
        self.decisions: List[OpusDecision] = []
        # running counts to evaluate floors
        self._lane_accepted: Dict[str, int] = {}
        self._total_accepted: int = 0

    def _lane_share(self, lane: str) -> float:
        if self._total_accepted == 0:
            return 0.0
        return self._lane_accepted.get(lane, 0) / self._total_accepted

    def decide(self, batch: Batch, stage_name: str,
               floors: Dict[str, float], checkpoint_id: str = "") -> OpusDecision:
        score = self.score_fn(batch)
        lane = batch.lane
        floor = floors.get(lane, 0.0)
        below_floor = self._lane_share(lane) < floor

        floor_override = False
        if score >= self.accept_threshold:
            status, reason = "accepted", "score>=threshold"
        elif below_floor:
            status, reason = "protected", "protected_floor_override"
            floor_override = True
        elif score >= self.accept_threshold - self.defer_band:
            status, reason = "deferred", "near_threshold_deferred"
        else:
            status, reason = "rejected", "low_proxy_utility"

        cand_id = f"cand-{batch.global_step:06d}-{lane}"
        dec = OpusDecision(
            candidate_id=cand_id, global_step=batch.global_step, lane=lane,
            stage=stage_name, shard_ids=batch.shard_ids, score=score,
            status=status, reason=reason, floor_override=floor_override,
            effective_tokens=batch.n_loss_tokens,
        )
        self.decisions.append(dec)
        if status in ("accepted", "protected"):
            self._lane_accepted[lane] = self._lane_accepted.get(lane, 0) + 1
            self._total_accepted += 1
        return dec

    # ------------------------------------------------------------- audit
    def ledgers(self) -> Dict[str, List[Dict]]:
        """Four ledgers: accepted / rejected / deferred / protected (assignment §10)."""
        out = {"accepted": [], "rejected": [], "deferred": [], "protected": []}
        for d in self.decisions:
            out[d.status].append(d.as_dict())
        return out

    def summary(self) -> Dict:
        led = self.ledgers()
        return {k: len(v) for k, v in led.items()}

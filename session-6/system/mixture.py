"""Mixture timeline compiler: curriculum stages -> per-step lane quotas.

Session 5 described the mixture in human terms (lane shares, protected floors,
anneal reserves, warmup bands). Session 6 turns it into an executable, replayable
per-step lane schedule (research/01 §Megatron-blend, assignment §7).

Mechanism (Megatron's BlendedDataset rule): at each step, pick the lane with the
greatest *sampling error* = target_count - consumed_count. This is deterministic
largest-remainder allocation, so lane frequencies converge to the target weights
and the whole sequence is a pure function of the stage weights -> replayable.

  * Stages carry a token range, per-lane weights, protected floors, and a warmup
    width over which weights blend linearly from the previous stage.
  * Protected floors raise a lane's effective weight to at least its floor, so
    scarce lanes (Indic/agentic/reasoning) are never scheduled below the floor.
  * An anneal reserve marks lanes/shards held back until the anneal stage.

The compiled `Timeline` precomputes the lane sequence up to `max_steps` once
(deterministic), and exposes `lane_schedule(seed, step) -> lane` for the
dataloader. Seed does not affect lane choice (only within-lane order), so replay
of the lane sequence is exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np


@dataclass
class Stage:
    name: str
    token_start: int
    token_end: int
    weights: Dict[str, float]        # per-lane target shares (need not sum to 1; normalized)
    floors: Dict[str, float] = field(default_factory=dict)   # min share per lane
    warmup_tokens: int = 0           # blend from previous stage over this many tokens

    def normalized_weights(self) -> Dict[str, float]:
        # apply floors, then normalize to sum 1
        w = dict(self.weights)
        for lane, fl in self.floors.items():
            w[lane] = max(w.get(lane, 0.0), fl)
        total = sum(w.values()) or 1.0
        return {k: v / total for k, v in w.items()}


class Timeline:
    def __init__(self, stages: List[Stage], tokens_per_step: int,
                 anneal_reserve_lanes: List[str] | None = None):
        assert stages, "need at least one stage"
        self.stages = sorted(stages, key=lambda s: s.token_start)
        self.tokens_per_step = tokens_per_step
        self.anneal_reserve_lanes = anneal_reserve_lanes or []
        self.lanes = sorted({l for s in self.stages for l in s.normalized_weights()})
        self._lane_seq: List[str] = []
        self._weights_seq: List[Dict[str, float]] = []

    # ---------------------------------------------------------------- stages
    def stage_for_token(self, token_pos: int) -> Stage:
        for s in self.stages:
            if s.token_start <= token_pos < s.token_end:
                return s
        return self.stages[-1]

    def stage_for_step(self, step: int) -> Stage:
        return self.stage_for_token(step * self.tokens_per_step)

    def weights_for_step(self, step: int) -> Dict[str, float]:
        """Stage weights, linearly blended across a warmup band from the prior stage."""
        token_pos = step * self.tokens_per_step
        stage = self.stage_for_token(token_pos)
        w = stage.normalized_weights()
        if stage.warmup_tokens > 0 and token_pos < stage.token_start + stage.warmup_tokens:
            prev = self._prev_stage(stage)
            if prev is not None:
                pw = prev.normalized_weights()
                frac = (token_pos - stage.token_start) / stage.warmup_tokens
                frac = min(max(frac, 0.0), 1.0)
                lanes = set(w) | set(pw)
                blended = {l: (1 - frac) * pw.get(l, 0.0) + frac * w.get(l, 0.0)
                           for l in lanes}
                total = sum(blended.values()) or 1.0
                return {k: v / total for k, v in blended.items()}
        return w

    def _prev_stage(self, stage: Stage) -> Stage | None:
        idx = self.stages.index(stage)
        return self.stages[idx - 1] if idx > 0 else None

    # ---------------------------------------------------------------- build
    def build(self, max_steps: int) -> "Timeline":
        """Precompute the deterministic lane sequence via error-diffusion.

        Each lane's target is the running sum of its per-step weight, so a
        stage change only affects steps from that point on. (Using
        `current_weight * total_steps` instead would re-price the whole history
        at the new stage's weights: a lane over-served earlier would then be
        starved for an entire later stage to pay the difference back.)
        """
        consumed = {l: 0.0 for l in self.lanes}
        target = {l: 0.0 for l in self.lanes}
        self._lane_seq = []
        self._weights_seq = []
        for step in range(max_steps):
            w = self.weights_for_step(step)
            self._weights_seq.append(w)
            for lane in self.lanes:
                target[lane] += w.get(lane, 0.0)
            # sampling error = target_count - consumed; pick the max (ties: lane name)
            best_lane, best_err = None, None
            for lane in self.lanes:
                err = target[lane] - consumed[lane]
                key = (err, -ord(lane[0]))  # deterministic tiebreak
                if best_err is None or key > best_err:
                    best_err, best_lane = key, lane
            consumed[best_lane] += 1.0
            self._lane_seq.append(best_lane)
        return self

    def lane_schedule(self, seed: int, step: int) -> str:
        if not self._lane_seq:
            raise RuntimeError("Timeline.build(max_steps) must be called first")
        return self._lane_seq[step % len(self._lane_seq)]

    # ---------------------------------------------------------------- audit
    def planned_shares(self, max_steps: int) -> Dict[str, float]:
        if not self._lane_seq:
            self.build(max_steps)
        counts = {l: 0 for l in self.lanes}
        for lane in self._lane_seq[:max_steps]:
            counts[lane] += 1
        return {l: counts[l] / max_steps for l in self.lanes}

    @staticmethod
    def actual_shares(lane_events: List[str]) -> Dict[str, float]:
        n = len(lane_events) or 1
        out: Dict[str, float] = {}
        for lane in lane_events:
            out[lane] = out.get(lane, 0.0) + 1.0 / n
        return out

    def scarcity_report(self, supply_per_lane: Dict[str, int], max_steps: int,
                        seqs_per_step: int) -> Dict[str, Dict]:
        """Flag lanes whose scheduled demand exceeds available packed sequences.

        Demand met only by repetition (epochs) or synthesis is reported, per §7.
        """
        planned = self.planned_shares(max_steps)
        report: Dict[str, Dict] = {}
        for lane in self.lanes:
            demand = planned[lane] * max_steps * seqs_per_step
            supply = supply_per_lane.get(lane, 0)
            epochs = demand / supply if supply else float("inf")
            report[lane] = {
                "demand_sequences": round(demand, 2),
                "supply_sequences": supply,
                "epochs_required": round(epochs, 2),
                "needs_repetition_or_synthesis": epochs > 1.0,
                "anneal_reserved": lane in self.anneal_reserve_lanes,
            }
        return report


def default_timeline(tokens_per_step: int, token_scale: float = 1.0) -> Timeline:
    """A small 3-stage curriculum for the demo: general -> capability -> anneal.

    Weights echo Session 5's mixture spirit (web-heavy start, capability-heavy
    midtrain, Indic/reasoning-upsampled anneal), with protected floors on the
    scarce lanes so OPUS can never starve them.

    `token_scale` shrinks every stage boundary and warmup band proportionally, so
    a short run can still pass through all three stages without changing the
    stage shapes.
    """
    def t(n: int) -> int:
        return int(round(n * token_scale))

    stages = [
        Stage(
            name="general-web-seed",
            token_start=0, token_end=t(40_000),
            weights={"en": 0.55, "code": 0.20, "math": 0.15, "indic": 0.10},
            floors={"indic": 0.08, "math": 0.05},
            warmup_tokens=0,
        ),
        Stage(
            name="capability-midtrain",
            token_start=t(40_000), token_end=t(90_000),
            weights={"en": 0.35, "code": 0.25, "math": 0.22, "indic": 0.18},
            floors={"indic": 0.12, "math": 0.10},
            warmup_tokens=t(10_000),
        ),
        Stage(
            name="anneal-indic-reasoning",
            token_start=t(90_000), token_end=t(10_000_000),
            weights={"en": 0.25, "code": 0.20, "math": 0.25, "indic": 0.30},
            floors={"indic": 0.25, "math": 0.20},
            warmup_tokens=t(8_000),
        ),
    ]
    return Timeline(stages, tokens_per_step, anneal_reserve_lanes=["indic", "math"])

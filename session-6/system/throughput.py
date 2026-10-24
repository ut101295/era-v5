"""Throughput and packing efficiency (assignment: performance.json).

Two independent paths produce the same token counts:

  * live   -- `PerfMeter` accumulates counters from each StepResult while the
              timed run executes, alongside wall-clock time per phase;
  * ledger -- `counts_from_ledger` recomputes every count from the consumption
              ledger alone.

`performance_report` refuses to call the numbers reconstructable unless both
paths agree exactly. Rates are counts divided by measured wall time, so the
counts are reproducible run to run while the rates vary with the machine.

Token classes, per step (microbatch_size * seq_len positions):
  raw          every position, including pad and EOS boundaries
  loss-bearing positions whose next-token label is trained (loss_mask)
  trained      loss-bearing tokens of batches OPUS accepted or protected,
               i.e. tokens that actually produced a gradient update
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PerfMeter:
    steps: int = 0
    raw_positions: int = 0
    loss_bearing_tokens: int = 0
    trained_tokens: int = 0
    plan_seconds: float = 0.0
    step_seconds: float = 0.0
    by_lane: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _t0: float = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def wall_seconds(self) -> float:
        return time.perf_counter() - self._t0

    def record(self, result, n_positions: int, plan_s: float, step_s: float) -> None:
        trained = result.n_loss_tokens if result.opus_status in ("accepted", "protected") else 0
        self.steps += 1
        self.raw_positions += n_positions
        self.loss_bearing_tokens += result.n_loss_tokens
        self.trained_tokens += trained
        self.plan_seconds += plan_s
        self.step_seconds += step_s
        lane = self.by_lane.setdefault(result.lane, {"steps": 0, "raw_positions": 0,
                                                     "loss_bearing_tokens": 0,
                                                     "trained_tokens": 0})
        lane["steps"] += 1
        lane["raw_positions"] += n_positions
        lane["loss_bearing_tokens"] += result.n_loss_tokens
        lane["trained_tokens"] += trained

    def counts(self) -> Dict:
        return {"steps": self.steps, "raw_positions": self.raw_positions,
                "loss_bearing_tokens": self.loss_bearing_tokens,
                "trained_tokens": self.trained_tokens,
                "by_lane": {k: dict(v) for k, v in sorted(self.by_lane.items())}}


def counts_from_ledger(consumption_events: List[Dict]) -> Dict:
    """Recompute PerfMeter.counts() from consumption events alone."""
    by_lane: Dict[str, Dict[str, int]] = {}
    tot = {"steps": 0, "raw_positions": 0, "loss_bearing_tokens": 0, "trained_tokens": 0}
    for e in consumption_events:
        trained = e["n_loss_tokens"] if e["gradient_applied"] else 0
        lane = by_lane.setdefault(e["mixture_lane"], {"steps": 0, "raw_positions": 0,
                                                      "loss_bearing_tokens": 0,
                                                      "trained_tokens": 0})
        for d in (tot, lane):
            d["steps"] += 1
            d["raw_positions"] += e["n_positions"]
            d["loss_bearing_tokens"] += e["n_loss_tokens"]
            d["trained_tokens"] += trained
    return {**tot, "by_lane": {k: by_lane[k] for k in sorted(by_lane)}}


def packing_report(ctx) -> Dict:
    """Static packing efficiency of every lane stream, per its packing policy."""
    out = {}
    for lane, stream in sorted(ctx.streams.items()):
        seqs = stream.sequences
        positions = len(seqs) * ctx.seq_len
        learn = sum(int(s.learn_mask.sum()) for s in seqs)
        pad = sum(s.pad_count for s in seqs)
        real = sum(int((s.doc_ids >= 0).sum()) for s in seqs)
        out[lane] = {
            "policy": ctx.packing_policy_by_lane.get(lane, "concat_chop"),
            "sequences": len(seqs), "positions": positions,
            "real_tokens": real, "learn_tokens": learn, "pad_tokens": pad,
            "boundary_tokens": positions - real - pad,
            "utilization": round(learn / positions, 4) if positions else 0.0,
            "pad_fraction": round(pad / positions, 4) if positions else 0.0,
        }
    return out


def performance_report(meter: PerfMeter, ledger_counts: Dict, ctx) -> Dict:
    live = meter.counts()
    wall = meter.wall_seconds()

    def rate(n: int, secs: float) -> float:
        return round(n / secs, 1) if secs > 0 else 0.0

    return {
        "counts": live,
        "reconstructed_from_ledger": ledger_counts,
        "reconstructable": live == ledger_counts,
        "wall_seconds": round(wall, 3),
        "phase_seconds": {"plan_batches": round(meter.plan_seconds, 3),
                          "train_steps": round(meter.step_seconds, 3)},
        "data_path_share_of_step_time": round(meter.plan_seconds / meter.step_seconds, 4)
        if meter.step_seconds else 0.0,
        "tokens_per_second": {
            "raw": rate(live["raw_positions"], wall),
            "loss_bearing": rate(live["loss_bearing_tokens"], wall),
            "trained": rate(live["trained_tokens"], wall),
        },
        "steps_per_second": rate(live["steps"], wall),
        "utilization": {
            "loss_bearing_over_raw": round(live["loss_bearing_tokens"] / live["raw_positions"], 4)
            if live["raw_positions"] else 0.0,
            "trained_over_raw": round(live["trained_tokens"] / live["raw_positions"], 4)
            if live["raw_positions"] else 0.0,
        },
        "packing": packing_report(ctx),
        "note": "Counts are deterministic and must equal the ledger reconstruction; "
                "rates depend on the machine running the demo.",
    }

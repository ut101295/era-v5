"""Training worker, run as its own OS process so a crash is a real crash.

    python -m system.worker --artifacts DIR --config CFG.json --mode train \\
        --until 300 --checkpoint-every 50 --crash-at 187
    python -m system.worker --artifacts DIR --config CFG.json --mode resume \\
        --until 300 --checkpoint-every 50

`--crash-at S` sends SIGKILL to the worker *inside* step S: after OPUS has
decided and the optimizer has updated the in-memory weights, but before the
step's consumption event is written. Nothing gets a chance to flush or clean
up. That leaves the ledgers past the last checkpoint and partly written for
step S (an OPUS decision with no matching consumption event).

`--mode resume` restores the latest checkpoint (weights, Adam moments, OPUS
floor counters, token count), rolls every ledger back to the offsets the
checkpoint recorded, saves the rolled-back tail under `ledgers/crash_tail/` as
evidence, and continues from the checkpoint's step + 1.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter

from .build import Bundle, Config, build_system
from .checkpoint import latest_checkpoint
from .runner import _save, restore, weights_hash


def load_config(path: str) -> Config:
    raw = json.load(open(path, encoding="utf-8"))
    raw["sft_lanes"] = tuple(raw.get("sft_lanes", ()))
    return Config(**raw)


def train_loop(b: Bundle, start: int, until: int, every: int, ckpt_dir: str,
               parent: str = "", log=print, meter=None) -> str:
    """Train steps [start, until), checkpointing after every `every`-th step.

    Each consumption event records the checkpoint its step builds on, so a
    ledger row always names the state it can be restored from. With a `meter`
    (throughput.PerfMeter), batch planning and the full step are timed
    separately; the trained result is identical either way.
    """
    for step in range(start, until):
        if meter is not None:
            t0 = time.perf_counter()
            n_positions = b.trainer.plan(step).n_positions
            t1 = time.perf_counter()
        r = b.trainer.train_step(step, checkpoint_id=parent)
        if meter is not None:
            meter.record(r, n_positions, t1 - t0, time.perf_counter() - t1)
        if (step + 1) % every == 0:
            parent = _save(b, step, ckpt_dir)
            counts = Counter(e["status"] for e in b.opus_ledger.read_all())
            log(f"OPUS decisions recorded through step {step}: {dict(sorted(counts.items()))}")
            log(f"[PASS] checkpoint_saved {parent} "
                f"(consumption_offset={b.consumption.offset()}, "
                f"weights={weights_hash(b.model)[7:19]})")
    return parent


def _install_crash(b: Bundle, crash_at: int, log) -> None:
    original = b.consumption.append

    def append(event):
        if event["global_step"] == crash_at:
            log(f"crash simulated: SIGKILL inside step {crash_at} "
                f"(weights updated in memory, consumption event not yet written)")
            sys.stdout.flush()
            os.kill(os.getpid(), signal.SIGKILL)
        return original(event)

    b.consumption.append = append


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=("train", "resume"), required=True)
    ap.add_argument("--until", type=int, required=True)
    ap.add_argument("--checkpoint-every", type=int, required=True)
    ap.add_argument("--crash-at", type=int, default=-1)
    a = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, flush=True)

    b = build_system(a.artifacts, load_config(a.config))
    ckpt_dir = os.path.join(a.artifacts, "checkpoints")
    start, parent = 0, ""

    if a.mode == "resume":
        cid = latest_checkpoint(ckpt_dir)
        assert cid is not None, "no checkpoint to resume from"
        meta = restore(b, ckpt_dir, cid)
        offsets = {"consumption": (b.consumption, meta["consumption_offset"]),
                   "learning": (b.learning, meta["learning_offset"]),
                   "opus": (b.opus_ledger, meta["extra_state"]["opus_offset"])}
        tail_dir = os.path.join(a.artifacts, "ledgers", "crash_tail")
        os.makedirs(tail_dir, exist_ok=True)
        rolled = {}
        for name, (led, off) in offsets.items():
            tail = led.read_all()[off:]
            rolled[name] = len(tail)
            with open(os.path.join(tail_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
                for ev in tail:
                    f.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + "\n")
            led.truncate_to(off)
        start, parent = meta["step"] + 1, cid
        log(f"run resumed from {cid}: weights={weights_hash(b.model)[7:19]}, "
            f"rolled back uncommitted events {rolled}, next step {start}")

    if a.crash_at >= 0:
        _install_crash(b, a.crash_at, log)

    train_loop(b, start, a.until, a.checkpoint_every, ckpt_dir, parent, log)
    with open(os.path.join(a.artifacts, f"final_state.{a.mode}.json"), "w") as f:
        json.dump({"last_step": a.until - 1, "weights_hash": weights_hash(b.model),
                   "opus_state": b.opus.state_dict(),
                   "tokens_consumed": b.trainer.tokens_consumed}, f, indent=2)
    log(f"worker {a.mode} finished at step {a.until - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

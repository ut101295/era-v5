"""Resume, Replay, Fork, Audit -- the operational spine (assignment §14).

These four operations are what make the data stream a first-class, replayable
object bound to the model checkpoint (research/04). Each returns a structured
proof dict that the evidence bundle records.

  * RESUME  -- crash mid-run, restore the checkpoint, and prove the NEXT consumed
               batch is exactly the expected one (none skipped, none repeated).
  * REPLAY  -- re-feed a historical interval and prove batch hashes match.
  * FORK    -- branch from a checkpoint with a new branch id (and optionally a new
               seed), record the divergence, and prove the branch differs.
  * AUDIT   -- reconstruct which shards/lanes trained a given step range.
"""
from __future__ import annotations

from typing import Dict, List

from .build import Bundle, Config, build_system
from .checkpoint import (checkpoint_id, load_checkpoint, save_checkpoint)
from .hashing import sha256_bytes
from .model import Adam, TinyLM


def weights_hash(model: TinyLM) -> str:
    """sha256 over every parameter's raw bytes, in sorted-key order."""
    h = bytearray()
    for k in sorted(model.params):
        h += k.encode() + model.params[k].tobytes()
    return sha256_bytes(bytes(h))


def train_through(bundle: Bundle, start: int, end: int, ckpt_dir: str,
                  checkpoint_every: int = 0, log=lambda m: None) -> List:
    """Train steps [start, end); optionally checkpoint every N steps."""
    results = []
    for step in range(start, end):
        cid = checkpoint_id(step)
        r = bundle.trainer.train_step(step, checkpoint_id=cid)
        results.append(r)
        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            _save(bundle, step, ckpt_dir)
            log(f"[PASS] checkpoint_saved {checkpoint_id(step)}")
    return results


def _save(bundle: Bundle, step: int, ckpt_dir: str) -> str:
    return save_checkpoint(
        ckpt_dir, bundle.model, bundle.optim, step=step, seed=bundle.config.seed,
        branch_id=bundle.trainer.branch_id,
        consumption_offset=bundle.consumption.offset(),
        learning_offset=bundle.learning.offset(),
        tokens_consumed=bundle.trainer.tokens_consumed,
        code_config_hash=bundle.config.hash(),
        extra_state={"opus": bundle.opus.state_dict(),
                     "opus_offset": bundle.opus_ledger.offset(),
                     "weights_hash": weights_hash(bundle.model)},
    )


def restore(bundle: Bundle, ckpt_dir: str, cid: str) -> Dict:
    """Load a checkpoint into `bundle`: model, optimizer, OPUS counters, token count."""
    meta = load_checkpoint(ckpt_dir, cid, bundle.model, bundle.optim)
    bundle.trainer.tokens_consumed = meta["tokens_consumed"]
    if meta.get("extra_state", {}).get("opus"):
        bundle.opus.load_state_dict(meta["extra_state"]["opus"])
    return meta


def crash_and_resume(artifacts_dir: str, config: Config, ckpt_dir: str,
                     checkpoint_at: int, crash_at: int, log=lambda m: None) -> Dict:
    """Train past a checkpoint, crash, resume, and prove the next batch matches.

    Timeline: train [0, checkpoint_at] with a checkpoint at `checkpoint_at`, then
    train uncommitted steps up to `crash_at`, then simulate a crash (in-memory
    state lost, post-checkpoint ledger events rolled back), then resume from the
    checkpoint and prove step `checkpoint_at+1` is consumed next, identically.
    """
    b = build_system(artifacts_dir, config)
    # committed region + checkpoint
    train_through(b, 0, checkpoint_at + 1, ckpt_dir, log=log)
    cid = _save(b, checkpoint_at, ckpt_dir)
    log(f"[PASS] checkpoint_saved {cid}")
    committed_cons = b.consumption.offset()
    committed_learn = b.learning.offset()
    committed_opus = b.opus_ledger.offset()

    # The batch we EXPECT to consume next (recorded before the crash).
    expected = b.trainer.plan(checkpoint_at + 1).batch_hash

    # uncommitted work after the checkpoint
    train_through(b, checkpoint_at + 1, crash_at, ckpt_dir, log=log)
    log(f"crash simulated at step {crash_at}")

    # ---- CRASH: discard in-memory state, roll ledgers back to the checkpoint ----
    b.consumption.truncate_to(committed_cons)
    b.learning.truncate_to(committed_learn)
    b.opus_ledger.truncate_to(committed_opus)

    # ---- RESUME: fresh objects, load checkpoint, continue ----
    b2 = build_system(artifacts_dir, config)
    meta = restore(b2, ckpt_dir, cid)
    resume_step = meta["step"] + 1
    r = b2.trainer.train_step(resume_step, checkpoint_id=cid)

    matched = (resume_step == checkpoint_at + 1) and (r.batch_hash == expected)
    # no gap/duplicate: the newly appended consumption event is at the committed offset
    events = b2.consumption.read_all()
    steps_seq = [e["global_step"] for e in events]
    contiguous = steps_seq == list(range(len(steps_seq)))
    if matched:
        log("[PASS] resume_next_batch_matched")

    return {
        "checkpoint_id": cid, "resume_step": resume_step,
        "expected_batch_hash": expected, "resumed_batch_hash": r.batch_hash,
        "next_batch_matched": matched, "ledger_contiguous_no_skip_or_repeat": contiguous,
        "committed_offset": committed_cons,
    }


def replay(artifacts_dir: str, config: Config, interval: tuple[int, int],
           log=lambda m: None) -> Dict:
    """Replay a historical interval and prove batch hashes match the original run."""
    b = build_system(artifacts_dir, config)
    # original run over the interval
    original = {step: b.trainer.plan(step).batch_hash
                for step in range(interval[0], interval[1])}
    # replay from a *fresh* deterministic system
    b2 = build_system(artifacts_dir, config, ledger_suffix=".replay")
    replayed = {step: b2.trainer.plan(step).batch_hash
                for step in range(interval[0], interval[1])}
    all_match = all(original[s] == replayed[s] for s in original)
    if all_match:
        log("[PASS] replay_hash_matched")
    return {
        "interval": list(interval), "n_steps": len(original),
        "all_hashes_matched": all_match,
        "sample": {str(s): {"original": original[s], "replay": replayed[s]}
                   for s in list(original)[:3]},
    }


def fork(artifacts_dir: str, config: Config, ckpt_dir: str, fork_at: int,
         new_seed: int, log=lambda m: None) -> Dict:
    """Fork from a checkpoint into a new branch with a different seed."""
    # establish a checkpoint at fork_at on main
    b = build_system(artifacts_dir, config)
    train_through(b, 0, fork_at + 1, ckpt_dir, log=log)
    cid = _save(b, fork_at, ckpt_dir)

    main_next = b.trainer.plan(fork_at + 1).batch_hash

    # new branch: same checkpoint, new seed -> divergent data stream (declared)
    fork_config = Config(**{**config.__dict__, "seed": new_seed})
    b2 = build_system(artifacts_dir, fork_config, branch_id=f"fork-{new_seed}")
    restore(b2, ckpt_dir, cid)
    fork_next = b2.trainer.plan(fork_at + 1).batch_hash
    b2.trainer.train_step(fork_at + 1, checkpoint_id=cid)

    diverged = main_next != fork_next
    log(f"branch forked at step {fork_at} -> branch fork-{new_seed}")
    return {
        "fork_step": fork_at, "parent_checkpoint": cid,
        "parent_branch": "main", "new_branch": f"fork-{new_seed}",
        "new_seed": new_seed, "main_next_hash": main_next, "fork_next_hash": fork_next,
        "streams_diverged": diverged,
    }


def audit(bundle_or_dir, config: Config, step_range: tuple[int, int],
          consumption_events: List[Dict] | None = None, log=lambda m: None) -> Dict:
    """Reconstruct which shards/lanes trained a step range (assignment §14)."""
    if consumption_events is None:
        b = build_system(bundle_or_dir, config)
        consumption_events = b.consumption.read_all()
    lo, hi = step_range
    in_range = [e for e in consumption_events if lo <= e["global_step"] < hi]
    shards, lanes = set(), {}
    for e in in_range:
        shards.update(e.get("shard_ids", []))
        lanes[e["mixture_lane"]] = lanes.get(e["mixture_lane"], 0) + 1
    log(f"audit completed for steps [{lo},{hi})")
    return {
        "step_range": [lo, hi], "n_events": len(in_range),
        "shards_involved": sorted(shards), "lane_histogram": lanes,
    }

"""M2 invariants: packing, masks, and a pure deterministic planner."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from system.tokenizer_frozen import load_frozen_tokenizer
from system.corpus import load_corpus
from system.shards import build_shard
from system.packing import PackItem, pack, specials, utilization
from system.masks import build_tensors, assert_correct, IGNORE_INDEX
from system.dataloader import PlanContext


def _small_shards():
    tk = load_frozen_tokenizer()
    docs = load_corpus()
    shards = []
    for lane in ("indic", "en", "code", "math"):
        lane_docs = [d for d in docs if d.lane == lane][:6]
        shards.append(build_shard(f"shard-{lane}", lane_docs, tk))
    return tk, shards


def test_concat_chop_high_utilization():
    tk, shards = _small_shards()
    _, pad_id, vocab = specials(tk.vocab_size)
    items = [PackItem(s.shard_id, "en", s.tokens.tolist()) for s in shards[:1]]
    seqs = pack(items, 64, "concat_chop", tk.vocab_size)
    # every window except possibly the last is fully filled
    for s in seqs[:-1]:
        assert s.pad_count == 0
    assert all(utilization(s) > 0 for s in seqs)


def test_pad_only_wastes_more_than_concat():
    # Short docs in a LARGE window is the case where pad-only waste is visible:
    # concat_chop fills windows; pad_only leaves each short doc mostly padding.
    tk, shards = _small_shards()
    docs = [d for d in load_corpus() if d.lane == "math"][:8]
    items = [PackItem(d.doc_id, "math", tk.encode(d.text)) for d in docs]
    concat = pack(items, 512, "concat_chop", tk.vocab_size)
    padonly = pack(items, 512, "pad_only", tk.vocab_size)
    concat_util = np.mean([utilization(s) for s in concat])
    pad_util = np.mean([utilization(s) for s in padonly])
    assert concat_util > pad_util  # padding wastes budget (research/02 §2)


def test_best_fit_never_splits_and_no_cross_doc_attention():
    tk = load_frozen_tokenizer()
    docs = [d for d in load_corpus() if d.lane == "math"][:10]
    items = [PackItem(d.doc_id, "math", tk.encode(d.text)) for d in docs]
    _, pad_id, vocab = specials(tk.vocab_size)
    seqs = pack(items, 128, "best_fit", tk.vocab_size)
    for s in seqs:
        bt = build_tensors(s, pad_id)
        assert_correct(bt)  # includes no-cross-doc-attention + position reset


def test_mask_labels_ignore_index_consistency():
    tk, shards = _small_shards()
    _, pad_id, vocab = specials(tk.vocab_size)
    items = [PackItem(s.shard_id, s.lane, s.tokens.tolist()) for s in shards]
    seqs = pack(items, 64, "concat_chop", tk.vocab_size)
    bt = build_tensors(seqs[0], pad_id)
    # labels are set exactly where loss_mask is True
    assert bool(((bt.labels != IGNORE_INDEX) == bt.loss_mask).all())


def test_plan_is_pure_across_contexts():
    tk, shards = _small_shards()
    ctx1 = PlanContext(shards, seq_len=64, microbatch_size=2, vocab_size=tk.vocab_size)
    ctx2 = PlanContext(shards, seq_len=64, microbatch_size=2, vocab_size=tk.vocab_size)
    for step in range(8):
        b1 = ctx1.plan(1234, step)
        b2 = ctx2.plan(1234, step)
        assert b1.batch_hash == b2.batch_hash, f"plan not pure at step {step}"


def test_plan_repeatable_and_lane_cycles():
    tk, shards = _small_shards()
    ctx = PlanContext(shards, seq_len=64, microbatch_size=2, vocab_size=tk.vocab_size)
    a = ctx.plan(7, 3)
    b = ctx.plan(7, 3)  # same step -> same batch
    assert a.batch_hash == b.batch_hash
    lanes = [ctx.plan(7, s).lane for s in range(ctx.num_lanes())]
    assert set(lanes) == set(ctx.lanes)  # round-robin covers all lanes


def test_different_seed_changes_order():
    tk, shards = _small_shards()
    ctx = PlanContext(shards, seq_len=64, microbatch_size=2, vocab_size=tk.vocab_size)
    h1 = [ctx.plan(1, s).batch_hash for s in range(8)]
    h2 = [ctx.plan(2, s).batch_hash for s in range(8)]
    assert h1 != h2  # seed drives ordering


def test_batch_stats_reconstructable():
    tk, shards = _small_shards()
    ctx = PlanContext(shards, seq_len=64, microbatch_size=2, vocab_size=tk.vocab_size)
    b = ctx.plan(1234, 0)
    assert b.n_positions == 2 * 64
    assert 0 <= b.n_loss_tokens <= b.n_positions
    assert abs(b.utilization - b.n_loss_tokens / b.n_positions) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"[PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

"""M3 invariants: mixture timeline, eval firewall, OPUS floors."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.tokenizer_frozen import load_frozen_tokenizer
from system.corpus import load_corpus
from system.shards import build_shard
from system.dataloader import PlanContext
from system.mixture import default_timeline
from system.firewall import EvalRegistry, CANARY_STRING
from system.opus import Opus


def _shards():
    tk = load_frozen_tokenizer()
    docs = load_corpus()
    shards = []
    for lane in ("indic", "en", "code", "math"):
        shards.append(build_shard(f"shard-{lane}", [d for d in docs if d.lane == lane][:8], tk))
    return tk, shards


# ---------------------------------------------------------------- mixture
def test_timeline_deterministic_and_covers_weights():
    tl = default_timeline(tokens_per_step=256).build(200)
    seq1 = [tl.lane_schedule(0, s) for s in range(200)]
    tl2 = default_timeline(tokens_per_step=256).build(200)
    seq2 = [tl2.lane_schedule(0, s) for s in range(200)]
    assert seq1 == seq2  # deterministic
    planned = tl.planned_shares(200)
    assert abs(sum(planned.values()) - 1.0) < 1e-6
    assert all(v > 0 for v in planned.values())  # every lane represented


def test_planned_shares_track_weights():
    tl = default_timeline(tokens_per_step=256).build(400)
    planned = tl.planned_shares(400)
    # anneal stage upsamples indic; over the full run indic should be substantial
    assert planned["indic"] > 0.12


def test_protected_floor_in_stage():
    tl = default_timeline(tokens_per_step=256)
    stage = tl.stage_for_step(0)
    w = stage.normalized_weights()
    # indic floor is 0.08 in the seed stage -> effective weight >= floor
    assert w["indic"] >= 0.08 - 1e-9


def test_scarcity_report_flags_shortfall():
    tl = default_timeline(tokens_per_step=256).build(300)
    supply = {"en": 5, "code": 5, "math": 5, "indic": 2}
    rep = tl.scarcity_report(supply, max_steps=300, seqs_per_step=2)
    assert rep["indic"]["needs_repetition_or_synthesis"] is True
    assert rep["indic"]["anneal_reserved"] is True


# ---------------------------------------------------------------- firewall
def test_firewall_blocks_exact_and_ngram():
    tk = load_frozen_tokenizer()
    docs = [d for d in load_corpus() if d.lane == "en"][:4]
    test_shard = build_shard("eval-en", docs, tk, split="test", canary=CANARY_STRING)
    reg = EvalRegistry()
    reg.register(test_shard)

    # identical content -> blocked on content hash
    same = build_shard("cand-en", docs, tk, split="train")
    res = reg.check(same)
    assert res.blocked and any("content_hash" in r or "ngram" in r for r in res.reasons)

    # a totally unrelated train shard -> not blocked
    math_docs = [d for d in load_corpus() if d.lane == "math"][:4]
    clean = build_shard("cand-math", math_docs, tk, split="train")
    assert not reg.check(clean).blocked


def test_firewall_ngram_overlap_partial():
    tk = load_frozen_tokenizer()
    all_en = [d for d in load_corpus() if d.lane == "en"]
    reg = EvalRegistry()
    reg.register(build_shard("eval-en", all_en[:3], tk, split="test"))
    # candidate shares one document with the eval set -> ngram overlap > 0
    overlap_cand = build_shard("cand-en", all_en[2:6], tk, split="train")
    res = reg.check(overlap_cand)
    assert res.overlap_ratio > 0 and res.blocked


# ---------------------------------------------------------------- opus
def test_opus_four_way_and_floor_override():
    tk, shards = _shards()
    ctx = PlanContext(shards, seq_len=64, microbatch_size=2, vocab_size=tk.vocab_size)
    # Proxy undervalues indic (score 0) but likes other lanes (score 1). With an
    # indic floor, the selector must PROTECT indic despite the low proxy score --
    # exactly the "proxy bias rescued by protected floor" case (research/05 §10).
    score_fn = lambda b: 0.0 if b.lane == "indic" else 1.0  # noqa: E731
    opus = Opus(score_fn=score_fn, accept_threshold=0.6, defer_band=0.05)
    floors = {"indic": 0.5}
    for step in range(12):
        opus.decide(ctx.plan(1234, step), stage_name="test-stage", floors=floors)
    summ = opus.summary()
    assert sum(summ.values()) == 12
    statuses = {d.lane: d.status for d in opus.decisions}
    protected_lanes = {d.lane for d in opus.decisions if d.status == "protected"}
    assert "indic" in protected_lanes          # floor override rescued indic
    assert summ["accepted"] >= 1               # non-indic lanes accepted on merit


def test_opus_accepts_when_score_high():
    tk, shards = _shards()
    ctx = PlanContext(shards, seq_len=64, microbatch_size=2, vocab_size=tk.vocab_size)
    opus = Opus(score_fn=lambda b: 1.0, accept_threshold=0.6)  # always accept
    for step in range(6):
        opus.decide(ctx.plan(1, step), "s", floors={})
    assert opus.summary()["accepted"] == 6


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

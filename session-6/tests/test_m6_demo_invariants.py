"""M6 invariants: the pieces run_demo.py relies on.

Covers the multi-stage mixture schedule, checkpointed OPUS state (bit-exact
resume), decontamination, SFT loss masks, token spans, ledger-reconstructed
throughput, and the evidence bundle's fail-closed rule.
"""
import collections
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.build import Config, build_system
from system.checkpoint import checkpoint_id
from system.decontam import contaminated_mask, decontaminate
from system.evidence import Evidence
from system.firewall import EvalRegistry
from system.mixture import default_timeline
from system.runner import _save, restore, weights_hash
from system.throughput import PerfMeter, counts_from_ledger
from system.tokenizer_frozen import load_frozen_tokenizer
from system.corpus import load_corpus
from system.shards import build_shard


def _cfg(**kw):
    base = dict(seed=7, seq_len=96, microbatch_size=2, docs_per_lane=20, max_steps=60,
                accept_threshold=1.02, defer_band=0.08, stage_token_scale=0.1,
                packing_policy={"math": "structure_preserving"}, sft_lanes=("math",))
    base.update(kw)
    return Config(**base)


def test_every_stage_tracks_its_own_weights():
    """Regression: targets used to be current_weight * total_steps, which starved
    lanes for a whole later stage (en/code got 0 batches in the anneal stage)."""
    tl = default_timeline(tokens_per_step=192, token_scale=0.5).build(300)
    for st in tl.stages:
        steps = [s for s in range(300) if tl.stage_for_step(s).name == st.name]
        n = len(steps)
        got = collections.Counter(tl.lane_schedule(0, s) for s in steps)
        for lane in tl.lanes:
            planned = np.mean([tl.weights_for_step(s).get(lane, 0.0) for s in steps])
            assert abs(got[lane] / n - planned) <= 2.0 / n, (st.name, lane)


def test_resume_restores_opus_state_bit_exactly():
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as tmp:
        ref = build_system(os.path.join(tmp, "ref"), cfg)
        for s in range(40):
            ref.trainer.train_step(s)
        statuses = collections.Counter(d.status for d in ref.opus.decisions)
        assert statuses["protected"] > 0  # floors must actually fire for this to mean much

        a = build_system(os.path.join(tmp, "a"), cfg)
        for s in range(20):
            a.trainer.train_step(s)
        ck = os.path.join(tmp, "ck")
        cid = _save(a, 19, ck)
        b = build_system(os.path.join(tmp, "b"), cfg)
        restore(b, ck, cid)
        assert b.opus.state_dict() == a.opus.state_dict()
        for s in range(20, 40):
            b.trainer.train_step(s, checkpoint_id=cid)
        assert weights_hash(b.model) == weights_hash(ref.model)
        assert b.opus.state_dict() == ref.opus.state_dict()


def test_train_shards_clear_firewall_after_decontamination():
    with tempfile.TemporaryDirectory() as tmp:
        b = build_system(tmp, _cfg())
        assert b.decontam_reports["shard-en"]["tokens_removed"] > 0  # real overlap existed
        for s in b.train_shards:
            assert not b.eval_registry.check(s).blocked, s.shard_id


def test_decontaminate_cuts_only_overlapping_tokens():
    tk = load_frozen_tokenizer()
    docs = [d for d in load_corpus() if d.lane == "en"]
    held = build_shard("held", [docs[25]], tk, split="test")
    reg = EvalRegistry()
    reg.register(held)
    victim = docs[0].__class__(**{**docs[0].as_dict(), "text": docs[0].text + "\n" + docs[25].text})
    clean, rep = decontaminate([victim], tk, reg._ngrams)
    assert rep["tokens_removed"] > 0
    assert rep["tokens_removed"] < rep["tokens_in"]  # the clean prefix survives
    for d in clean:
        assert not any(contaminated_mask(tk.encode(d.text), reg._ngrams))


def test_sft_loss_only_on_response_tokens():
    with tempfile.TemporaryDirectory() as tmp:
        b = build_system(tmp, _cfg())
        tk = b.tokenizer
        spans = b.ctx.sft_agentic_spans
        assert spans
        step = next(s for s in range(60) if b.trainer.plan(s).lane == "math")
        batch = b.trainer.plan(step)
        bt = batch.tensors[0]
        targets = tk.decode([int(bt.labels[i]) for i in np.flatnonzero(bt.loss_mask)])
        assert "Question" not in targets and "Reasoning:" in targets


def test_token_spans_account_for_every_loss_token():
    with tempfile.TemporaryDirectory() as tmp:
        b = build_system(tmp, _cfg())
        for s in range(30):
            batch = b.trainer.plan(s)
            assert sum(sp["loss_tokens"] for sp in batch.token_spans) == batch.n_loss_tokens


def test_throughput_counts_reconstruct_from_ledger():
    with tempfile.TemporaryDirectory() as tmp:
        b = build_system(tmp, _cfg())
        meter = PerfMeter()
        meter.start()
        for s in range(25):
            n = b.trainer.plan(s).n_positions
            r = b.trainer.train_step(s, checkpoint_id=checkpoint_id(0))
            meter.record(r, n, 0.0, 0.0)
        assert counts_from_ledger(b.consumption.read_all()) == meter.counts()


def test_evidence_fails_closed():
    ev = Evidence()
    assert ev.result("replay") == "FAIL"  # no checks recorded -> not a pass
    ev.check("replay", "ok", True)
    assert ev.result("replay") == "PASS"
    ev.check("replay", "bad", False)
    assert ev.result("replay") == "FAIL"
    assert not ev.all_passed()

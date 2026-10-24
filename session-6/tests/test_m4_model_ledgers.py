"""M4 invariants: real model learning + append-only ledgers + per-token trace."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from system.tokenizer_frozen import load_frozen_tokenizer
from system.corpus import load_corpus
from system.shards import build_shard
from system.dataloader import PlanContext
from system.mixture import default_timeline
from system.opus import Opus
from system.model import TinyLM, Adam
from system.ledger import Ledger
from system.trainer import Trainer


def _setup(tmp, seq_len=48, mb=2):
    tk = load_frozen_tokenizer()
    docs = load_corpus()
    shards = [build_shard(f"shard-{l}", [d for d in docs if d.lane == l][:8], tk)
              for l in ("indic", "en", "code", "math")]
    tl = default_timeline(tokens_per_step=seq_len * mb).build(400)
    ctx = PlanContext(shards, seq_len=seq_len, microbatch_size=mb,
                      vocab_size=tk.vocab_size, lane_schedule=tl.lane_schedule)
    model = TinyLM(ctx.model_vocab, dim=24, hidden=32, max_pos=seq_len, seed=0)
    opt = Adam(model.params, lr=0.05)
    opus = Opus(score_fn=lambda b: 1.0)  # accept everything for training
    cons = Ledger(os.path.join(tmp, "consumption.jsonl"))
    learn = Ledger(os.path.join(tmp, "learning.jsonl"))
    tr = Trainer(model, opt, ctx, tl, opus, cons, learn, run_id="run-test", seed=7)
    return tk, tr, cons, learn


def test_gradient_check():
    with tempfile.TemporaryDirectory() as tmp:
        tk, tr, _, _ = _setup(tmp)
        b = tr.plan(0).tensors
        loss0, _, grads = tr.model.loss_and_grad(b)
        eps = 1e-5
        k, i, j = "Wout", 4, 10
        orig = tr.model.params[k][i, j]
        tr.model.params[k][i, j] = orig + eps
        lp, _, _ = tr.model.loss_and_grad(b)
        tr.model.params[k][i, j] = orig - eps
        lm, _, _ = tr.model.loss_and_grad(b)
        tr.model.params[k][i, j] = orig
        num = (lp - lm) / (2 * eps)
        assert abs(num - grads[k][i, j]) < 1e-5


def test_training_reduces_loss():
    with tempfile.TemporaryDirectory() as tmp:
        tk, tr, cons, learn = _setup(tmp)
        losses = [tr.train_step(s).mean_loss for s in range(40)]
        assert losses[-1] < losses[0], f"loss did not drop: {losses[0]:.3f}->{losses[-1]:.3f}"
        # every step wrote exactly one consumption + one learning event
        assert cons.offset() == 40
        assert learn.offset() == 40


def test_ledger_append_only_and_offsets():
    with tempfile.TemporaryDirectory() as tmp:
        led = Ledger(os.path.join(tmp, "x.jsonl"))
        assert led.offset() == 0
        o1 = led.append({"a": 1})
        o2 = led.append({"a": 2})
        assert (o1, o2) == (0, 1)
        assert led.offset() == 2
        rows = led.read_all()
        assert [r["a"] for r in rows] == [1, 2]


def test_consumption_event_reconstructs_batch():
    with tempfile.TemporaryDirectory() as tmp:
        tk, tr, cons, _ = _setup(tmp)
        for s in range(5):
            tr.train_step(s)
        events = cons.read_all()
        # the batch_hash in the ledger must equal a fresh re-plan of that step
        for ev in events:
            step = ev["global_step"]
            assert tr.plan(step).batch_hash == ev["batch_hash"]
            assert ev["tokenizer_version"].startswith("sha256:")
            assert ev["opus_decision_id"].startswith("cand-")


def test_learning_ledger_has_real_signal():
    with tempfile.TemporaryDirectory() as tmp:
        tk, tr, _, learn = _setup(tmp)
        for s in range(20):
            tr.train_step(s)
        rows = learn.read_all()
        assert all("loss_delta" in r and "gradient_norm" in r for r in rows)
        # at least some steps should be 'useful' (loss dropped on exposure)
        assert any(r["usefulness"] == "useful" for r in rows)


def test_token_perplexity_trace():
    with tempfile.TemporaryDirectory() as tmp:
        tk, tr, _, _ = _setup(tmp)
        for s in range(5):
            tr.train_step(s)
        trace = tr.token_trace(0, tk, max_tokens=20)
        assert len(trace) > 0
        for t in trace:
            assert t["perplexity"] >= 1.0 - 1e-6      # ppl = exp(CE) >= 1
            assert t["loss_mask"] is True
            assert "decoded_preview" in t


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

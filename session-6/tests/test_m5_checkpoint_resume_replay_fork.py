"""M5 invariants: checkpoint, crash/resume, replay, fork, audit."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.build import Config, build_system
from system.checkpoint import (checkpoint_id, latest_checkpoint,
                               load_checkpoint, save_checkpoint)
from system.model import Adam, TinyLM
from system import runner


def _cfg():
    return Config(seed=1234, seq_len=32, microbatch_size=2, docs_per_lane=8, max_steps=40)


def test_checkpoint_roundtrip_restores_exact_state():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        b = build_system(tmp, cfg)
        ckpt_dir = os.path.join(tmp, "checkpoints")
        for s in range(6):
            b.trainer.train_step(s, checkpoint_id=checkpoint_id(s))
        cid = save_checkpoint(ckpt_dir, b.model, b.optim, step=5, seed=cfg.seed,
                              branch_id="main", consumption_offset=b.consumption.offset(),
                              learning_offset=b.learning.offset(),
                              tokens_consumed=b.trainer.tokens_consumed,
                              code_config_hash=cfg.hash())
        # snapshot a param, restore into a fresh model, compare
        before = b.model.params["Wout"].copy()
        m2 = TinyLM(b.ctx.model_vocab, dim=cfg.dim, hidden=cfg.hidden, max_pos=cfg.seq_len)
        o2 = Adam(m2.params)
        meta = load_checkpoint(ckpt_dir, cid, m2, o2)
        assert (m2.params["Wout"] == before).all()
        assert meta["step"] == 5 and o2.t == b.optim.t


def test_resume_next_batch_matches_exactly():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        ckpt_dir = os.path.join(tmp, "checkpoints")
        proof = runner.crash_and_resume(tmp, cfg, ckpt_dir, checkpoint_at=8, crash_at=12)
        assert proof["next_batch_matched"] is True
        assert proof["resume_step"] == 9
        assert proof["ledger_contiguous_no_skip_or_repeat"] is True


def test_replay_hashes_match():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        proof = runner.replay(tmp, cfg, interval=(3, 15))
        assert proof["all_hashes_matched"] is True
        assert proof["n_steps"] == 12


def test_fork_diverges_with_new_seed():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        ckpt_dir = os.path.join(tmp, "checkpoints")
        proof = runner.fork(tmp, cfg, ckpt_dir, fork_at=6, new_seed=9999)
        assert proof["streams_diverged"] is True
        assert proof["new_branch"] == "fork-9999"


def test_audit_reconstructs_interval():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        b = build_system(tmp, cfg)
        for s in range(20):
            b.trainer.train_step(s, checkpoint_id=checkpoint_id(s))
        events = b.consumption.read_all()
        rep = runner.audit(tmp, cfg, (5, 15), consumption_events=events)
        assert rep["n_events"] == 10
        assert len(rep["shards_involved"]) >= 1
        assert sum(rep["lane_histogram"].values()) == 10


def test_latest_checkpoint_discovery():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        b = build_system(tmp, cfg)
        ckpt_dir = os.path.join(tmp, "checkpoints")
        for s in (2, 5, 9):
            save_checkpoint(ckpt_dir, b.model, b.optim, step=s, seed=cfg.seed,
                            branch_id="main", consumption_offset=0, learning_offset=0,
                            tokens_consumed=0, code_config_hash=cfg.hash())
        assert latest_checkpoint(ckpt_dir) == checkpoint_id(9)


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

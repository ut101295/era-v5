"""Deterministic system assembly.

One place that wires the whole system together from config, so tests, the runner,
and `run_demo.py` all build the *identical* deterministic system. Everything here
is a pure function of the config -> reproducible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

from .corpus import load_corpus
from .dataloader import PlanContext
from .firewall import CANARY_STRING, EvalRegistry
from .hashing import sha256_json
from .ledger import Ledger
from .manifest import build_manifest, admit
from .mixture import default_timeline
from .model import Adam, TinyLM
from .opus import Opus
from .shards import Shard, build_shard
from .tokenizer_frozen import load_frozen_tokenizer
from .trainer import Trainer

CLEANING_HASH = "sha256:session4-pipeline-v1"


@dataclass
class Config:
    seed: int = 1234
    seq_len: int = 48
    microbatch_size: int = 2
    docs_per_lane: int = 10
    max_steps: int = 120
    dim: int = 24
    hidden: int = 32
    lr: float = 0.05
    accept_threshold: float = 0.35   # OPUS: accept most, exercise floors on the rest

    def hash(self) -> str:
        return sha256_json(self.__dict__)


@dataclass
class Bundle:
    config: Config
    tokenizer: object
    train_shards: List[Shard]
    eval_shard: Shard
    manifests: List[Dict]
    admissions: List[Dict]
    eval_registry: EvalRegistry
    timeline: object
    ctx: PlanContext
    model: TinyLM
    optim: Adam
    opus: Opus
    consumption: Ledger
    learning: Ledger
    trainer: Trainer
    lanes: List[str] = field(default_factory=list)


def build_shards(tokenizer, config: Config):
    """Build one train shard per lane + one held-out eval/test shard."""
    docs = load_corpus()
    train_shards: List[Shard] = []
    for lane in ("indic", "en", "code", "math"):
        lane_docs = [d for d in docs if d.lane == lane][: config.docs_per_lane]
        train_shards.append(build_shard(f"shard-{lane}", lane_docs, tokenizer))
    # A held-out English test shard (last few en docs), carrying the canary.
    en_docs = [d for d in docs if d.lane == "en"]
    eval_docs = en_docs[config.docs_per_lane: config.docs_per_lane + 3] or en_docs[:3]
    eval_shard = build_shard("eval-en-heldout", eval_docs, tokenizer,
                             split="test", canary=CANARY_STRING)
    return train_shards, eval_shard


def build_system(artifacts_dir: str, config: Config | None = None,
                 branch_id: str = "main", ledger_suffix: str = "") -> Bundle:
    config = config or Config()
    tk = load_frozen_tokenizer()
    train_shards, eval_shard = build_shards(tk, config)

    # manifests + admission gate
    manifests, admissions = [], []
    ledger_dir = os.path.join(artifacts_dir, "ledgers")
    shard_dir = os.path.join(artifacts_dir, "shard_store")
    os.makedirs(shard_dir, exist_ok=True)
    from .shards import write_shard
    for s in train_shards:
        write_shard(s, shard_dir)
        m = build_manifest(s, cleaning_pipeline_hash=CLEANING_HASH)
        res = admit(m, tk.tokenizer_hash, shard_dir, {CLEANING_HASH})
        manifests.append(m)
        admissions.append({"shard_id": s.shard_id, **res.as_dict()})

    # eval firewall
    reg = EvalRegistry()
    reg.register(eval_shard)

    # mixture timeline
    tokens_per_step = config.seq_len * config.microbatch_size
    timeline = default_timeline(tokens_per_step).build(config.max_steps + 50)

    # dataloader (mixture-driven lane schedule)
    ctx = PlanContext(train_shards, seq_len=config.seq_len,
                      microbatch_size=config.microbatch_size,
                      vocab_size=tk.vocab_size, lane_schedule=timeline.lane_schedule)

    model = TinyLM(ctx.model_vocab, dim=config.dim, hidden=config.hidden,
                   max_pos=config.seq_len, seed=config.seed)
    optim = Adam(model.params, lr=config.lr)
    opus = Opus(accept_threshold=config.accept_threshold)

    sfx = ledger_suffix or (f".{branch_id}" if branch_id != "main" else "")
    cons = Ledger(os.path.join(ledger_dir, f"consumption{sfx}.jsonl"))
    learn = Ledger(os.path.join(ledger_dir, f"learning{sfx}.jsonl"))

    trainer = Trainer(model, optim, ctx, timeline, opus, cons, learn,
                      run_id=f"run-{config.hash()[7:19]}", seed=config.seed,
                      branch_id=branch_id)

    return Bundle(
        config=config, tokenizer=tk, train_shards=train_shards, eval_shard=eval_shard,
        manifests=manifests, admissions=admissions, eval_registry=reg,
        timeline=timeline, ctx=ctx, model=model, optim=optim, opus=opus,
        consumption=cons, learning=learn, trainer=trainer, lanes=ctx.lanes,
    )

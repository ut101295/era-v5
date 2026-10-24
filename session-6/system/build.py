"""Deterministic system assembly.

One place that wires the whole system together from config, so tests, the runner,
and `run_demo.py` all build the *identical* deterministic system. Everything here
is a pure function of the config -> reproducible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .corpus import load_corpus
from .decontam import decontaminate
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
    defer_band: float = 0.1
    stage_token_scale: float = 1.0   # shrink curriculum stage boundaries for short runs
    # lane -> packing policy (lanes not listed use concat_chop)
    packing_policy: Dict[str, str] = field(default_factory=dict)
    # lanes trained SFT-style: loss only on the response, never on the prompt
    sft_lanes: Tuple[str, ...] = ()
    # cut train-doc spans that share a 13-gram with the eval/validation sets
    decontaminate: bool = True

    def hash(self) -> str:
        return sha256_json(self.__dict__)


@dataclass
class Bundle:
    config: Config
    tokenizer: object
    train_shards: List[Shard]
    eval_shard: Shard
    validation_shard: Shard
    decontam_reports: Dict[str, Dict]
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
    opus_ledger: Ledger
    trainer: Trainer
    lanes: List[str] = field(default_factory=list)


def build_shards(tokenizer, config: Config):
    """Build one train shard per lane + one held-out eval/test shard."""
    train_shards, eval_shard, _, _ = build_all_shards(tokenizer, config)
    return train_shards, eval_shard


def build_all_shards(tokenizer, config: Config):
    """Held-out test + validation shards, then decontaminated train shards.

    The English docs are split three ways with no shared document: train
    [0, docs_per_lane), test [docs_per_lane, +3), validation [+3, +6). Held-out
    shards are built first so every train lane can be decontaminated against
    their n-grams before it is tokenized into a shard.

    Returns (train_shards, eval_shard, validation_shard, decontam_reports).
    """
    docs = load_corpus()
    en_docs = [d for d in docs if d.lane == "en"]
    eval_docs = en_docs[config.docs_per_lane: config.docs_per_lane + 3] or en_docs[:3]
    eval_shard = build_shard("eval-en-heldout", eval_docs, tokenizer,
                             split="test", canary=CANARY_STRING)
    k = config.docs_per_lane + 3
    val_docs = en_docs[k: k + 3] or en_docs[-3:]
    validation_shard = build_shard("val-en-heldout", val_docs, tokenizer, split="validation")

    heldout = EvalRegistry()
    heldout.register(eval_shard)
    heldout.register(validation_shard)

    train_shards: List[Shard] = []
    reports: Dict[str, Dict] = {}
    for lane in ("indic", "en", "code", "math"):
        lane_docs = [d for d in docs if d.lane == lane][: config.docs_per_lane]
        if config.decontaminate:
            lane_docs, reports[f"shard-{lane}"] = decontaminate(
                lane_docs, tokenizer, heldout._ngrams)
        train_shards.append(build_shard(f"shard-{lane}", lane_docs, tokenizer))
    return train_shards, eval_shard, validation_shard, reports


SFT_RESPONSE_MARKER = "Reasoning:"


def sft_learn_spans(shards: List[Shard], tokenizer, lanes) -> Dict[str, List]:
    """doc_id -> [(start, len)] covering only the response tokens.

    The math lane is prompt/response formatted ("Question: ...\nReasoning: ...").
    The response starts at the first token whose decoded prefix reaches the
    marker, so the question never bears loss.
    """
    spans: Dict[str, List] = {}
    for shard in shards:
        if shard.lane not in lanes:
            continue
        toks = shard.tokens.tolist()
        for span in shard.doc_spans:
            ids = toks[span.start:span.start + span.length]
            full = tokenizer.decode(ids)
            cut = full.find(SFT_RESPONSE_MARKER)
            if cut <= 0:
                continue
            start = next(k for k in range(len(ids) + 1)
                         if len(tokenizer.decode(ids[:k])) >= cut)
            spans[span.doc_id] = [(start, len(ids))]
    return spans


def build_system(artifacts_dir: str, config: Config | None = None,
                 branch_id: str = "main", ledger_suffix: str = "") -> Bundle:
    config = config or Config()
    tk = load_frozen_tokenizer()
    train_shards, eval_shard, validation_shard, decontam = build_all_shards(tk, config)

    # manifests + admission gate
    manifests, admissions = [], []
    ledger_dir = os.path.join(artifacts_dir, "ledgers")
    shard_dir = os.path.join(artifacts_dir, "shard_store")
    os.makedirs(shard_dir, exist_ok=True)
    from .shards import write_shard
    for s in train_shards:
        write_shard(s, shard_dir)
        rep = decontam.get(s.shard_id, {})
        m = build_manifest(
            s, cleaning_pipeline_hash=CLEANING_HASH,
            contamination_status="decontaminated" if rep.get("tokens_removed") else "clean",
            extra={"decontamination": {k: v for k, v in rep.items() if k != "per_doc"}}
            if rep else None)
        res = admit(m, tk.tokenizer_hash, shard_dir, {CLEANING_HASH})
        manifests.append(m)
        admissions.append({"shard_id": s.shard_id, **res.as_dict()})

    # eval firewall
    reg = EvalRegistry()
    reg.register(eval_shard)
    reg.register(validation_shard)

    # mixture timeline
    tokens_per_step = config.seq_len * config.microbatch_size
    timeline = default_timeline(tokens_per_step, config.stage_token_scale).build(
        config.max_steps + 50)

    # dataloader (mixture-driven lane schedule)
    ctx = PlanContext(train_shards, seq_len=config.seq_len,
                      microbatch_size=config.microbatch_size,
                      vocab_size=tk.vocab_size, lane_schedule=timeline.lane_schedule,
                      packing_policy_by_lane=config.packing_policy,
                      sft_agentic_spans=sft_learn_spans(train_shards, tk, config.sft_lanes))

    model = TinyLM(ctx.model_vocab, dim=config.dim, hidden=config.hidden,
                   max_pos=config.seq_len, seed=config.seed)
    optim = Adam(model.params, lr=config.lr)
    opus = Opus(accept_threshold=config.accept_threshold, defer_band=config.defer_band)

    sfx = ledger_suffix or (f".{branch_id}" if branch_id != "main" else "")
    cons = Ledger(os.path.join(ledger_dir, f"consumption{sfx}.jsonl"))
    learn = Ledger(os.path.join(ledger_dir, f"learning{sfx}.jsonl"))
    opus_led = Ledger(os.path.join(ledger_dir, f"opus{sfx}.jsonl"))

    trainer = Trainer(model, optim, ctx, timeline, opus, cons, learn,
                      run_id=f"run-{config.hash()[7:19]}", seed=config.seed,
                      branch_id=branch_id, opus_ledger=opus_led)

    return Bundle(
        config=config, tokenizer=tk, train_shards=train_shards, eval_shard=eval_shard,
        validation_shard=validation_shard, decontam_reports=decontam,
        manifests=manifests, admissions=admissions, eval_registry=reg,
        timeline=timeline, ctx=ctx, model=model, optim=optim, opus=opus,
        consumption=cons, learning=learn, opus_ledger=opus_led, trainer=trainer,
        lanes=ctx.lanes,
    )

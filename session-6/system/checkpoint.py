"""Checkpoints that bind MODEL state to DATA state.

"A checkpoint without a data position is incomplete" (assignment §14). A
checkpoint here serializes everything needed to resume the *experiment*
(research/01 §3, research/04): model weights, optimizer state, and the data
position -- which, because ordering is a pure function of (seed, step), is just
the step, the seed, the branch id, and the ledger offsets.

Stored as a `.npz` (arrays) plus a `.meta.json` (scalars) under a checkpoints/
directory. `checkpoint_id` is derived from the step.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from .model import Adam, TinyLM


def checkpoint_id(step: int) -> str:
    return f"ckpt-{step:06d}"


def save_checkpoint(
    ckpt_dir: str, model: TinyLM, optim: Adam, *, step: int, seed: int,
    branch_id: str, consumption_offset: int, learning_offset: int,
    tokens_consumed: int, code_config_hash: str, extra_state: Dict | None = None,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    cid = checkpoint_id(step)
    arrays: Dict[str, np.ndarray] = {}
    for k, v in model.state_dict().items():
        arrays[f"model.{k}"] = v
    opt_sd = optim.state_dict()
    for k, v in opt_sd["m"].items():
        arrays[f"opt.m.{k}"] = v
    for k, v in opt_sd["v"].items():
        arrays[f"opt.v.{k}"] = v
    np.savez(os.path.join(ckpt_dir, f"{cid}.npz"), **arrays)

    meta = {
        "checkpoint_id": cid, "step": step, "seed": seed, "branch_id": branch_id,
        "consumption_offset": consumption_offset, "learning_offset": learning_offset,
        "tokens_consumed": tokens_consumed, "opt_t": opt_sd["t"],
        "code_config_hash": code_config_hash,
        "model_dims": {"V": model.V, "d": model.d, "h": model.h, "max_pos": model.max_pos},
        "extra_state": extra_state or {},
    }
    with open(os.path.join(ckpt_dir, f"{cid}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return cid


def load_checkpoint(ckpt_dir: str, cid: str, model: TinyLM, optim: Adam) -> Dict:
    """Restore model + optimizer in place; return the meta dict."""
    with open(os.path.join(ckpt_dir, f"{cid}.meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    npz = np.load(os.path.join(ckpt_dir, f"{cid}.npz"))
    model_sd = {k[len("model."):]: npz[k] for k in npz.files if k.startswith("model.")}
    model.load_state_dict(model_sd)
    m = {k[len("opt.m."):]: npz[k] for k in npz.files if k.startswith("opt.m.")}
    v = {k[len("opt.v."):]: npz[k] for k in npz.files if k.startswith("opt.v.")}
    optim.load_state_dict({"t": meta["opt_t"], "m": m, "v": v})
    return meta


def latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    metas = [f for f in os.listdir(ckpt_dir) if f.endswith(".meta.json")]
    if not metas:
        return None
    return sorted(metas)[-1][: -len(".meta.json")]

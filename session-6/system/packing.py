"""Packing policies: fixed-length windows from variable-length documents.

Turns tokenized documents into fixed-length training sequences (research/02,
assignment §5). The Session 2 tokenizer has no special tokens beyond <unk>, so
we reserve two synthetic ids *above* the tokenizer vocab:

    EOS_ID = vocab_size        # document boundary marker
    PAD_ID = vocab_size + 1    # padding
    MODEL_VOCAB = vocab_size + 2

An "item" fed to a packer is a `PackItem`: a document's token ids plus its
`learn_spans` -- the token ranges that should bear loss. For plain pretraining
the whole document is a learn span; for SFT the response is; for agentic the
model's own tokens are (never the tool observations). Packers only arrange
tokens into windows and record `doc_ids`; the loss/attention/position machinery
lives in masks.py.

Policies (all deterministic given input order -> feeds replay in research/01):
  * pad_only             -- one item per window, pad the rest (worst utilization)
  * concat_chop          -- join with EOS, chop into fixed windows (pretraining)
  * best_fit             -- sort by length desc, first-fit-decreasing bin packing
  * structure_preserving -- best_fit but never split an item (SFT/agentic safe)
  * long_context         -- one item per (large) window; every unused slot is
                            high-value waste, so it is reported explicitly
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class PackItem:
    doc_id: str
    lane: str
    token_ids: List[int]
    # (start, end) half-open ranges within token_ids that bear loss.
    learn_spans: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.learn_spans:
            self.learn_spans = [(0, len(self.token_ids))]  # pretrain default


@dataclass
class PackedSequence:
    input_ids: np.ndarray     # int32 [seq_len], includes EOS/PAD sentinels
    doc_ids: np.ndarray       # int32 [seq_len]; -1 for PAD/EOS-boundary positions
    learn_mask: np.ndarray    # bool  [seq_len]; True where the token bears loss
    member_doc_ids: List[str] # human doc ids packed into this window (order kept)
    lanes: List[str]          # lane of each member doc
    pad_count: int


def specials(vocab_size: int):
    """Return (EOS_ID, PAD_ID, MODEL_VOCAB) for a given tokenizer vocab size."""
    return vocab_size, vocab_size + 1, vocab_size + 2


def _emit(chunk_ids, chunk_docnum, chunk_learn, member_ids, member_lanes,
          seq_len, pad_id):
    """Finalize one window: pad to seq_len and build arrays."""
    n = len(chunk_ids)
    pad = seq_len - n
    ids = chunk_ids + [pad_id] * pad
    docn = chunk_docnum + [-1] * pad
    learn = chunk_learn + [False] * pad
    return PackedSequence(
        input_ids=np.asarray(ids, dtype=np.int32),
        doc_ids=np.asarray(docn, dtype=np.int32),
        learn_mask=np.asarray(learn, dtype=bool),
        member_doc_ids=list(member_ids),
        lanes=list(member_lanes),
        pad_count=pad,
    )


def _item_streams(item: PackItem):
    """Yield (token_id, is_learn) for each token in an item."""
    learn_flags = [False] * len(item.token_ids)
    for (s, e) in item.learn_spans:
        for k in range(max(0, s), min(len(item.token_ids), e)):
            learn_flags[k] = True
    return learn_flags


def pack(items: List[PackItem], seq_len: int, policy: str,
         vocab_size: int) -> List[PackedSequence]:
    """Pack items into fixed-length sequences under the chosen policy."""
    eos_id, pad_id, _ = specials(vocab_size)

    if policy == "concat_chop":
        return _concat_chop(items, seq_len, eos_id, pad_id)
    if policy == "pad_only":
        return _one_per_window(items, seq_len, eos_id, pad_id, allow_truncate=True)
    if policy == "long_context":
        return _one_per_window(items, seq_len, eos_id, pad_id, allow_truncate=True)
    if policy in ("best_fit", "structure_preserving"):
        return _best_fit(items, seq_len, eos_id, pad_id)
    raise ValueError(f"unknown packing policy: {policy}")


def _concat_chop(items, seq_len, eos_id, pad_id):
    """Concatenate all items with EOS boundaries, chop into fixed windows."""
    ids: List[int] = []
    docn: List[int] = []
    learn: List[bool] = []
    members_by_docnum = {}
    for k, it in enumerate(items):
        flags = _item_streams(it)
        for tid, lf in zip(it.token_ids, flags):
            ids.append(tid); docn.append(k); learn.append(lf)
        ids.append(eos_id); docn.append(-1); learn.append(False)  # boundary
        members_by_docnum[k] = (it.doc_id, it.lane)

    out: List[PackedSequence] = []
    for start in range(0, len(ids), seq_len):
        c_ids = ids[start:start + seq_len]
        c_docn = docn[start:start + seq_len]
        c_learn = learn[start:start + seq_len]
        # remap doc numbers to a compact per-window namespace + collect members
        member_ids, member_lanes, remap = [], [], {}
        new_docn = []
        for dn in c_docn:
            if dn == -1:
                new_docn.append(-1); continue
            if dn not in remap:
                remap[dn] = len(member_ids)
                did, lane = members_by_docnum[dn]
                member_ids.append(did); member_lanes.append(lane)
            new_docn.append(remap[dn])
        out.append(_emit(c_ids, new_docn, c_learn, member_ids, member_lanes,
                         seq_len, pad_id))
    return out


def _one_per_window(items, seq_len, eos_id, pad_id, allow_truncate):
    out: List[PackedSequence] = []
    for it in items:
        flags = _item_streams(it)
        ids = list(it.token_ids)[: seq_len - 1]
        flags = flags[: seq_len - 1]
        ids.append(eos_id); flags.append(False)
        docn = [0] * (len(ids) - 1) + [-1]
        out.append(_emit(ids, docn, flags, [it.doc_id], [it.lane], seq_len, pad_id))
    return out


def _best_fit(items, seq_len, eos_id, pad_id):
    """First-fit-decreasing: sort by length desc, place each into the tightest
    open bin with room (research/02). Never splits an item, so it is also the
    structure-preserving policy. Items longer than seq_len-1 are truncated to a
    single window (reported via pad/utilization stats)."""
    order = sorted(range(len(items)), key=lambda i: len(items[i].token_ids), reverse=True)
    bins: List[dict] = []   # each: {ids, docn, learn, members, lanes, used}
    for idx in order:
        it = items[idx]
        need = min(len(it.token_ids), seq_len - 1) + 1  # +1 for EOS boundary
        placed = False
        for b in bins:
            if b["used"] + need <= seq_len:
                _append_item(b, it, eos_id, seq_len)
                placed = True
                break
        if not placed:
            b = {"ids": [], "docn": [], "learn": [], "members": [], "lanes": [], "used": 0}
            _append_item(b, it, eos_id, seq_len)
            bins.append(b)
    out = []
    # Deterministic output order: by first member doc id.
    for b in sorted(bins, key=lambda x: x["members"][0] if x["members"] else ""):
        out.append(_emit(b["ids"], b["docn"], b["learn"], b["members"], b["lanes"],
                         seq_len, pad_id))
    return out


def _append_item(b, it, eos_id, seq_len):
    flags = _item_streams(it)
    dn = len(b["members"])
    keep = min(len(it.token_ids), seq_len - 1)
    for tid, lf in zip(it.token_ids[:keep], flags[:keep]):
        b["ids"].append(tid); b["docn"].append(dn); b["learn"].append(lf)
    b["ids"].append(eos_id); b["docn"].append(-1); b["learn"].append(False)  # boundary
    b["members"].append(it.doc_id); b["lanes"].append(it.lane)
    b["used"] += keep + 1


def utilization(seq: PackedSequence) -> float:
    """Fraction of the window that bears loss (useful tokens / seq_len)."""
    return float(seq.learn_mask.sum()) / float(seq.input_ids.shape[0])

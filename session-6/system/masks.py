"""Loss masks, block-diagonal causal attention masks, and per-document position ids.

This is the correctness core of the batch layer (research/02, assignment §3-§4).
Given a `PackedSequence` (input_ids + doc_ids + learn_mask), it produces the four
tensors the training loop consumes, with the three invariants that matter:

  1. LOSS MASK -- loss only where the token bears loss AND the *next* token is in
     the same document. Padding, EOS boundaries, and (in SFT/agentic) context
     tokens never bear loss. Putting loss on cross-document or context tokens is
     the classic failure that teaches false continuations / environment
     hallucination.
  2. ATTENTION MASK -- block-diagonal causal: position i may attend to j iff
     j <= i AND doc_ids[i] == doc_ids[j] AND both are real tokens. No attention
     edge crosses a document boundary.
  3. POSITION IDS -- reset to 0 at every document boundary, so a second packed
     document starts at position 0 (correct RoPE phase), not at len(doc1).

The mode (pretrain | sft | agentic) is expressed entirely through `learn_mask`
upstream in packing.py, so this module is mode-agnostic: it just honors the
learn flags. That keeps "which tokens the model learns from" a single, testable
concept.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .packing import PackedSequence


@dataclass
class BatchTensors:
    input_ids: np.ndarray      # int32 [L]
    labels: np.ndarray         # int32 [L]; next-token targets, -100 where ignored
    loss_mask: np.ndarray      # bool  [L]; True where loss is applied
    attention_mask: np.ndarray # bool  [L, L]; True where attention is allowed
    position_ids: np.ndarray   # int32 [L]
    doc_ids: np.ndarray        # int32 [L]

    def n_loss_tokens(self) -> int:
        return int(self.loss_mask.sum())


IGNORE_INDEX = -100


def build_tensors(seq: PackedSequence, pad_id: int) -> BatchTensors:
    L = seq.input_ids.shape[0]
    ids = seq.input_ids
    doc = seq.doc_ids

    real = doc >= 0  # real tokens (not PAD, not EOS-boundary)

    # ---- position ids: restart the counter at each document boundary ----
    position_ids = np.zeros(L, dtype=np.int32)
    counter = 0
    current_doc = -2
    for i in range(L):
        if doc[i] < 0:
            position_ids[i] = 0
            current_doc = -2
            counter = 0
            continue
        if doc[i] != current_doc:
            current_doc = doc[i]
            counter = 0
        position_ids[i] = counter
        counter += 1

    # ---- attention mask: causal AND same-document AND both real ----
    idx = np.arange(L)
    causal = idx[None, :] <= idx[:, None]                 # j <= i
    same_doc = (doc[:, None] == doc[None, :]) & (doc[:, None] >= 0)
    both_real = real[:, None] & real[None, :]
    attention_mask = causal & same_doc & both_real
    # ensure every real token can at least attend to itself
    for i in range(L):
        if real[i]:
            attention_mask[i, i] = True

    # ---- labels + loss mask: next-token within the same document ----
    labels = np.full(L, IGNORE_INDEX, dtype=np.int32)
    loss_mask = np.zeros(L, dtype=bool)
    for i in range(L - 1):
        # predict token i+1 from position i, only if both real and same doc,
        # and the *predicted* token (i+1) is a learn token.
        if real[i] and real[i + 1] and doc[i] == doc[i + 1] and seq.learn_mask[i + 1]:
            labels[i] = ids[i + 1]
            loss_mask[i] = True

    return BatchTensors(
        input_ids=ids.copy(),
        labels=labels,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        doc_ids=doc.copy(),
    )


def assert_correct(bt: BatchTensors) -> None:
    """Runtime invariant checks (used by tests and, cheaply, in the demo)."""
    L = bt.input_ids.shape[0]
    # No attention edge crosses a document boundary.
    for i in range(L):
        for j in range(L):
            if bt.attention_mask[i, j]:
                assert bt.doc_ids[i] == bt.doc_ids[j] >= 0, "cross-doc attention edge"
                assert j <= i, "non-causal attention edge"
    # Position ids reset at boundaries: first real token of each doc has pos 0.
    seen = set()
    for i in range(L):
        d = int(bt.doc_ids[i])
        if d >= 0 and d not in seen:
            seen.add(d)
            assert bt.position_ids[i] == 0, "position id did not reset at doc start"
    # Loss only where labels are set.
    assert bool((bt.loss_mask == (bt.labels != IGNORE_INDEX)).all()), \
        "loss_mask and labels disagree"

"""The planned, deterministic dataloader: `plan(seed, step) -> Batch`.

The central design principle (research/01): **order is derived, not stored.** A
batch is a pure function of `(seed, global_step)` over a frozen set of admitted
shards. That is what makes crash-resume land on the exact next batch and replay
reproduce byte-identical batches.

Construction:
  1. From admitted shards, build a deterministic list of `PackItem`s per lane and
     pack each lane into fixed-length `PackedSequence`s once (deterministic).
  2. Derive a per-lane index permutation from the seed (Megatron-style shuffle
     index) -- a pure function of the seed, so no order is persisted.
  3. `plan(step)` chooses a lane (via the mixture schedule if provided, else a
     round-robin), then reads the next `microbatch_size` packed sequences from
     that lane using the permutation with epoch wraparound. The per-lane cursor
     is *recomputed* from `step` (by replaying the lane schedule), so `plan` needs
     no mutable state and is trivially pure.

Every batch carries a `batch_hash` over its tensors, which the ledger records and
resume/replay check against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .hashing import sha256_bytes
from .masks import BatchTensors, build_tensors
from .packing import PackItem, PackedSequence, pack, specials, utilization
from .shards import Shard


@dataclass
class LaneStream:
    lane: str
    sequences: List[PackedSequence]
    shard_ids: List[str]           # shard id contributing each member (parallel to sequences? no: per seq)
    seq_shard_ids: List[List[str]] # shard ids per packed sequence


@dataclass
class Batch:
    global_step: int
    seed: int
    lane: str
    shard_ids: List[str]
    member_doc_ids: List[str]
    tensors: List[BatchTensors]
    batch_hash: str
    n_loss_tokens: int
    n_pad: int
    n_positions: int               # total token positions (microbatch_size * seq_len)
    utilization: float
    seq_indices: List[int]         # which permuted sequence indices were consumed

    def as_ledger_span_ids(self) -> List[str]:
        return [f"{sid}#{did}" for sid, did in zip(self.shard_ids, self.member_doc_ids)]


# Lane schedule signature: (seed, step) -> lane name.
LaneSchedule = Callable[[int, int], str]


class PlanContext:
    """Immutable planning context: frozen shards + packed lane streams + config."""

    def __init__(
        self,
        shards: List[Shard],
        seq_len: int,
        microbatch_size: int,
        vocab_size: int,
        packing_policy_by_lane: Optional[Dict[str, str]] = None,
        lane_schedule: Optional[LaneSchedule] = None,
        sft_agentic_spans: Optional[Dict[str, List]] = None,
    ):
        self.seq_len = seq_len
        self.microbatch_size = microbatch_size
        self.vocab_size = vocab_size
        _, self.pad_id, self.model_vocab = specials(vocab_size)
        self.packing_policy_by_lane = packing_policy_by_lane or {}
        self.lane_schedule = lane_schedule  # None -> round robin (M2 default)
        self.tokenizer_hash = shards[0].tokenizer_hash if shards else "sha256:unknown"

        # Build per-lane packed streams deterministically.
        self.lanes: List[str] = sorted({s.lane for s in shards})
        self.streams: Dict[str, LaneStream] = {}
        for lane in self.lanes:
            items: List[PackItem] = []
            item_shard: List[str] = []
            for shard in sorted([s for s in shards if s.lane == lane],
                                key=lambda s: s.shard_id):
                toks = shard.tokens.tolist()
                for span in shard.doc_spans:
                    ids = toks[span.start:span.start + span.length]
                    if not ids:
                        continue
                    items.append(PackItem(doc_id=span.doc_id, lane=lane, token_ids=ids))
                    item_shard.append(shard.shard_id)
            policy = self.packing_policy_by_lane.get(lane, "concat_chop")
            seqs = pack(items, seq_len, policy, vocab_size)
            # Map each packed sequence back to the shard ids of its members.
            doc_to_shard = {it.doc_id: sid for it, sid in zip(items, item_shard)}
            seq_shard_ids = [
                sorted({doc_to_shard.get(d, "?") for d in s.member_doc_ids}) for s in seqs
            ]
            self.streams[lane] = LaneStream(
                lane=lane, sequences=seqs,
                shard_ids=sorted({sid for sid in item_shard}),
                seq_shard_ids=seq_shard_ids,
            )

    # ------------------------------------------------------------- ordering
    def _permutation(self, lane: str) -> np.ndarray:
        """Per-lane index permutation, a pure function of (seed, lane)."""
        n = len(self.streams[lane].sequences)
        # seed folded with a stable per-lane salt for independent shuffles.
        salt = int(sha256_bytes(lane.encode())[7:15], 16)
        rng = np.random.default_rng(self._seed ^ salt)
        return rng.permutation(n) if n > 0 else np.array([], dtype=int)

    def _lane_for_step(self, step: int) -> str:
        if self.lane_schedule is not None:
            return self.lane_schedule(self._seed, step)
        return self.lanes[step % len(self.lanes)]  # M2 round-robin

    def _lane_cursor(self, lane: str, step: int) -> int:
        """How many microbatches of `lane` were consumed strictly before `step`.

        Recomputed by replaying the (pure) lane schedule -> no stored state.
        """
        return sum(1 for s in range(step) if self._lane_for_step(s) == lane)

    # ------------------------------------------------------------- plan
    def plan(self, seed: int, step: int) -> Batch:
        self._seed = seed
        lane = self._lane_for_step(step)
        stream = self.streams[lane]
        n = len(stream.sequences)
        assert n > 0, f"lane {lane} has no packed sequences"
        perm = self._permutation(lane)
        cursor = self._lane_cursor(lane, step)

        seq_indices: List[int] = []
        tensors: List[BatchTensors] = []
        member_docs: List[str] = []
        shard_ids: set = set()
        n_loss = n_pad = 0
        for k in range(self.microbatch_size):
            pos = (cursor * self.microbatch_size + k) % n   # epoch wraparound
            si = int(perm[pos])
            seq = stream.sequences[si]
            bt = build_tensors(seq, self.pad_id)
            seq_indices.append(si)
            tensors.append(bt)
            member_docs.extend(seq.member_doc_ids)
            shard_ids.update(stream.seq_shard_ids[si])
            n_loss += bt.n_loss_tokens()
            n_pad += seq.pad_count

        n_positions = self.microbatch_size * self.seq_len
        batch_hash = self._hash_batch(step, lane, tensors)
        return Batch(
            global_step=step, seed=seed, lane=lane,
            shard_ids=sorted(shard_ids), member_doc_ids=member_docs,
            tensors=tensors, batch_hash=batch_hash,
            n_loss_tokens=n_loss, n_pad=n_pad, n_positions=n_positions,
            utilization=n_loss / n_positions if n_positions else 0.0,
            seq_indices=seq_indices,
        )

    def _hash_batch(self, step: int, lane: str, tensors: List[BatchTensors]) -> str:
        h = bytearray()
        h += step.to_bytes(8, "little", signed=False)
        h += lane.encode()
        for bt in tensors:
            h += bt.input_ids.astype("<i4").tobytes()
            h += bt.labels.astype("<i4").tobytes()
            h += bt.loss_mask.astype(np.uint8).tobytes()
            h += bt.position_ids.astype("<i4").tobytes()
        return sha256_bytes(bytes(h))

    # convenience
    def num_lanes(self) -> int:
        return len(self.lanes)

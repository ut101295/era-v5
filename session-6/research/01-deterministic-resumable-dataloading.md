# Dossier 01 — Deterministic, Resumable, Replayable Dataloading

> **The question this dossier answers:** how do real large-scale training systems make
> "the batch at global step *N*" a **pure function** of a seed and an index — so that a
> crashed run resumes on the *exact next* batch (none skipped, none repeated), an old
> checkpoint can **replay** the identical historical stream, and a run can **fork** into a
> new, explicitly-different stream? This is the operational heart of Session 6 §14
> ("Resume, Replay, and Fork") and the reproducibility rule in §9
> (`experiment = model checkpoint + optimizer state + data stream + code/config`).

All claims below are grounded in primary docs (fetched during research); URLs are inline.

---

## 1. The core principle: ordering is a pure function of (seed, index)

Every serious system converges on the same idea: **do not remember the order, *derive* it.**
If sample order is a deterministic function of a fixed random seed plus a global counter,
then you never have to persist the order — you only persist *where you are in it* (a small
integer or two), and the loader regenerates the rest.

### Megatron-Core — three index arrays from one seed

Megatron-Core's `GPTDataset` builds a fully deterministic sample ordering from a **single
random seed `R`** via three 1-D index arrays:

- **Document index** — the list of documents, *shuffled according to `R`*.
- **Sample index** — maps each training sample to a (document, offset) span in the
  concatenated token stream.
- **Shuffle index** — a length-`N` array mapping position `k → j`, *shuffled according to
  `R`*; retrieval "uses the shuffle index to get the index `j` into the sample index."

Because all three are derived from `R`, sample `k` is reproducible forever: seed + `k` →
shuffle index → sample index → document index → exact token span. The built indices are
**cached to disk keyed by a hash generated in the dataset constructor**, so the same
configuration deterministically reuses the same ordering, and indices are "built/cached
sequentially on one process rank and subsequently loaded in parallel on other ranks."
Source: <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/readme.md>

**Blending** multiple datasets (our "mixture") is also deterministic: `BlendedDataset` is
parameterized by per-dataset weights `W` and total size `S`; at each step it "draw[s] a
single sample from the dataset which has the greatest sampling error" (a deterministic
largest-remainder / error-diffusion rule), producing a dataset-index array `Da_idx` of
length `S`, cached by its own constructor hash. This is exactly how you turn "32% web /
22% code / …" into a concrete, replayable per-step lane assignment. Same source.

> **Takeaway for our build:** the planned stream is `(seed, global_step) → lane → shard →
> sample span`, computed, not stored. This is the baseline mental model the assignment's
> §17 references call out ("Megatron … indexed GPT datasets through document, sample and
> shuffle indices … the right baseline mental model for planned sample lookup").

### PyTorch `StatefulDataLoader` (torchdata) — snapshot the position

A drop-in `DataLoader` replacement exposing `state_dict()` / `load_state_dict()` for
**mid-epoch** checkpointing. By default the state "includes the number of batches yielded
and uses this to naively fast-forward the sampler"; if the sampler/dataset define their own
`state_dict` methods (e.g. returning `{"i": self.i, "rng": self.g.get_state()}` and
`torch.get_rng_state()`), the loader calls them so **RNG state travels with the checkpoint.**
Two operational constraints matter for us:

- It "handles aggregation and distribution of state across multiprocess workers (**but not
  across ranks**)" — i.e. each rank checkpoints its own loader.
- Resuming an iterable dataset "requires StatefulDataLoader to have the same `num_workers`
  as those of the provided `state_dict`."
- `snapshot_every_n_steps` (default 1) tunes how often worker state is transferred back;
  best practice is to align it to the **training checkpoint frequency** to bound overhead
  — i.e. **consumption records are cheapest when co-timed with checkpoints.**

Source: <https://github.com/meta-pytorch/data/blob/main/torchdata/stateful_dataloader/README.md>
(the `snapshot_every_n_steps` note is corroborated by the API page, flagged as secondary).

### Hugging Face `IterableDataset` — (shard id, example index)

HF makes a stream resumable by storing **the current shard being read plus the example
index within that shard** in its `state_dict`. On resume "the dataset skips all the shards
that were previously read to restart from the current shard. Then it reads the shard and
skips examples until it reaches the exact example from the checkpoint." Resumption is exact
**except when `.shuffle()` is used** — shuffle-buffer contents are lost on resume and the
buffer is refilled with new data, so the position is only approximate under buffered shuffle.
In distributed streaming, a **fixed shuffle seed is required** so every node computes the
same shuffled shard list and knows which shards to skip.
Source: <https://huggingface.co/docs/datasets/stream>

> **Design consequence we adopt:** avoid *buffered* shuffle for the parts that must replay
> exactly. Use **index-permutation shuffle** (like Megatron) where the permutation is a
> pure function of the seed, so position is a single integer and resume is exact.

### MosaicML StreamingDataset — "elastic determinism"

The strongest guarantee in the wild: sample order is **identical regardless of the number
of GPUs, nodes, or CPU workers.** Their Figure 3 shows "the exact same loss curve" across
1/8/16/32/64 GPUs, and a documented drill trains 300 steps on 8 GPUs (checkpoint every 100
batches), **resumes on 32 GPUs from batch-100, then on 128 GPUs from batch-200, and the loss
curves match within numerics.** Fast deterministic resumption requires their
`StreamingDataLoader` object whose `state_dict`/`load_state_dict` are saved with the
checkpoint; iteration then resumes ("will start from batch 5 now") rather than restarting —
which "can save thousands of dollars in egress fees and idle GPU compute."
Sources: <https://www.databricks.com/blog/mosaicml-streamingdataset> ·
<https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/fast_resumption.html>

> This is the assignment's §17 reference ("Mosaic StreamingDataset documents mid-epoch
> resumable streaming and deterministic sample ordering for practical recovery"), and the
> gold standard our tiny system imitates: **decouple order from device topology.**

---

## 2. The minimal state that must be saved to resume *exactly*

Cross-referencing all four systems, the state needed to reconstruct "which samples were
consumed" is tiny:

| System | Persisted resume state |
| --- | --- |
| Megatron | `seed R` + `global_step` (index into shuffle→sample→doc arrays) |
| torchdata StatefulDataLoader | `#batches yielded` + optional sampler `{i, rng}` |
| HF IterableDataset | `current shard id` + `example index in shard` |
| Mosaic Streaming | `StreamingDataLoader.state_dict()` (position, epoch, seed) |

The unifying insight: **`(seed, step)` is enough** when order is a pure function of the
seed. Everything else in a checkpoint (RNG, epoch) exists only to cover the non-pure parts
(buffered shuffle, augmentation). If we make our loader index-pure, our resume state is just
two integers plus the seed — and it is trivially verifiable.

---

## 3. Reproducibility of the *whole experiment*, not just the loader

Two independent secondary sources state the same rule the assignment adopts verbatim in §9:
a reproducible run must record **code (git commit) + data (version + hash) + config
(hyperparameters) + metrics + environment**, and "if you can't answer all five … you have a
versioning problem." Resuming *exactly* "requires saving optimizer and scheduler state
alongside model weights, not weights alone."
(<https://www.techsaas.cloud/blog/ml-model-versioning-registry/>,
<https://medium.com/hub-by-littlebigcode/mlops-effortlessly-track-your-model-experiments-with-dvc-and-mlflow-db650cffab22> — both secondary/blog.)

Tooling that operationalizes "checkpoint → data version + code commit + config":

- **DVC `repro`** records MD5 hashes of each stage's deps/outputs in `dvc.lock` and re-runs
  only stages whose recorded hashes changed; committing `dvc.yaml`+`dvc.lock` to Git binds a
  git commit to a specific data/config/pipeline state.
  <https://doc.dvc.org/command-reference/repro>
- **Weights & Biases** builds a DAG lineage by marking artifacts as run inputs
  (`use_artifact()`) and outputs (`log_artifact()`), each auto-versioned, enabling a model to
  be traced back to the exact data/code that produced it.
  <https://docs.wandb.ai/guides/artifacts/explore-and-traverse-an-artifact-graph/>

> **Design consequence:** our checkpoint must serialize **model + optimizer + scheduler +
> RNG + dataloader position + ledger offset + code/config hash.** "A checkpoint without a
> data position is incomplete" (assignment §14) is exactly the StreamingDataLoader lesson.

---

## 4. Minimal implementation recipe (what we will build)

A pure-Python, numpy-only, **index-based planned dataloader**:

1. **Frozen inputs:** a set of immutable shards (Dossier 03), a mixture schedule
   (`stage → per-lane weights + floors`), and a single integer `seed`.
2. **Per-step plan (pure function):**
   ```
   plan(seed, step):
       stage      = stage_for_token_position(step * global_batch_tokens)
       lane       = blend_pick(stage.weights, stage.floors, step, seed)   # Megatron error-diffusion
       shard      = shard_permutation(lane, seed)[ counter[lane] ]
       sample_ids = pack_next(shard, seq_len)                             # deterministic packer cursor
       return Batch(step, lane, shard, sample_ids, ...)
   ```
   No global state beyond per-lane counters, which are themselves derivable from `step`.
3. **Checkpoint = {model, optim, sched, rng, loader_counters, ledger_offset, code_cfg_hash}.**
4. **Resume:** load checkpoint → set counters → `plan(seed, step+1)` **must byte-for-byte
   equal** the batch recorded in the ledger at `step+1`. This is the graded invariant
   `[PASS] resume_next_batch_matched`.
5. **Replay:** ignore live counters; feed steps straight from the recorded ledger offsets,
   re-`plan()` each, and assert `batch_hash` matches the original — `[PASS] replay_hash_matched`.
6. **Fork:** restore an old checkpoint, mint a **new branch id + new seed** (or same seed,
   different mixture), and record the exact divergence step in the ledger. Loss curves are
   then attributable to the *declared* data difference, not a silent one.
7. **Elastic determinism (optional but cheap at toy scale):** because order is a function of
   `(seed, step)` and not of worker/rank layout, splitting steps across N simulated workers
   (round-robin by `step % N`) yields the identical global sequence — we can demonstrate the
   Mosaic property in miniature.

**Determinism hygiene** (from the harness docs): fix all seeds, avoid buffered shuffle on
the replayable path, keep `num_workers` fixed across resume, and set `PYTHONHASHSEED=0` if
any Python `hash()` enters an ordering decision (the EleutherAI decontaminator requires this
too — see Dossier 05).

---

## 5. Sources

- Megatron-Core datasets readme — <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/readme.md>
- torchdata StatefulDataLoader — <https://github.com/meta-pytorch/data/blob/main/torchdata/stateful_dataloader/README.md>
- HF `datasets` streaming/resumption — <https://huggingface.co/docs/datasets/stream>
- MosaicML StreamingDataset blog — <https://www.databricks.com/blog/mosaicml-streamingdataset>
- Mosaic Streaming fast resumption — <https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/fast_resumption.html>
- DVC `repro` — <https://doc.dvc.org/command-reference/repro>
- W&B artifact lineage — <https://docs.wandb.ai/guides/artifacts/explore-and-traverse-an-artifact-graph/>
- (secondary) reproducibility equation / optimizer-state resume — techsaas.cloud, littlebigcode Medium

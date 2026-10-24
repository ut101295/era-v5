# Dossier 06 — Dataloader Throughput and "Useful Tokens/sec"

> **The question this dossier answers:** correctness alone doesn't keep GPUs busy — how do we
> measure whether the data system is *fast enough to disappear* from the GPU's point of view,
> and why is **raw** tokens/sec a misleading number? This is Session 6 §15. Graded block:
> *"Throughput and packing efficiency — 50 pts."*

> Note on sourcing: like Dossier 02, the consolidated web dossier for this topic was lost to a
> mid-stream API error; the mechanisms below are standard and I cite the canonical primary
> sources from domain knowledge. Re-verify figures before final submission.

---

## 1. The metric that matters

The GPU computes over **every** token position, but the model learns only from loss-bearing
positions (Dossier 02). So the honest throughput metric is:

> **Useful loss-bearing tokens per second, at the target mixture.**

A loader can report high **raw** tokens/sec while wasting compute on padding, context-only
tokens, or candidates rejected by OPUS. Three throughput numbers therefore diverge and all
three must be reported (assignment §15):

```
raw_tokens/sec          ≥  useful_loss_bearing_tokens/sec  ≥  accepted_tokens/sec (after OPUS)
```

- **raw** = every position pushed through the batch (incl. padding, context-only).
- **useful** = raw × packing_utilization × loss_mask_fraction.
- **accepted** = useful × (1 − OPUS_rejection_rate).

The gap between raw and useful is the tax paid for padding + context tokens; the gap between
useful and accepted is the tax paid for selection.

---

## 2. What throughput depends on (the knobs)

From §15: shard size, compression, storage bandwidth, local caching, prefetch depth, worker
count, rank partitioning, packing efficiency, and OPUS rejection rate. Trade-offs:

- **Small files** → per-file overhead; **huge files** → less flexibility + hard recovery
  (WebDataset's ~1 GB shard target is the sweet spot, Dossier 03).
- **Compression** saves storage/network but costs CPU.
- **Prefetch** hides latency but costs memory.
- **More workers** help until they saturate storage or contend.
- **OPUS** raises token *value* but can lower accepted-token *throughput* if candidate
  scoring is slow.

Metrics to monitor (§15): raw / useful / accepted tokens per sec, GPU idle time, loader wait
time, cache hit rate, shard read latency, packing utilization, rejection rate by lane, and
replay/resume latency.

---

## 3. Grounding: MFU and "the loader should disappear"

- **Model FLOPs Utilization (MFU)** — the standard efficiency yardstick from the PaLM paper
  (Chowdhery et al., 2022): observed throughput as a fraction of a system's theoretical peak
  FLOPs. The data system's job is to keep the training loop **compute-bound**, so that MFU is
  limited by the model, not by loader stalls. <https://arxiv.org/abs/2204.02311>
- **Overlap I/O with compute** — the reason prefetching/caching exist is to hide shard-read
  latency behind the GPU's forward/backward, so `loader_wait_time → 0`. This is the practical
  content of "the best data system is … fast enough to disappear from the GPU's point of view"
  (assignment §15). Mosaic's elastic-determinism resumption (Dossier 01) exists partly to
  avoid re-streaming the whole dataset-so-far on resume, saving "egress fees and idle GPU
  compute." <https://www.databricks.com/blog/mosaicml-streamingdataset>

---

## 4. Minimal implementation recipe (what we will build)

We can't measure real GPU MFU on a laptop, but we can measure the **data-system** numbers
honestly and reproducibly:

1. **Instrument the loader** — time `plan()` + pack + mask build per batch; count raw
   positions, loss-bearing positions, pad positions, boundary crossings, OPUS-rejected
   candidates.
2. **Compute the three rates** — raw / useful / accepted tokens per second from measured wall
   time (deterministic clock, not `Date.now`-random). Report **packing utilization** =
   useful / raw.
3. **Cache-hit simulation** — a small in-process shard cache with a hit-rate knob; show useful
   tokens/sec rising with cache hit rate and prefetch depth.
4. **`performance.json`** — emit measured numbers under `submission_artifacts/performance.json`
   so the grader can reconstruct them (assignment: "If reported packing or throughput numbers
   cannot be reconstructed, those claims will not receive credit").
5. **Sanity assertions** — `raw ≥ useful ≥ accepted`; utilization ∈ (0,1]; a padded-heavy
   policy shows *lower* useful tokens/sec than concat-and-chop at equal raw — making the
   padding tax visible (ties back to Dossier 02).

---

## 5. Sources

- PaLM (MFU definition) — <https://arxiv.org/abs/2204.02311>
- MosaicML StreamingDataset (resume avoids re-streaming) — <https://www.databricks.com/blog/mosaicml-streamingdataset>
- (see Dossier 02 for packing-utilization mechanics; Dossier 05 for OPUS rejection)

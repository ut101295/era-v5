# Dossier 04 — The Consumption/Learning Ledger, and Replay / Fork / Audit

> **The question this dossier answers:** how do we keep an **append-only record of what the
> training run actually consumed** (and what it learned) so the run is inspectable even when
> "the world around the run is messy" — workers restart, ranks retry, checkpoints are
> restored, files vanish? And how does that log make **resume / replay / fork / audit**
> well-defined operations rather than hopeful ones? This is Session 6 §8–§12 and §14.
> Graded blocks: *"Consumption and learning ledgers — 150 pts"* and *"Checkpoint, crash,
> resume, replay and fork — 150 pts."*

All claims below are grounded in primary docs fetched during research; URLs inline.

---

## 1. The pattern: an append-only, immutable, ordered commit log

Three production data systems independently converge on **one primitive** — an append-only,
immutable, ordered log/tree of commits, where new state is *added*, never mutated in place —
which is exactly what enables time-travel, replay, and audit:

- **LakeFS ("Git for data"):** commits are "immutable 'checkpoints' containing all the
  contents of a repository at a given point"; a branch "is simply a pointer to a commit along
  with a set of uncommitted changes," and branching is **zero-copy**. A merge "is a new
  commit, with the destination as the first parent and the source as the second," found via
  the nearest common ancestor. Atomicity is optimistic (**Compare-And-Set**), and committed
  metadata is a **content-addressable two-layer Merkle tree** (ranges + meta-range) so a new
  commit "reuses every range except the one containing the modified keys."
  <https://docs.lakefs.io/concepts/internals/>
- **Apache Iceberg:** a **snapshot** is "the state of a table at some point in time, including
  the set of all data files"; exactly one **manifest list** per snapshot; each snapshot has a
  unique id and a **parent-snapshot-id** forming a lineage chain; a monotonic
  **sequence-number** orders changes; "an atomic swap of one table metadata file for another
  provides the basis for serializable isolation," with optimistic-commit + retry.
  <https://raw.githubusercontent.com/apache/iceberg/main/format/spec.md>
- **Delta Lake:** the transaction log is JSON files in `_delta_log`, "named using the next
  available version number, zero-padded to 20 digits"; each commit file is "the unit of
  atomicity," an "atomic set of actions"; state is "computed by replaying the events committed
  … in ascending order by commit version" (latest `metaData`/`protocol` wins); **checkpoints**
  are Parquet snapshots of the full replay, found via `_delta_log/_last_checkpoint`; and
  **time travel** (`VERSION AS OF` / `TIMESTAMP AS OF`) supports "debugging or auditing,
  especially in regulated industries," re-creating past reports, and snapshot isolation.
  <https://raw.githubusercontent.com/delta-io/delta/master/PROTOCOL.md> ·
  <https://docs.delta.io/latest/delta-batch.html>

The software-architecture name for this is **Event Sourcing**: "all changes to application
state are stored as a sequence of events," so "we can discard the application state
completely and rebuild it by re-running the events from the event log," the log "can be the
official record," and it "is a purely additive structure that requires minimal locking."
<https://martinfowler.com/eaaDev/EventSourcing.html>

> **What we adopt:** the **consumption ledger is an append-only event log**; the "current
> state" of the run (which shards trained which interval) is *derived by replay*, and periodic
> **checkpoints** are just materialized snapshots of that replay (the Delta idea) bound to a
> `ledger_offset`. Forking mints a new **branch id** with a **parent pointer** (the
> Iceberg/LakeFS idea).

---

## 2. The consumption ledger — one event per consumed batch

The assignment §8 lists the fields; grounded rationale from Dossier 01 (only a tiny tuple is
needed to *reconstruct* a batch) and §3's lineage. Per consumed microbatch:

```json
{
  "run_id": "uuid", "branch_id": "main", "global_step": 843219,
  "checkpoint_id": "ckpt-000843000", "rank": 0, "microbatch_id": 3,
  "packed_sample_ids": ["shard7#doc12@0:512", "shard7#doc55@0:480"],
  "shard_ids": ["shard-000007"], "token_span_ids": ["…"],
  "batch_hash": "sha256:…",              // over input_ids+labels+masks+position_ids
  "loss_mask_hash": "sha256:…",
  "attn_pos_policy": "block_diag_causal / per_doc_reset",
  "mixture_lane": "indic", "curriculum_stage": "reasoning-heavy-midtrain",
  "tokenizer_version": "sha256:…", "dataloader_version": "v1",
  "opus_decision_id": "cand-00042",     // where applicable (Dossier 05/§10)
  "seed": 1234                          // makes the batch a pure function → replayable
}
```

This log is "the run's memory." It lets us **reconstruct which tokens contributed to a
checkpoint, resume from a ledger offset, and locate the data that shaped a specific interval
of training** (assignment §8). Because the batch is a pure function of `(seed, step)`
(Dossier 01), the `batch_hash` in the ledger is the ground truth that resume/replay check
against.

---

## 3. Why the ledger matters (the confounded-experiment argument, §9)

The assignment's §9 makes the case precisely: restore an old checkpoint two ways.

- **No ledger binding:** the model starts from old weights but "the data stream has silently
  changed" (new seed / worker count / shard set). If the new strategy looks better, you can't
  tell whether the strategy helped or the model just *saw different data*. The experiment is
  **confounded.**
- **Ledger-bound:** the run either **replays** the historical segment or **forks** into a new
  branch where "every data difference is explicit." Now the comparison has a defined model
  state *and* a defined data stream.

The rule (which Dossier 01 §3 corroborates from independent MLOps sources):
```
experiment = model checkpoint + optimizer state + data stream + code/config
```
"If any one of these changes silently, the comparison is weakened." This is the entire
justification for building a data ledger at all.

---

## 4. Four operations, made well-defined (§14)

| Mode | Definition | Our implementation |
| --- | --- | --- |
| **Resume** | continue same run from latest checkpoint + latest ledger offset | load ckpt → `plan(seed, offset+1)` → assert `batch_hash == ledger[offset+1]` (`[PASS] resume_next_batch_matched`) |
| **Replay** | restore older checkpoint, feed the **same** historical stream | iterate recorded offsets, re-`plan()`, assert hashes match (`[PASS] replay_hash_matched`) |
| **Fork** | restore a checkpoint, start a **new** data branch | mint `branch_id` + parent pointer + divergence step; new events tagged with the new branch |
| **Audit** | reconstruct data that trained a checkpoint/interval | query ledger by `global_step` range → shards, spans, lanes, OPUS decisions ("which shards influenced the model between 5.4B–5.6B tokens?") |

"Checkpointing therefore binds model state to data state. A checkpoint without a data
position is incomplete." (assignment §14; = the Mosaic StreamingDataLoader lesson, Dossier 01.)

---

## 5. The learning ledger and per-token perplexity (§11–§12)

The **consumption ledger** records what the model *saw*; the **learning ledger** attaches the
*outcome* back to the data — a two-way link that turns the data system into a **measuring
instrument** and produces the training signal for the next corpus version ("V5 teaches V6").

- **Per-token signal (§11):** for each loss-bearing token, `loss_t = -log p(true tokenₜ)`,
  `ppl_t = exp(loss_t)`. Storing this is expensive, so use tiers: **full traces** for
  proxy/debug/suspicious shards; **compressed/quantized** losses for selected intervals;
  **aggregated stats** (by shard / language / token-id / position bucket / lane / phase) for
  the full run. The value: a shard is "not uniformly useful or useless" — the *pattern of
  surprise inside it* is the map for future data collection (e.g. an Indic conjunct or a
  tool-call argument that stays high-perplexity).
- **Per-shard report card (§12):** average token loss, high-perplexity clusters, **loss delta
  before/after exposure**, gradient norm/alignment, OPUS score, repeated-pass effect, model
  phase, and a **useful / neutral / harmful** classification for future planning. This is the
  feedback that says "high OPUS score + strong early loss drop ⇒ foundation data" vs "high
  OPUS score + little loss drop ⇒ proxy overvaluing it."

> **What we build (toy but real):** because we train a real tiny model (Dossier 06 in the
> plan), we can compute *genuine* per-token cross-entropy and perplexity, write a real
> heatmap-able trace for one batch, and produce a real per-shard report card with measured
> loss deltas — not simulated numbers (graded: "Learning trace — loss linked to source data").

---

## 6. Minimal implementation recipe

1. **`Ledger`** — append-only JSONL under `submission_artifacts/ledgers/`; one file per branch;
   an event per consumed microbatch (schema §2). `offset` = line count.
2. **`checkpoint(step)`** — serialize `{model, optim, sched, rng, loader_counters,
   ledger_offset, code_cfg_hash, branch_id}` under `checkpoints/`. This is the Delta-style
   materialized snapshot of the replay.
3. **`resume()` / `replay(interval)` / `fork(at_step)` / `audit(step_range)`** — the four ops
   in §4, each emitting explicit `run.log` lines and PASS markers.
4. **Learning ledger** — after each step, append measured per-shard aggregates and (for a
   chosen batch) a full per-token `{token_id, decoded_preview, ppl_t, lane, shard, boundary}`
   trace.
5. **The deliberate crash** — kill mid-run, then `resume()` and assert the next batch equals
   the expected one; `replay()` an earlier interval and assert hashes match. These two proofs
   are the spine of the demo (assignment §16).

---

## 7. Sources

- LakeFS internals — <https://docs.lakefs.io/concepts/internals/>
- Iceberg table spec — <https://raw.githubusercontent.com/apache/iceberg/main/format/spec.md> (also <https://iceberg.apache.org/spec/>)
- Delta transaction-log protocol — <https://raw.githubusercontent.com/delta-io/delta/master/PROTOCOL.md>
- Delta time travel — <https://docs.delta.io/latest/delta-batch.html>
- Event Sourcing (Fowler) — <https://martinfowler.com/eaaDev/EventSourcing.html>
- (see Dossier 01 for the reproducibility-equation and StatefulDataLoader sources)

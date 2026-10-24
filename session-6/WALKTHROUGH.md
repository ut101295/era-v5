# Session 6 — Building the Training Dataset · Walkthrough

> This document is the running narrative of how I read the Session 6 assignment, what it
> actually asks for, how I researched it, and how the planning artifacts were assembled. It
> is written to be read top-to-bottom as one continuous story — nothing is abbreviated, so
> the "flow" of the work is preserved and auditable end to end.
>
> The **evidence** of the research lives in [`research/`](research/) (six grounded dossiers,
> each with inline source URLs). The **build design** lives in
> [`plan/training-data-execution-system.md`](plan/training-data-execution-system.md). This
> document is the narrative that ties them together and explains *why* each artifact exists.
>
> As of this writing we are at the **end of the planning phase**: the assignment is fully
> understood, the research is done, and the implementation is specified module-by-module and
> mapped to the 1,000-point rubric. Code has not been written yet — that begins once the four
> open questions at the end of the plan are answered.

---

## Part 0 — Where this session sits in the arc

Sessions 1–5 each signed a *contract*, and Session 6 is the session that has to **execute all
of them at once**. The assignment says this explicitly, and it's worth internalizing because
every design decision downstream is a consequence of one of these contracts:

- **Session 1 — the training contract.** The model predicts the next token; loss applies only
  to the *intended* tokens; attention operates within a bounded context window. → This becomes
  our **loss masks** and **attention masks** (Dossier 02).
- **Session 2 — the tokenizer contract.** The same raw text must always produce the same token
  IDs, with canonical special tokens and Indic-safe normalization. → This becomes our
  **frozen tokenizer + tokenizer hash** (Dossier 03). We literally reuse Session 2's
  `tokenizer.json`.
- **Session 3 — the source contract.** Every source carries provenance, license, held-out
  status, capability tags, token accounting. → This becomes our **manifest fields**.
- **Session 4 — the admission contract.** Only cleaned, deduplicated, PII-screened,
  language-validated, contamination-scanned data may enter. → This becomes our **admission
  gate** and **cleaning-pipeline hash**. We reuse Session 4's cleaned Assamese corpus as real
  Indic supply.
- **Session 5 — the mixture contract.** Curriculum stages, protected floors, OPUS selection,
  annealing reserves, long-context timing, reasoning-length bands. → This becomes our
  **mixture timeline compiler** and **OPUS audit trail**.

The single sentence the whole session pivots on:

> **The dataloader is downstream of every design choice made so far.**

Session 5 produced a *recipe*. Session 6 turns that recipe into "the actual data stream
consumed by the training loop" — token windows, loss masks, attention masks, position IDs,
mixture tags, packed microbatches — and, crucially, keeps that stream **controlled,
inspectable, and replayable** across many workers, GPUs, restarts, and checkpoint boundaries.

---

## Part 1 — What the assignment actually asks (decomposed)

Unlike Session 5 (which was a *written spec*), Session 6 is a **working software system**
worth **1,000 points**, graded by actually running it. I read the whole lesson (16 numbered
sections + submission/eval rules) and distilled the deliverable to this:

> Build a *small but complete* **Training Data Execution System** that implements the full
> path `documents → tokenized shards → manifests → mixture schedule → packing → batches →
> training → consumption ledger → learning ledger → checkpoint → crash → resume → replay →
> audit`, and **proves** — with generated, reconstructable evidence — that it is *correct,
> reproducible, auditable, and efficient*.

The submission must be a GitHub repo containing: the implementation, a short README, **one
command** that runs the whole demonstration (`python run_demo.py`), automated tests for the
important invariants, a generated `run.log`, and a machine-readable evidence bundle
(`evidence.json` + `evidence.md`) plus generated manifests, ledgers, checkpoints, and a
performance report — all under `submission_artifacts/`.

The two proofs that are the *spine* of the whole thing (the assignment calls them out
specifically):

1. **The run must deliberately crash, resume from checkpoint, and prove the next batch is
   exactly the expected batch** (none skipped, none repeated).
2. **It must replay an earlier interval and prove the reconstructed batch IDs, token spans,
   and hashes match the original run.**

And the grading is adversarial in a specific way (Step 3: *Inspect Code*): "the code will be
inspected to verify that the evidence was produced by the implementation and that the required
behaviour was not simulated or hardcoded." So **every number must be real and
reconstructable**. That single rule shaped the entire tech-choice section of the plan (use a
*real* tiny model so per-token perplexity and loss deltas are measured, not faked).

I mapped the 1,000 points into nine graded blocks and kept that table in front of me the whole
time — it's reproduced in [the plan §6](plan/training-data-execution-system.md). The biggest
blocks (150 each) are packing/masks, mixture/floors/OPUS, the ledgers, and
checkpoint/crash/resume/replay/fork — so those got the deepest research.

---

## Part 2 — How I researched it (five parallel deep-research fan-outs)

The assignment's §17 hands you the reading list: Megatron-Core dataset indexing, Mosaic
StreamingDataset, NeMo Curator dedup, WebDataset sharded tar, and LakeFS/Iceberg/Delta
transaction logs. Rather than read those one at a time, I decomposed the problem into the
distinct engineering questions the system must answer and launched **parallel research
agents**, each of which further fanned out into its own web searches and source-fetches. The
questions:

1. **How is a batch made a pure function of `(seed, step)` so resume is exact and replay is
   hash-identical?** (deterministic/resumable dataloading)
2. **How do you pack documents into fixed windows without wasting compute on padding and
   without cross-document attention leakage — and how do you mark exactly which tokens the
   model learns from?** (packing + the three masks)
3. **How do you store tokenized data as immutable, content-addressed, self-verifying objects
   with audit-grade manifests?** (shards + manifests + lineage)
4. **What is the reusable pattern behind versioned, replayable, forkable data — and what does
   an append-only consumption/learning ledger need to record?** (ledger + replay/fork/audit)
5. **How do you keep evaluation data out of training, and how does contamination detection
   actually work (and fail)?** (the eval firewall + OPUS audit)

The fan-outs returned a large body of **grounded, quotable claims** against primary sources.
The highlights that most shaped the design:

- **Determinism is achieved by *deriving* order, not remembering it.** Megatron-Core builds
  three index arrays (document / sample / shuffle) from a single seed `R` and caches them by a
  constructor hash; the whole ordering is reproducible from `R` forever. torchdata's
  `StatefulDataLoader`, HF's `IterableDataset` (shard-id + example-index), and MosaicML's
  StreamingDataset (identical loss curve across 1/8/16/32/64 GPUs; resume on a *different*
  device count and the curves still match) all confirm the same principle. This is why
  Dossier 01's core recipe is "persist two integers + a seed, regenerate the rest," and why
  the plan forbids buffered shuffle on the replay path (HF documents that buffered shuffle
  breaks exact resumption).

- **The three masks have one lethal failure mode each.** Loss on "grey" tokens (tool
  observations) teaches the model to hallucinate the environment; attention across a packed
  boundary teaches false continuations; position IDs that don't reset per document corrupt
  RoPE phase for the second document. The block-diagonal-causal + per-document-position-reset
  recipe (production systems do this via FlashAttention `cu_seqlens`; we'll do it with a dense
  mask at toy scale) is in Dossier 02 with a worked 8-token micro-example that our tests
  assert against.

- **Content addressing gives immutability, dedup, and lineage for free.** IPFS Merkle DAGs and
  DVC's MD5-keyed cache both prove the point: identity = hash of bytes, so "modify a shard ⇒
  new hash ⇒ new lineage" is enforced by construction, not by discipline. That's the backbone
  of Dossier 03's manifest schema and admission gate.

- **Replay/fork/audit is just Event Sourcing over an append-only commit log.** LakeFS
  (immutable content-addressed commits, zero-copy branches, CAS atomicity), Iceberg (snapshots
  with parent-snapshot-id lineage + atomic metadata swap), and Delta (zero-padded versioned
  JSON commits, state = replay in ascending order, checkpoints = materialized replay, time
  travel for audit) are three instances of *one* primitive. Martin Fowler's Event Sourcing
  names it: "discard the application state completely and rebuild it by re-running the events."
  Dossier 04 adopts this directly — the consumption ledger is the log, run state is its replay,
  checkpoints are Delta-style snapshots bound to a `ledger_offset`, forks are Iceberg-style
  branches with a parent pointer.

- **The firewall is a tripwire, not a wall — and honesty about that matters.** The BIG-bench
  canary GUID (`26b5c67b-…`) and 13-gram overlap (GPT-3's threshold; EleutherAI's harness even
  mandates `PYTHONHASHSEED=0`) catch *copies*. But GPT-4 memorized BIG-bench *despite* the
  canary, and the LMSYS decontaminator showed rephrased/translated test data sails past n-gram
  detection (a 13B model hit MMLU 85.9 trained on rephrases). Dossier 05 builds the
  canary+n-gram(+MinHash) gate *and* states the limitation plainly, which is exactly the kind
  of intellectual honesty Session 5's proxy write-up was rewarded for.

Two of the five fan-outs (packing/masks and throughput) lost their *consolidated* synthesis
step to mid-stream API errors after their sub-searches had already run. Rather than burn more
tokens re-running them, I wrote Dossiers 02 and 06 from domain knowledge with canonical
citations (FlashAttention repo, the two packing papers, the PaLM MFU definition) and flagged
inline that those specific pages should be re-verified before final submission. Everything in
Dossiers 01, 03, 04, and 05 is backed by directly-fetched primary docs.

---

## Part 3 — What I decided to reuse (so the lanes are *real*)

A recurring trap in this course is a data system that demonstrates itself on fake data. I
deliberately wired the plan to **real assets from earlier sessions**, so every capability lane
means something:

- **Tokenizer:** Session 2's `multilingual-tokenizer/output/tokenizer.json` (vocab 10k,
  en/hi/te/mr, score 1415.05) — frozen and hashed. This honors the Session 2 tokenizer
  contract and gives a genuine `tokenizer_hash`.
- **Indic lane:** Session 4's `pipeline/data/clean_asm.jsonl` — real cleaned, provenance-
  stamped Assamese (the same corpus Session 5's proxy used as its Indic arm).
- **Code lane:** the repo's own Python (Session 2/4 scripts).
- **English lane:** the repo's own markdown.
- **Math/reasoning lane:** deterministically generated arithmetic chain-of-thought (a
  synthesized-reasoning stand-in, honestly labeled).

I confirmed the Session 4 `.venv` already has **numpy 2.5.1** and nothing else heavy — which is
why the plan leans numpy+stdlib-only for zero-friction reproducibility, mirroring Session 5's
proxy. (The one genuinely open question is whether loading Session 2's BPE needs the
`tokenizers` library or a small pure-Python port of the verified widget JS — see Part 5.)

---

## Part 4 — How the plan is shaped (breadth + evidence beats depth)

The rubric rewards having **all nine subsystems present and evidenced** more than having one
subsystem gold-plated. So the plan (`plan/training-data-execution-system.md`) specifies
fourteen small, real modules under `session-6/system/`, one `run_demo.py` that runs the exact
canonical path and emits the required `run.log` events and `[PASS]` markers, the precise
`submission_artifacts/` output tree the assignment demands, and ten pytest invariants — one
per correctness property the grader will probe (mask correctness, `plan()` purity, resume
exactness, replay hash-identity, fork divergence, firewall block, protected floors, throughput
reconstructability, …).

I also wrote a **build order in six milestones**, each independently testable and each earning
a distinct rubric block, so partial completion still scores. M1 (sealed data layer) is the
foundation everything else sits on; M5 (checkpoint/crash/resume/replay/fork) is the
operational spine that carries the two headline proofs.

The five design non-negotiables that fell out of the research and now sit at the top of the
plan: (1) order is derived not stored; (2) objects are content-addressed and immutable; (3)
the ledger is the run's memory; (4) evidence is generated never hardcoded; (5) the model is
real, tiny, and honest.

---

## Part 5 — What's decided, and what happens next

The plan ended with four decisions; all four are now **locked** (confirmed with the user), and
all four matched the recommendations (see [plan §9](plan/training-data-execution-system.md)):

- **A. Tokenizer → pure-Python BPE port** of Session 2's verified widget JS — zero deps
  (numpy + stdlib only), preserves the Session 2 tokenizer contract.
- **B. Model → numpy from scratch** — a tiny hand-written LM so per-token perplexity, loss
  deltas, and gradient norms are genuinely measured, with a serializable optimizer state.
- **C. Artifact surface → code + evidence bundle + tests + a few deterministic static SVG
  charts** (mixture / utilization / loss curve), data-driven from the ledger so they can't
  drift from the numbers.
- **D. Scope → all nine subsystems** — breadth + evidence is how the 1,000 points are weighted.

With those settled, the work proceeds through the six milestones, and this walkthrough will
grow a **Part 6+** appended as each milestone lands — the same "flow preserved end to end"
discipline Session 5 used, so by submission this document reads as the complete story from
"first read of the assignment" to "the demo crashed, resumed on the exact next batch, and the
evidence bundle proves it."

---

## Appendix — What's where

- **The assignment (verbatim lesson clip):** [`assignment.md`](assignment.md) /
  [`session-notes.md`](session-notes.md)
- **The research (evidence, with source URLs):** [`research/`](research/) — six dossiers +
  [index](research/README.md)
- **The build design (graded spec of the system):**
  [`plan/training-data-execution-system.md`](plan/training-data-execution-system.md)
- **This narrative:** `WALKTHROUGH.md`
- **Reusable assets:** Session 2 `tokenizer.json`, Session 4 `clean_asm.jsonl`, Session 4
  `.venv` (numpy 2.5.1)

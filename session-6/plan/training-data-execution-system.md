# Session 6 — Implementation Plan: Training Data Execution System (V5)

> **The one-sentence goal (assignment §16):** build a *small but complete* system that proves
> it can say **what it consumed, why it consumed it, what the model learned from it, and how
> the run can be reconstructed** — correct, reproducible, auditable, and efficient. Not scale;
> **proof**.
>
> This plan turns the six research dossiers (`../research/`) into concrete modules, a single
> `run_demo.py`, an evidence bundle, and invariant tests, and maps every piece to the
> 1,000-point rubric.

---

## 0. Design principles (the non-negotiables)

1. **Order is derived, not stored.** Every batch is a pure function of `(seed, global_step)`
   (Dossier 01). This is what makes resume exact and replay hash-identical.
2. **Objects are content-addressed and immutable.** Shard identity = `sha256(bytes)`; editing
   ⇒ new hash ⇒ new lineage (Dossier 03).
3. **The ledger is the run's memory.** Append-only event log; run state = replay of the log;
   checkpoints = materialized snapshots bound to a ledger offset (Dossier 04).
4. **Evidence is generated, never hardcoded.** Every PASS/FAIL in `evidence.json` points at a
   real artifact the grader can recompute (assignment §Evaluation, Step 3).
5. **Real, tiny, honest.** A genuine gradient-trained model produces *real* per-token
   perplexity and loss deltas — no simulated numbers (Dossier 04 §5).

---

## 1. Tech choices (and why)

| Choice | Decision | Why |
| --- | --- | --- |
| Language | Python 3, **numpy only** (+ stdlib `hashlib`, `json`, `struct`) | Session 4's `.venv` already has numpy 2.5.1; zero extra deps ⇒ trivially reproducible for the grader. Mirrors Session 5's proxy approach. |
| Tokenizer | **Reuse Session 2 `tokenizer.json`**, frozen + hashed | Honors the "Session 2 tokenizer contract"; gives a real `tokenizer_hash`. **Open question A** below: depend on `tokenizers`, or port the widget's pure-JS BPE to Python, or fall back to a byte tokenizer. |
| Corpus | **Reuse Session 4 `clean_asm.jsonl`** (Indic) + repo English markdown + Session 2/4 Python (code) + generated arithmetic CoT (math) | Real, cleaned, provenance-stamped data across ≥4 capability lanes — exactly Session 5's proxy corpus, so lanes are meaningful. |
| Model | Tiny char/token-level LM in numpy (hand-written forward/backward), ~10k–100k params | Enough to produce real cross-entropy, perplexity, loss deltas, gradient norms, and to save/restore optimizer state. Not about quality. |
| Storage | Local files under `submission_artifacts/` | Grader runs `run_demo.py` and inspects the tree. |

**Open question A (tokenizer dependency):** cleanest is a **pure-Python byte-level BPE
encoder** that loads Session 2's `tokenizer.json` merges (port of the verified widget JS), so
the whole system stays numpy+stdlib only. Fallback: `pip install tokenizers`. Decision needed
before coding — see "Questions for you" at the end.

---

## 2. Module map (what we build)

```
session-6/system/                      # the implementation
  tokenizer_frozen.py    # load + hash Session-2 tokenizer; freeze; encode/decode (Dossier 03)
  shards.py              # build_shard, verify_shard, content_hash, WebDataset-style records (03)
  manifest.py            # manifest schema + admission gate (blocks unsafe/eval shards) (03)
  firewall.py            # eval registry; canary + 13-gram (+MinHash) overlap check (05)
  packing.py             # concat_chop / best_fit / structure_preserving / long_context (02)
  masks.py               # loss mask (pretrain|sft|agentic), block-diag attn, per-doc pos ids (02)
  mixture.py             # curriculum stages -> per-step quotas; floors; anneal reserve (04/§7)
  dataloader.py          # plan(seed, step) -> Batch; pure fn; counters; batch_hash (01)
  opus.py                # candidate scoring; accept/reject/defer; protected-floor override (05/§10)
  model.py               # tiny numpy LM: forward, backward, optimizer (Adam), state_dict (01)
  ledger.py              # append-only JSONL consumption + learning ledgers; offsets (04)
  checkpoint.py          # {model,optim,sched,rng,loader,ledger_offset,cfg_hash,branch} (01/04)
  runner.py              # resume / replay(interval) / fork(at) / audit(range) (04/§14)
  throughput.py          # raw/useful/accepted tokens/sec; utilization; performance.json (06)
  evidence.py            # collect PASS/FAIL -> evidence.json + evidence.md (assignment)
run_demo.py              # ONE command: runs the whole path end to end
tests/                   # pytest invariants (see §5)
```

---

## 3. The end-to-end demo flow (`run_demo.py`)

Follows the assignment's canonical path (§16) and emits the required `run.log` events (§Execution Log):

```
1.  freeze tokenizer, verify hash                      -> [PASS] tokenizer_hash_verified
2.  build tokenized shards from 4-lane corpus          -> "shards created"
3.  build manifests; run admission gate                -> "manifests validated"
4.  register eval/val/test shards in firewall
5.  inject an eval shard into candidate selection      -> [PASS] eval_shard_blocked
6.  compile mixture timeline (stages, floors, anneal)  -> "mixture compiled"
7.  pack batches (per-lane policy) + build masks       -> "batches packed"
8.  train N steps: plan(seed,step) -> model.step
       OPUS scores candidates (accept/reject/defer)    -> "OPUS decisions recorded"
       append consumption + learning ledger events
       per-token perplexity trace for one batch
9.  checkpoint at step K                                -> [PASS] checkpoint_saved
10. DELIBERATE CRASH mid-step
11. resume from checkpoint; assert next batch == expected -> [PASS] resume_next_batch_matched
12. replay earlier interval; assert batch hashes match -> [PASS] replay_hash_matched
13. fork from an earlier checkpoint (new branch id)    -> "branch forked"
14. audit: which shards trained steps [a,b]            -> "audit completed"
15. measure throughput                                 -> "performance measured"
16. write evidence.json + evidence.md
```

Everything runs without manual intervention (assignment: "one command … without manual
intervention").

---

## 4. Output tree (exactly what the assignment asks for)

```
submission_artifacts/
  run.log                 # full event sequence + [PASS] markers
  evidence.json           # machine-readable: every requirement PASS/FAIL + evidence pointer
  evidence.md             # human table (Requirement | Result | Evidence)
  manifests/              # shard + eval-registry manifests (JSON)
  ledgers/                # consumption.jsonl, learning.jsonl, opus.jsonl (+ per branch)
  checkpoints/            # ckpt-*.npz (+ dataloader/ledger offsets)
  performance.json        # raw/useful/accepted tokens/sec, utilization, cache-hit
```

`evidence.md` table (assignment §Evidence Bundle) — one row per graded requirement:

| Requirement | Result | Evidence |
| --- | --- | --- |
| Tokenizer integrity | PASS/FAIL | manifest tokenizer_hash record |
| Evaluation firewall | PASS/FAIL | blocked-shard event in run.log/ledger |
| Packing correctness | PASS/FAIL | packed-batch report + mask assertions |
| Mixture compliance | PASS/FAIL | planned vs actual lane shares |
| OPUS audit trail | PASS/FAIL | candidate decision records |
| Crash recovery | PASS/FAIL | expected vs resumed batch ids/hash |
| Replay | PASS/FAIL | original vs replay batch hashes |
| Learning trace | PASS/FAIL | per-token loss linked to shard/lane |
| Throughput | PASS/FAIL | performance.json |

---

## 5. Invariant tests (`tests/`, the graded "important invariants")

1. **`test_tokenizer_frozen`** — re-hash tokenizer == recorded hash; encode(text) is stable.
2. **`test_shard_immutability`** — mutating a shard byte changes `content_hash`; admission
   re-derives hash and detects tampering.
3. **`test_mask_correctness`** — no attention edge crosses a `doc_id` boundary; position ids
   reset per doc; loss mask 0 on pad and (SFT) on prompt. (Dossier 02 §5 micro-example.)
4. **`test_plan_is_pure`** — `plan(seed, step)` returns byte-identical batch on repeat calls
   and across process restarts.
5. **`test_resume_next_batch`** — crash at step K, resume, next batch hash == ledger[K+1].
   **No batch skipped or repeated.**
6. **`test_replay_hashes`** — replaying interval reproduces recorded batch hashes exactly.
7. **`test_fork_divergence`** — fork records a new branch id + divergence step; branches share
   history up to the fork and differ after.
8. **`test_firewall_blocks_eval`** — an eval shard injected into candidates is blocked + logged.
9. **`test_mixture_floors`** — protected floors are never violated even when OPUS wants to cut a
   lane; anneal reserve is untouched until the anneal stage.
10. **`test_throughput_reconstructable`** — `raw ≥ useful ≥ accepted`; utilization ∈ (0,1];
    numbers in `performance.json` recompute from the ledger.

---

## 6. Rubric mapping (how we earn the 1,000 points)

| Area | Pts | Where earned |
| --- | --- | --- |
| End-to-end execution | 150 | `run_demo.py` runs the full path, one command, generates the tree |
| Shards, manifests, tokenizer integrity | 100 | `shards.py` + `manifest.py` + frozen tokenizer hash + immutability test |
| Packing, masks, batch correctness | 150 | `packing.py` + `masks.py` + mask/purity tests |
| Mixture, protected floors, OPUS | 150 | `mixture.py` + `opus.py` + floor/anneal tests + planned-vs-actual |
| Consumption & learning ledgers | 150 | `ledger.py` + per-token perplexity trace + per-shard report card |
| Checkpoint, crash, resume, replay, fork | 150 | `checkpoint.py` + `runner.py` + resume/replay/fork tests |
| Eval/validation firewall | 50 | `firewall.py` + block test + never-train flag |
| Throughput & packing efficiency | 50 | `throughput.py` + `performance.json` + reconstructability test |
| Tests, evidence quality, docs | 50 | `tests/` + `evidence.py` + README + this plan + walkthrough |

---

## 7. Build order (proposed milestones)

1. **M1 — Sealed data layer:** tokenizer freeze + shards + manifests + admission gate + tests
   (earns the 100-pt block; foundation for everything).
2. **M2 — Batch layer:** packing + masks + position ids + the pure `plan()` + batch_hash (150).
3. **M3 — Mixture + firewall + OPUS:** stages/floors/anneal, eval registry + block, OPUS
   decisions (150 + 50).
4. **M4 — Train + ledgers:** tiny model training loop, consumption + learning ledgers,
   per-token perplexity (150).
5. **M5 — Checkpoint/crash/resume/replay/fork/audit:** the operational spine (150).
6. **M6 — Throughput + evidence + `run_demo.py` + docs:** wire it all into one command, emit
   the bundle, write README (150 e2e + 50 throughput + 50 docs).

Each milestone is independently testable and independently earns rubric points, so partial
completion still scores.

---

## 8. Risks & mitigations

- **Tokenizer dependency** (Open question A) — resolve up front; pure-Python BPE keeps deps at
  numpy+stdlib.
- **numpy model too weak to show loss deltas** — mitigate by using tiny vocab/tiny corpus so
  loss visibly drops in a few hundred steps (Session 5 proved this works at ~43k params).
- **Determinism leaks** — forbid buffered shuffle on the replay path; fix all seeds; set
  `PYTHONHASHSEED=0`; hash batches to *catch* any leak in tests.
- **Scope creep** — the assignment says "small but complete." Keep each subsystem minimal but
  *real*; breadth (all subsystems present + evidenced) beats depth in one.

---

## 9. Decisions (locked — confirmed with the user)

- **A. Tokenizer → pure-Python BPE port.** Port Session 2's verified widget JS BPE to Python;
  load `tokenizer.json`, freeze it, hash it. Keeps the whole system **numpy + stdlib only** and
  preserves the Session 2 tokenizer contract.
- **B. Model → numpy from scratch.** Tiny hand-written LM (Session-5-proxy style) with real
  forward/backward, Adam, and a serializable `state_dict` — so per-token perplexity, loss
  deltas, and gradient norms are genuinely measured. Zero deps.
- **C. Artifact surface → code + evidence bundle + tests + a few static SVG charts.** Required
  `submission_artifacts/` + pytest invariants, plus deterministic stdlib-only SVG charts
  (mixture shares, packing utilization, loss/perplexity curve) embedded in the README —
  data-driven from the same numbers as the ledger so they can't drift.
- **D. Scope → all 9 rubric subsystems.** Breadth + evidence is how the 1,000 points are
  weighted; each of the six milestones (§7) earns a distinct block.

All four match the plan's recommendations, so nothing in §§1–8 changes.

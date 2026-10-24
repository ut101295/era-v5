# Session 6 — Training Data Execution System

A small but complete data system for V5 training. It proves what the model consumed,
why it consumed it, what it learned from it, and how to reconstruct the run. Everything is
numpy + stdlib, runs on a laptop CPU in under a minute, and is deterministic down to the
byte.

```bash
cd session-6
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py        # regenerates submission_artifacts/, exits non-zero on any FAIL
pytest -q                 # 45 invariant tests
```

The demo deletes and rebuilds `submission_artifacts/`. Its summary is
[evidence.md](submission_artifacts/evidence.md), and the full event sequence is
[run.log](submission_artifacts/run.log). Current result: **12/12 requirements PASS**.

## The path, and where each step lives

```
corpus snapshot ─▶ decontaminate ─▶ tokenized shards ─▶ manifests + admission ─▶ eval firewall
   corpus.py        decontam.py       shards.py           manifest.py              firewall.py

 ─▶ mixture timeline ─▶ packing + masks ─▶ plan(seed, step) ─▶ OPUS ─▶ train step
      mixture.py        packing.py masks.py   dataloader.py     opus.py  trainer.py model.py

 ─▶ consumption / learning / OPUS ledgers ─▶ checkpoint ─▶ crash ─▶ resume ─▶ replay / fork / audit
      ledger.py                               checkpoint.py   worker.py         runner.py run_demo.py
```

`build.py` wires it all from one `Config`; `throughput.py` and `evidence.py` measure and
report. The model (`model.py`) is a tiny mean-context LM: each position averages the
embeddings its attention mask allows (causal, same document) plus a learned position
embedding, so masks and position ids really shape learning. It is not a transformer and
quality is not the point; the per-token losses, loss deltas and gradient norms in the ledgers
are measured, not invented.

## Design decisions

**Order is derived, not stored.** A batch is a pure function of `(seed, step)` over frozen
shards. Resume, replay and audit all reduce to "plan step N again and compare hashes", so
no data-loader state needs saving beyond the step.

**The crash is a real crash.** Training runs in a separate process (`system/worker.py`).
It is killed with `SIGKILL` inside step 187, after the optimizer updated the weights in
memory but before that step's consumption event was written. A second process restores
`ckpt-000149`, rolls every ledger back to the offsets that checkpoint recorded (37
consumption, 37 learning and 38 OPUS events, the extra one being the half-written step
187), and keeps the rolled-back tail in `ledgers/crash_tail/` as evidence.

**Recovery is checked against an uninterrupted run, not just the next batch.** The same
config also runs straight through in-process. The crashed-and-resumed run's consumption,
learning and OPUS ledgers are byte-identical to it, and so is every checkpoint's weight
file. Replay re-runs steps 50–99 from `ckpt-000049` and matches batch hashes, doc ids,
token spans, loss-mask hashes *and* losses. Two artifact runs differ only in wall-clock
timings.

**Checkpoints carry OPUS state.** OPUS's protected-floor decisions depend on running
per-lane acceptance counts. Without them a resumed run re-evaluates floors from zero: in
testing, step 28 flipped from `protected` to `rejected`, a batch that should have trained
was skipped, and the weights diverged. Checkpoints now save the counters, the OPUS ledger
offset and a weights hash.

**Held-out data is decontaminated, not just checked.** The firewall found that the
training lanes really overlapped the held-out sets: 535 shared 13-grams in English and 79
in code, mostly assignment text quoted in session notes. Instead of dropping 16 of 20
English documents, `decontam.py` cuts only the tokens covered by an overlapping 13-gram
(2,136 English tokens, 270 code tokens) and keeps the clean pieces. Afterwards every train
shard passes the firewall, and a post-run scan finds zero held-out n-grams in any batch
that trained, across main, replay and fork.

**One packing policy per data type.** Pretraining lanes (Indic, English, code) use
`concat_chop` with an EOS between documents. The math lane is prompt/response formatted, so
it uses `structure_preserving` (never splits a pair) and an SFT loss mask: the question
never bears loss, only `Reasoning: …` onward. Attention is block-diagonal and causal, and
position ids reset at every document. `reports/packing.json` compares all four policies on
every lane.

**OPUS threshold, and the bias it exposes.** OPUS scores a batch by loss-bearing
utilization. `accept_threshold=1.02` sits inside the observed score range, so all four
outcomes occur: 202 accepted, 38 protected, 27 rejected, 33 deferred. The proxy is
biased against the SFT math lane, because its masked prompt lowers utilization. The
protected floor is what keeps that lane alive, and `opus_summary.json` shows it by lane.
That's the case the assignment means when it says a rejected batch may reveal proxy bias.

**Frozen inputs.** `corpus/snapshot.jsonl` (204 KB) holds the exact documents used:
Assamese from Session 4 (its source file is gitignored), the repo's own Markdown and
Python, and synthetic arithmetic. Reading the repo live would change shard hashes with
every commit. `python -m system.corpus --rebuild` regenerates it. The tokenizer is Session
2's *faithful* resubmission (Metaspace + byte fallback), ported to pure Python;
`scripts/verify_tokenizer_port.py` checks it token-for-token against HuggingFace
`tokenizers` (0 mismatches over 103,965 tokens). The original Session 2 tokenizer turned
about 10% of English and code tokens into `<unk>`.

**Short run, full curriculum.** `stage_token_scale=0.5` shrinks the three stage boundaries
so 300 steps pass through general (0–104), capability midtrain (105–234) and anneal
(235–299). In every stage each lane's actual share stays within two batches of its plan
(the 2/n bound the error-diffusion scheduler guarantees).

## Bugs found while building the demo

The library's tests passed, yet the end-to-end evidence exposed four real defects. All
are fixed and covered by tests in `tests/test_m6_demo_invariants.py`:

1. **The scheduler starved lanes after a stage change.** Lane targets were
   `current_weight × total_steps`, which re-priced earlier stages at the new weights.
   English and code got zero batches for the entire anneal stage. Targets now accumulate
   per step.
2. **Resume wasn't exact** when OPUS floors fired (above).
3. **Training data overlapped the eval and validation sets** (above).
4. **It failed on a fresh clone.** The Indic lane read a gitignored file and silently came
   back empty (fixed by the snapshot).

## Artifacts

| Path | What it proves |
| --- | --- |
| `run.log`, `evidence.json`, `evidence.md` | Event sequence; every requirement with its checks and evidence pointers |
| `manifests/` | Tokenizer record, shard manifests, admission (incl. tampered/mis-tokenized/held-out refusals), eval registry, mixture timeline |
| `shard_store/` | Immutable tokenized shards (`.bin` + doc index) |
| `ledgers/` | Consumption, learning and OPUS ledgers per branch; firewall events; token-level loss trace; validation losses; crash tail |
| `checkpoints/` | Checkpoint metadata and `index.json` (offsets, weights hash, `.npz` sha256) |
| `reports/` | Packing, mixture compliance, OPUS summary, crash recovery, replay, fork, audit, learning trace, decontamination |
| `performance.json` | Raw / loss-bearing / trained tokens per second, utilization, per-lane packing; counts rebuilt from the ledger |
| `reference_run/` | The uninterrupted yardstick run's ledgers and checkpoint metadata |

Checkpoint weights (`*.npz`, 13 MB each) are not committed. Their sha256s are in
`checkpoints/index.json`.

What reproduces where: shards, manifests, batch hashes, token spans, the lane schedule and
OPUS decisions (scored from token counts and a hash, not the model) regenerate identically
on any machine. Losses and weights are floating point. They regenerate byte-identically on
the same numpy/BLAS build (two runs here differ only in timings), but a different build
may differ in the last bits. Every PASS/FAIL check compares results produced within one run, so
the checks themselves hold on any machine.

## Limitations

- n-gram decontamination catches copies, not paraphrases or translations.
- OPUS uses a utilization proxy, not a learned value model. Deferred batches are recorded
  but never re-offered.
- One process, one rank: the ledgers carry `rank` and `microbatch_id` fields but nothing
  exercises multi-rank sharding.
- Throughput rates are for this tiny CPU model and only meaningful relative to each other.
  The counts are exact.

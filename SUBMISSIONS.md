# Axiom — Submission Tracker

One place to collect every link the course portal asks for. Each section lists the
**exact fields the portal shows** and the **artifact that answers it** — GitHub links are
filled in and verified, and so are all five Netlify links.

- GitHub repo: `https://github.com/ut101295/era-v5`
- Convention for GitHub links below: `https://github.com/ut101295/era-v5/blob/main/<path>`

---

## 0. Before you submit anything

**✅ Everything is pushed to `main`.** Merged via PR #2 (`14ec751`); local `main` matches
`origin/main`. Every GitHub link below was verified to resolve against the pushed tree,
along with every `.ipynb` and training log the briefs require. The tree is
clean — no `.venv/`, `__pycache__/`, `.pytest_cache/`, or `.DS_Store` was committed.

**✅ All five Netlify sites are live and verified** (2026-10-24). Every site returns 200,
and every served file is byte-identical to the committed source. The only differences are
an HTML comment and a toolbar script that Netlify adds at serve time. Every asset the pages
fetch (`tokenizer.json`, `metrics.json`, `engine.mjs`, all four `corpus/*.faithful.txt`,
`stats.json`) loads and matches the repo.

| Session | Live URL | Source folder |
| --- | --- | --- |
| 1 | https://era-v5-s1-playground.netlify.app | [session-1/assignment-playground/](session-1/assignment-playground/) |
| 2 | https://era-v5-s2-tokenizer.netlify.app | [session-2/faithful-tokenizer/widget/](session-2/faithful-tokenizer/widget/) |
| 3 | https://era-v5-s3-model-design.netlify.app | [session-3/model-design-report/](session-3/model-design-report/) |
| 4 | https://era-v5-s4-cleaning.netlify.app | [session-4/cleaning-widget/](session-4/cleaning-widget/) |
| 8 | https://era-v5-s8-attn-timeline.netlify.app | [session-8/attention-timeline/](session-8/attention-timeline/) |

The sites for sessions 1 and 2 were renamed from their random Drop names, and the
session 8 site was renamed from `era-v5-s8-attention-vkdp`. **The old URLs now return 404**
(`funny-rabanadas-b407f7`, `stalwart-centaur-f5cfaf`, `era-v5-s8-attention-vkdp`).
If either old link was already pasted into the portal, update it there. The original
0-scoring tokenizer site (`coruscating-taffy-9b4346`) has been deleted.

To redeploy after a change, run from the repo root:
`npx netlify-cli deploy --prod --no-build --dir <source folder> --site <site name>`

**Session 6 is built and pushed to branch `session-6-demo`.** It needs to be merged into
`main` before its links resolve. See its section below.

---

## Status at a glance

| # | Topic | Portal fields | Points | Ready? |
| --- | --- | --- | --- | --- |
| 1 | NN playground | Netlify link; LinkedIn (opt) | 500 (+250) | ✅ Submitted |
| 2 | BPE tokenizer | Widget link; tokenizer.json link | 1000 | ✅ Submitted |
| 3 | 40B data/tokenizer report | Netlify link | 500 | ✅ Submitted |
| 4 | Cleaning strategies widget | Netlify link | 500 | ✅ Submitted |
| 5 | Mixture & curriculum plan | GitHub README | 500 | ✅ Submitted |
| 6 | Training Data Execution System | Repo; run.log; evidence.json; evidence.md | 1150 | 🟡 Merge, then submit |
| 7 | Kronecker problem | Which problem; GitHub README/app | 1000 | ✅ Submitted |
| 8 | Attention timeline | Netlify link (+ Q2, Q3) | 1000 (+1250) | ✅ Submitted |
| 9 | Loss harness | GitHub README | 1000 | ✅ Submitted |
| 10 | Training loop truth | GitHub README | 1000 | ✅ Submitted |
| 11 | Optimizers / Adam by hand | GitHub README | 1000 | ✅ Submitted |
| 12 | 32 virtual GPUs, ZeRO 0–3 | GitHub README | 1000 | ✅ Submitted |
| 13 | 20M LLM + reversibility | GitHub README | 1000 | ✅ Submitted |
| 14 | Linear → MoE upcycle | GitHub README | 1000 | ✅ Submitted |

Core total **12,150** · optional bonus **1,500**.

Submitted: every session except 6. Session 6 (**1,150 points**) is ready once
`session-6-demo` is merged into `main`.

---

## Session 1 — Activations, depth, embeddings, generalization

Portal: `.../assignments/cmqutyi1i01hi08njprnpfxwq`

| Field | Points | Source |
| --- | --- | --- |
| Netlify Link — one link showing all 4 examples | 500 | [session-1/assignment-playground/index.html](session-1/assignment-playground/index.html) |
| LinkedIn (optional) | 250 | your post |

- ✅ **Paste this:** `https://era-v5-s1-playground.netlify.app`
- LinkedIn: `________________`

The single `index.html` covers all four claims (ring boundary, 5-linear collapse,
embedding clusters, generalization gap), so one drop fills the field.

---

## Session 2 — Multilingual BPE tokenizer

Portal: `.../assignments/cmr4vjd9l03we09pl452oio80`

| Field | Points | Source |
| --- | --- | --- |
| Widget Link | 500 | [session-2/faithful-tokenizer/widget/index.html](session-2/faithful-tokenizer/widget/index.html) |
| Tokenizer.json — direct download, **no Google Drive** | 500 | [session-2/faithful-tokenizer/widget/tokenizer.json](session-2/faithful-tokenizer/widget/tokenizer.json) |

- ✅ **Widget Link:** `https://era-v5-s2-tokenizer.netlify.app`
- ✅ **Tokenizer.json:** `https://era-v5-s2-tokenizer.netlify.app/tokenizer.json`

Verified 2026-10-24: running `node verify.mjs` on the deployed files reproduces every
per-language token count, passes the strict round-trip on all four languages, and gives a
score of **28,026.41**. That matches `metrics.json` and what the live widget shows.

**Submit the `faithful-tokenizer/`, not `multilingual-tokenizer/`.** Per
[session-2/RESUBMISSION_FIX.md](session-2/RESUBMISSION_FIX.md) the original byte-level
tokenizer scored **0** on regrade — it failed the faithful round-trip gate by dropping
`[`, `]`, and 106 other characters. The resubmission passes the strict gate and scores
≈ **28,026**. Deploying the widget folder satisfies both fields at once: the widget URL
plus `<widget-url>/tokenizer.json` as a direct, non-Drive download.

---

## Session 3 — 40B India-first model: data & tokenizer design

Portal: `.../assignments/cmreurukl0jan08jxagxso0bf`

| Field | Points | Source |
| --- | --- | --- |
| Netlify Link | 500 | [session-3/model-design-report/index.html](session-3/model-design-report/index.html) |

- ✅ **Paste this:** `https://era-v5-s3-model-design.netlify.app`

The brief grades depth of thought and explicitly penalizes length — *"longer submissions
will result in lower scores."* Worth a skim of the report for padding before deploying.
It must answer all four: data collection (pre-train / post-train / RL), cleaning,
evaluation, and per-language fertility targets → vocab size.

---

## Session 4 — Cleaning strategies applied to a real dataset

Portal: `.../assignments/cmrokwadd012z08o1eou3k34c`

| Field | Points | Source |
| --- | --- | --- |
| Netlify Link | 500 | [session-4/cleaning-widget/index.html](session-4/cleaning-widget/index.html) |

- ✅ **Paste this:** `https://era-v5-s4-cleaning.netlify.app`

The widget must state: how many strategies the session listed and what they are, which
dataset was picked, what was cleaned and why, any extra strategy/concern, and final
statistics. [session-4/cleaning-widget/stats.json](session-4/cleaning-widget/stats.json)
backs the numbers.

---

## Session 5 — Mixture & curriculum plan for V5

Portal: `.../assignments/cmryodmtg00fe09pd4l2n8jst`

| Field | Points | Source |
| --- | --- | --- |
| Github README.md Link | 500 | [session-5/README.md](session-5/README.md) |

- ✅ **Paste this:** `https://github.com/ut101295/era-v5/blob/main/session-5/README.md`

Highest marks go to a plan written as a **testable hypothesis** with a concrete 1B/3B
proxy experiment — and the very highest to one that actually ran it.
[session-5/proxy/](session-5/proxy/) contains `train.py` and `results.json`, so make sure
the README surfaces those numbers rather than burying them.

---

## Session 6 — Training Data Execution System

Portal: `.../assignments/cms8j538f0kgx09mmbjztek4j`

**Status (2026-10-24): built, verified, pushed to branch `session-6-demo` (commit `9cf1bea`).
Merge it into `main`, then paste the links below.**

To merge, either open
https://github.com/ut101295/era-v5/compare/main...session-6-demo and click
*Create pull request*, then *Merge*, or from the repo root run:

```bash
git checkout main && git pull --ff-only
git merge --no-ff session-6-demo
git push origin main
```

| Field | Points | Paste this (resolves after the merge) |
| --- | --- | --- |
| Github Repo Link | 1000 | `https://github.com/ut101295/era-v5/tree/main/session-6` |
| Github run.log link | 50 | `https://github.com/ut101295/era-v5/blob/main/session-6/submission_artifacts/run.log` |
| Github evidence.json link | 50 | `https://github.com/ut101295/era-v5/blob/main/session-6/submission_artifacts/evidence.json` |
| Github evidence.md link | 50 | `https://github.com/ut101295/era-v5/blob/main/session-6/submission_artifacts/evidence.md` |

**Verified:** `python run_demo.py` gives **12/12 requirements PASS** in about 47 s, and
`pytest -q` passes 45 tests. A fresh clone with a fresh venv (different numpy patch version)
reproduced every ledger, manifest, report and checkpoint hash exactly; only the timings
differed.

**What was built:** `run_demo.py` (one command), a real `SIGKILL` crash and resume in
separate processes checked byte-for-byte against an uninterrupted run, replay, fork, audit,
a token-level learning trace, a decontaminated firewall, ledger-reconstructed throughput,
and a fail-closed evidence bundle. Details are in [session-6/README.md](session-6/README.md).

**Bugs fixed along the way:** the mixture scheduler starved English and code for the whole
anneal stage; OPUS state was missing from checkpoints (resume diverged); train lanes
overlapped the held-out sets; the corpus failed on a fresh clone; the old tokenizer mapped
about 10% of English and code tokens to `<unk>`.

---

## Session 7 — Kronecker Embedding V2

Portal: `.../assignments/cmsmro6ak1a2n08nhbmvtglqd`

| Field | Points | Source |
| --- | --- | --- |
| Which Problem did you work on? | 0 | **Problem 3 — dynamic word length** (removing the 32-byte cap) |
| GitHub README or App link | 1000 | [session-7/kronecker-dynamic/README.md](session-7/kronecker-dynamic/README.md) |

- Problem: `Problem 3 — dynamic-length Kronecker embeddings (continuous position function φ(p) replacing the 32-slot one-hot)`
- ✅ **Paste this:** `https://github.com/ut101295/era-v5/blob/main/session-7/kronecker-dynamic/README.md`

Proof artifacts are in place: `run_demo.py`, `collision_report.json`,
`training_report.json`, plus the trained-transformer comparison described in
[session-7/WALKTHROUGH.md](session-7/WALKTHROUGH.md).

---

## Session 8 — Attention mechanisms, chronological

Portal: `.../assignments/cms8amsskpsztzvtm1je`

| Field | Points | Source |
| --- | --- | --- |
| Netlify Link (Q1 — live app + repo) | 1000 | [session-8/attention-timeline/index.html](session-8/attention-timeline/index.html) |
| Q2 — what the timeline shows; a mechanism not covered | 1000 | see below |
| Q3 — LinkedIn/X/Medium (optional) | 250 | your post |

- ✅ **Live app:** `https://era-v5-s8-attn-timeline.netlify.app`
- ✅ GitHub repo link: `https://github.com/ut101295/era-v5/tree/main/session-8/attention-timeline`
- ✅ Q2 answer: drafted below, condensed from the app's closing section and the README
- Q3 post: `________________`

**Paste-ready text for the single portal field:**

```text
Live app: https://era-v5-s8-attn-timeline.netlify.app
GitHub:   https://github.com/ut101295/era-v5/tree/main/session-8/attention-timeline
          (README lists the source for every date: arXiv v1 submission date, or the
          original post for NTK-aware scaling)

Q2 — What the timeline shows that a list can't

1. It oscillates; it doesn't progress. The field cuts compute (Sparse Transformers 2019,
   top-k 2019, sliding window 2020), then wants memory back (MQA 2019 -> GQA 2023 ->
   MLA 2024), then wants length (RoPE 2021, ALiBi 2021, NTK 2023, YaRN 2023, sinks
   2023), then wants memory back again by replacing the quadratic core with a recurrent
   state. That last thread starts early (linear attention 2020, delta rule 2021) but only
   becomes mainstream with DeltaNet and Gated DeltaNet in 2024.

2. 2019 is the fork. MQA (cut memory) and Sparse Transformers (cut compute) appear
   months apart, and nearly every later branch descends from one of the two.

3. Position encoding is a 2017 decision the field never stopped paying for.
   Sinusoidal -> RoPE -> ALiBi -> NTK -> YaRN -> DroPE is one eight-year thread of
   patching the same choice. It ends in 2025 with a paper arguing you should drop
   positional embeddings after training, i.e. undo the 2017 move.

4. 2024-25 is convergence, not divergence. MLA, NSA and Gated DeltaNet each combine
   earlier threads (compression + sparsity + gated recurrence) instead of opening a new
   axis. The prediction: the next step is a hybrid that pays all three bills at once.

Mechanisms not covered in class (with sourced dates):

- FlashAttention, Dao et al., arXiv:2205.14135, v1 2022-05-27. It's the odd one out:
  every other 2019-22 entry makes attention cheaper by approximating it, while
  FlashAttention keeps it exact and attacks the memory-IO cost instead, tiling the
  computation so it stays in SRAM. Much of the later long-context work assumes it.
- Ring Attention, Liu et al., arXiv:2310.01889, v1 2023-10-03. Shards the sequence
  across devices so context length scales with the device count.
- Infini-attention, Munkhdalai et al., arXiv:2404.07143, v1 2024-04-10. Adds a
  compressive memory to a standard attention block for unbounded context with bounded
  memory.
```

✅ **Submitted.** Still open in the app: the "Learned absolute positions" card is dated to
*Attention Is All You Need* (2017-06-12), but that paper credits learned positional
embeddings to Gehring et al., *Convolutional Sequence to Sequence Learning*,
arXiv:1705.03122 (v1 2017-05-08). Fixing it means updating the card and the README date
table, then redeploying.

The portal exposes **one** field but the brief asks three questions — paste the live
link, the repo link, and the Q2 answer into that field together, or Q2's 1,000 points go
unclaimed.

Two things this is graded hardest on:
- **Dates must be verified** against the actual paper/release, and the README must name
  the sources. The brief warns the agent will invent dates.
- **Honest pros and cons** per mechanism — "if you write down a technique with only
  pros, you have not understood it yet."

Q2 wants what the *date order* reveals that a list cannot — the walkthrough frames it as
the field oscillating: exactness → memory → length → memory again. Naming an uncovered
mechanism with a sourced date is where the extra 1,000 sits.

---

## Session 9 — Loss harness

Portal: `.../assignments/cmt3xb6yo0j9e09s1cp2vbydr`

| Field | Points | Source |
| --- | --- | --- |
| GitHub README.md link (repo must hold the ipynb / training logs) | 1000 | [session-9/loss-harness/README.md](session-9/loss-harness/README.md) |

- ✅ **Paste this:** `https://github.com/ut101295/era-v5/blob/main/session-9/loss-harness/README.md`

Write-up must carry **the seven numbers from Part 1** and **both losses from Part 2**:

1. every tensor shape + one line per dimension
2. shift verified by printing token **strings**, inputs beside targets
3. padding masked, contributing-token count changes
4. two documents packed, boundary masked, loss before vs. after
5. perplexity — untrained model near vocab size
6. tied vs. untied head parameter counts
7. peak memory, ordinary vs. your chunked cross-entropy, plus the ratio

Part 2: second head predicting `t+2`, both losses separately and their sum, and what
happens to head 2's loss over training relative to head 1.

Backing files present: `loss_harness.ipynb`, `harness.py`, `results.txt`.

---

## Session 10 — Make the training loop tell the truth

Portal: `.../assignments/cmtdud74c173g09s136ltf9zh`

| Field | Points | Source |
| --- | --- | --- |
| GitHub README.md (repo must have the ipynb) | 1000 | [session-10/training-loop/README.md](session-10/training-loop/README.md) |

- ✅ **Paste this:** `https://github.com/ut101295/era-v5/blob/main/session-10/training-loop/README.md`

Must cover: every tensor shape with meaning; one gradient verified by hand against
`backward()`; gradient accumulation deliberately broken (average-of-averages with
uneven micro-batches) with **both curves plotted together**; grad norm logged every step
plus the step where it moved before the loss; your own MFU, honestly reported, with what
costs you the distance to 40%; and **0.1 written out bit-by-bit in fp32, bf16 and fp8
E4M3** with which you'd train in and why.

Backing files: `train_loop.ipynb`, `fig_accum.png`, `fig_gradnorm.png`, `results.txt`.

---

## Session 11 — Optimizers

Portal: `.../assignments/cmts6t1qa009o09qk13ucubdq`

| Field | Points | Source |
| --- | --- | --- |
| GitHub README.md | 1000 | [session-11/optimizers/README.md](session-11/optimizers/README.md) |

- ✅ **Paste this:** `https://github.com/ut101295/era-v5/blob/main/session-11/optimizers/README.md`

Must cover: Adam reproduced by hand (one weight, five gradients → m, v, m̂, v̂, step)
checked against PyTorch; bias correction disabled, first twenty steps both ways, plus
the step count after which the difference stops mattering; update-to-weight ratio per
layer and the step where warmup stops changing it; cosine vs. WSD trained 300 steps and
both stopped at 200, with which model you'd keep; LR sweep at widths 256/512/1024 with
the three minima marked, your pick for width 4,096, and your confidence in it.

Backing figures all present (`fig_bias_correction.png`, `fig_lr_sweep.png`,
`fig_schedules.png`, `fig_update_ratio.png`) plus `adam_lab.ipynb` and `results.txt`.

---

## Session 12 — 32 virtual GPUs, ZeRO 0–3

Portal: `.../assignments/cmtwxt7pb0bqn0fn00f05jfex`

| Field | Points | Source |
| --- | --- | --- |
| GitHub README link (+ ipynb) | 1000 | [session-12/zero-sim/README.md](session-12/zero-sim/README.md) |

- ✅ **Paste this:** `https://github.com/ut101295/era-v5/blob/main/session-12/zero-sim/README.md`

The README is graded on showing **you** understood this, not your agent — explain the
memory/compute shift across ZeRO, ZeRO-1, ZeRO-2, ZeRO-3 in your own voice.
`zero_sim.ipynb` and the three figures are in place.

---

## Session 13 — 20M LLM, 50M tokens, reversibility

Portal: `.../assignments/cmu6k8tf608i509l7m9gaotpr`

| Field | Points | Source |
| --- | --- | --- |
| GitHub README.md (repo must have the ipynb notebooks) | 1000 | [session-13/reversible-llm/README.md](session-13/reversible-llm/README.md) |

- ✅ **Paste this:** `https://github.com/ut101295/era-v5/blob/main/session-13/reversible-llm/README.md`

Three runs required: baseline at a fixed batch size; reversible at the same batch size
(state which variant — mid-point, Euler, etc.); reversible pushed to maximum batch size.
Report **final loss, tokens/s, peak memory**, and other findings for each.
`run_log.txt`, `results.txt`, `fig_training.png`, `fig_memory_vs_depth.png` are present.

---

## Session 14 — Linear model → MoE

Portal: `.../assignments/cmumdrhxw1c0j09l7px6v6d8d`

| Field | Points | Source |
| --- | --- | --- |
| GitHub README Link — **repo MUST have training logs** | 1000 | [session-14/moe-upcycle/README.md](session-14/moe-upcycle/README.md) |

- ✅ **Paste this:** `https://github.com/ut101295/era-v5/blob/main/session-14/moe-upcycle/README.md`

Must show both models **continue to train and reduce loss** after the conversion.
Training logs are a hard requirement and are present:
[session-14/moe-upcycle/training_log.txt](session-14/moe-upcycle/training_log.txt) and
`run_log.txt`, with `fig_moe.png` and `results.txt` alongside.

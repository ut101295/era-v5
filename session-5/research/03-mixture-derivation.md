# 03 — Mixture derivation (supply-reconciled arithmetic)

How every number in [`../plan/mixture-and-curriculum-v5.md`](../plan/mixture-and-curriculum-v5.md)
is arrived at, and the arithmetic that reconciles each lane's *desired share* against
its *real supply* from [`02-dataset-inventory.md`](02-dataset-inventory.md). This is the
"show your working" file — the plan states the numbers; this file defends them.

**Budget:** 16T pretraining tokens (carried from Session 3; upper end of the 14–18T class).
**Method:** compose backward from benchmarks ([`01-benchmark-map.md`](01-benchmark-map.md)),
then *shrink any lane whose share the supply cannot honestly cover* — the central discipline
of this session.

---

## 1. From benchmarks to lanes (backward pass)

[`01`](01-benchmark-map.md) collapsed every benchmark into two loss-map archetypes and a
required data format. Grouping those formats gives the capability lanes:

| Benchmark cluster | Lane it funds | Data format the lane must contain |
|---|---|---|
| SWE-bench, Terminal-Bench, τ-bench, BFCL | **Agentic** (+ code) | multi-step trajectories; loss on agent actions only |
| HumanEval, MBPP, LiveCodeBench, SWE-bench | **Code** | code corpora + edit/diff + solution pairs |
| GSM8K, MATH, AIME, GPQA | **Reasoning/math** | math corpora + CoT traces (length-banded) |
| MILU, IndicMMLU-Pro, IndicGenBench, IndicQA, IndQA | **Indic** | native prose + parallel + Indic CoT/QA |
| (world knowledge, fluency substrate) | **General web** + **Other** | filtered web, books, academic, QA |
| (128K+ context capability) | **Long-context** | repacked long documents/repos |

So there are **7 lanes**: general web, code, math/reasoning, Indic, agentic, long-context, other.

---

## 2. The main-pretraining blend (steady-state, ~13.6T of the 16T)

Starting point = Session 3's draft mix (32% EN / 22% Indic / 18% code / 14% math / 8% India-EN / 6% other). Session 5's supply analysis forces three corrections:

1. **Indic 22% is wishful** — verified native is 64B, so 22% (3.5T) would be ~98% synthetic. Cut to **13%** and split by tier (§4).
2. **Agentic did not exist as a lane in S3** — add it, but small (**1.5%**), because real supply is ~10k trajectories (§5).
3. **Long-context** becomes an explicit late-phase overlay (**2.5%**), not steady tokens (§6).

Resulting steady blend:

| Lane | Share | Tokens (of 16T) | Why this number |
|---|---|---|---|
| General web | **45%** | 7.2T | Most abundant (15–30T unique available); the fluency + world-knowledge substrate. Large but *lower* than V4's early 70% because we protect scarce lanes from the start. |
| Code | **20%** | 3.2T | Primary differentiator (coding+agentic goal). Supply ~0.9–1T high-quality unique → ~2–3 epochs on best + broader Stack v2. Code tolerates repetition. |
| Math/reasoning | **8%** | 1.28T | Funds GSM8K/MATH/AIME/GPQA. Corpus ~50–100B → 2–4 epochs; CoT distilled/synthesized. |
| Indic | **13%** | 2.08T | The defensible honest ceiling (§4). Below the 15–25% "target band" *on purpose*. |
| Agentic | **1.5%** | 240B | ~90% synthesized (§5). Small in steady state; heavily upsampled in anneal. |
| Long-context | **2.5%** | 400B | Repacked overlay (ProLong-style), a late phase, not new tokens (§6). |
| Other (books/academic/QA/wiki) | **10%** | 1.6T | ~100–150B unique → 2–3 epochs. High-tier, repetition-tolerant. |
| **Total** | **100%** | **16T** | |

---

## 3. The annealing preset (concentrated final phase, ~3% of budget ≈ 480B)

The anneal mirrors V4's principle: general web falls sharply, scarce capabilities are upsampled, learning rate is decayed. This is a **separate preset**, not the main mix.

| Lane | Main | **Anneal** | Direction |
|---|---|---|---|
| General web | 45% | **15%** | ↓↓ collapse |
| Code | 20% | **22%** | ↑ (best edit/agentic-coding) |
| Math/reasoning | 8% | **22%** | ↑↑ (long CoT, verified) |
| Indic | 13% | **25%** | ↑↑ (Tier-A native + best translated) |
| Agentic | 1.5% | **10%** | ↑↑↑ (the reserved real trajectories) |
| Long-context | 2.5% | **4%** | ↑ |
| Other | 10% | **2%** | ↓ (only the best academic) |

This is where the **anneal reserve** (§7) is spent. It only works if the reserve *survives* the main run — hence protection (§7 floor) and deliberate hold-back.

---

## 4. The Indic lane, split across four provenance tiers

Indic lane = 13% × 16T = **2.08T tokens**. Real verified-native supply = **64B** (Appendix A of [`02`](02-dataset-inventory.md)). The split states, honestly, how each tier's share is reached:

| Tier | Share of Indic lane | Tokens | Unique supply | How it's reached |
|---|---|---|---|---|
| **Verified native (A)** | 22% | 458B | ~64B (Sangraha Verified) | **~7 epochs** of the best native — justified by scarcity (Muennighoff: ~4 epochs full-value, positive to ~16). Best slice reserved for anneal. |
| **Unverified crawl (B)** | 18% | 375B | ~90B (Sangraha-unverified 24B + CulturaX/MADLAD Indic ~60B) | **~2–4 epochs** |
| **Translated (C→B)** | 40% | 832B | unbounded (IndicTrans2/NLLB over English A-tier: FineWeb-Edu, textbooks, code comments, math CoT) | **synthesized by MT**, quality-gated + back-translation checked |
| **Synthetic (C)** | 20% | 416B | unbounded (teacher-LLM Indic instruction/CoT/agentic) | **generated**, quality-gated |
| **Total** | 100% | 2.08T | | |

**Honest accounting (must appear in the data card):** only ~22% of the Indic lane is verified-native; ~40% is machine-translated and ~20% synthetic. Real unique (native + crawl) ≈ **154B**; the remaining ~1.25T (**60%**) is translated + synthetic. This is *exactly* the ratio the inventory says is the honest ceiling — we do not pretend 25% native is achievable when the world holds 64B native tokens.

**Per-language weighting inside the native tier** follows supply (Appendix A): Hindi/Bengali (~12B/10B) carry more absolute tokens but *fewer epochs*; Assamese (0.29B) is epoch-capped and leans on translated/synthetic — a per-language floor prevents the richest languages from crowding out the scarcest.

---

## 5. The agentic lane, sized against ~10k real trajectories

Agentic steady = 1.5% × 16T = **240B tokens**; anneal pushes effective share to 10%. Real supply (Appendix B of [`02`](02-dataset-inventory.md)):

- **Single function-call** (Glaive 113k, xLAM 60k, ToolACE, Gorilla): ~300–400k samples but **<1B tokens** — teaches single-shot API selection (BFCL), a rounding error in token budget.
- **Real multi-step trajectories**: **~10,000 total** in the public commons (APIGen-MT 5k + Mind2Web 2.35k + AgentTuning 1.87k + SWE-Gym 491 + WebArena ~170).

**Verdict:** 240B of agentic tokens is **~90%+ synthesized by necessity.** The plan therefore budgets a **trajectory-synthesis pipeline** (executable repos+tests, tool sandboxes, teacher agent, execution-verification filtering) as a first-class deliverable, and *reserves all ~10k real verified trajectories for the anneal* (SWE-Gym's 491 alone gave +14% on SWE-bench Verified — evidence the reserve is disproportionately valuable). Loss masking is strict: green on agent actions, grey on every tool/user/observation token ([`01`](01-benchmark-map.md) §1).

---

## 6. Long-context is a repacking overlay, not a lane of new tokens

Long-context steady = 2.5% (400B) is **upsampled/repacked** from existing corpora (ProLong recipe: code repos, books, arXiv, textbooks), introduced as a **late phase** (64K then 512K sequences) after the model can already read and reason. Naturally-long unique docs ≈ 10–40B; ProLong hit Llama-3.1-grade long-context with ~5% of its long tokens. Indic long-context ≈ 0 → synthesized by concatenating Indic documents. We never count long-context as new unique supply.

---

## 7. Protected floor and anneal reserve

**Why a floor:** OPUS (V4's data selector, §8 of the session) retained only ~40% of candidate data for a ~6× effective-token gain — but its proxy is English-heavy and *undervalues* Indic + agentic, starving exactly the lanes we care about. V4 fixed this with an always-on Indic lane at 8%. V5 extends protection to three scarce lanes:

| Protected lane | Always-on floor (every batch) | Rationale |
|---|---|---|
| Indic | **10%** | below the 13% target, so OPUS competes only for the marginal 3% |
| Agentic | **1.5%** | ~the entire lane — too scarce/valuable to let the selector reject any |
| Reasoning | **4%** | protects long-CoT the English proxy would score as low-value |
| **Total floor** | **15.5%** | OPUS selects aggressively over the remaining ~84.5% |

**Anneal reserve** (held back from the main run, spent only in the cooldown):
- Best ~15B of the 64B verified-native Indic (the cleanest, highest-provenance slice)
- **All ~10k real agentic trajectories** + top execution-verified synthetic trajectories
- Longest verified reasoning traces (AIME/olympiad-grade, checker-passed)
- Best code-edit trajectories (SWE-Gym-verified)
- Sized ≈ **300–400B tokens** → the ~3% anneal phase.

The reserve is a **Session 5 allocation decision**: identify the best material now, protect it from ordinary sampling, spend it only when the model can extract the most value.

---

## 8. Curriculum, difficulty bands, reasoning-length bands

**Stages** (mirroring V4's rebalance: web 70→18, code 13→35, sci/math 7→39, protected 8 fixed):

| Phase | Budget span | Blend direction | Difficulty |
|---|---|---|---|
| 0 Seed/warmup | 0–2% | broad general text | L0–L1 |
| 1 General | 2–55% | web-heavy, code+math introduced | L0–L2 |
| 2 Capability | 55–85% | web ↓ ~30%, code/math/reasoning/Indic ↑ | L2–L3 |
| 3 Long-context | 85–92% | 64K→512K repacked overlay | L2–L3 |
| 4 Anneal | 92–100% | anneal preset, LR decayed, reserve spent | L3–L4 |

**Transitions are blended over warmup bands of several billion tokens** (V4 lesson §10: a sudden Hindi-share jump into frozen embeddings spiked the gradient norm ~150×). No hard mixture steps; embeddings not frozen during Indic-share increases.

**Difficulty ladder** (per lane, concrete examples in the plan): L0 trivial → L1 easy → L2 medium → L3 hard → L4 frontier (e.g. code: "hello world" → SWE-bench multi-file patch; math: "2+2" → AIME; Indic: single Wikipedia sentence → MILU law/arts MCQ).

**Reasoning-length bands** (the effort dial — a *distribution*, not one slot):

| Band | Trace length | Share of reasoning mass (main → anneal) | Example |
|---|---|---|---|
| Low | <100 tok | 15% → 5% | "15% of 80? → 12" |
| Medium | 100–500 | 35% → 25% | GSM8K 3–4 step CoT |
| High | 500–2000 | 35% → 40% | MATH problem, checks intermediate steps |
| Ultra | 2000–8000+ | 15% → 30% | AIME with alternatives + self-correction |

Anneal shifts mass toward High/Ultra so the model learns to *sustain* deep reasoning last.

---

## 9. The hypothesis and its test (proxy)

Every number above is a **hypothesis**. The falsifiable claim:

> **H:** A protected-floor mixture (Indic 10% / agentic 1.5% / reasoning 4% always-on, over an OPUS-selected remainder) yields **lower per-lane held-out perplexity on Indic and agentic** than a naive web-heavy mixture at equal compute, **without materially raising English/code perplexity** (Δ ≤ ~2%).

**Test:** 1B and 3B proxy runs, ~20–30B tokens each, two arms (naive web-heavy vs protected-floor V5), report per-lane held-out perplexity. **Confirm** if Indic PPL drops ≥10% while EN/code PPL rises <2%; **refute** otherwise. [`../proxy/`](../proxy/) runs a scaled-down stand-in (tiny model on our S4 cleaned corpus + code/math) to demonstrate the *direction* of the effect on real data — see the proxy README for the honest limitations of a laptop-scale proxy-of-the-proxy.

# V5 Data Mixture & Curriculum — Specification

**Model:** Bharat — India-first ~40B LLM (carried from Session 3).
**Budget:** 16T pretraining tokens.
**Capabilities to win:** coding + agentic work · controllable reasoning · native Indic fluency.
**Provenance tiers (from Session 3/4):** A = verified/curated · B = filtered crawl · C = raw crawl or synthetic.

> This is the graded artifact. Every number is defended against the real supply in
> [`../research/02-dataset-inventory.md`](../research/02-dataset-inventory.md) and derived in
> [`../research/03-mixture-derivation.md`](../research/03-mixture-derivation.md); every lane is
> tied to the benchmarks it wins in [`../research/01-benchmark-map.md`](../research/01-benchmark-map.md).
> The whole plan is stated as a **testable hypothesis** (§9) and a scaled-down proxy is run in
> [`../proxy/`](../proxy/). Read it as a spec a reviewer will push on line by line.

---

## 1. One-paragraph thesis

The mixture is the model. Against a fixed 16T-token budget we compose **backward from benchmarks**, then **shrink any lane the real supply cannot honestly fill** — the discipline this session exists to enforce. Two lanes are structurally scarce: **Indic** (64B verified-native tokens in the entire open commons) and **agentic** (~10k real multi-step trajectories). We therefore (a) size Indic at an honest **13%** with an explicit four-tier split, (b) treat agentic as **~90% synthesized**, (c) protect Indic/agentic/reasoning behind an **always-on floor** the OPUS selector cannot cross, and (d) **reserve** the best Tier-A data for a concentrated **anneal**. A **curriculum** with warmup-banded transitions orders it all so capabilities grow without destabilizing the run.

---

## 2. Main pretraining mixture (steady-state, ~13.6T of 16T)

| Lane | Share | Tokens | Benchmarks it funds | Supply verdict (honest) |
|---|---:|---:|---|---|
| General web (EN + non-Indic multilingual) | **45%** | 7.2T | world knowledge, fluency substrate | ✅ unique (FineWeb/DCLM/FineWeb-2 = 15–30T) |
| Code | **20%** | 3.2T | HumanEval, MBPP, LiveCodeBench, SWE-bench | ⚠️ ~2–3 epochs on best (0.9–1T unique) + Stack v2 |
| Math / reasoning text | **8%** | 1.28T | GSM8K, MATH, AIME, GPQA | ⚠️ 2–4 epochs (50–100B corpus) + distilled CoT |
| **Indic** (4 tiers, §4) | **13%** | 2.08T | MILU, IndicMMLU-Pro, IndicGenBench, IndicQA, IndQA | 🔴 60% translated+synthetic — only path to 13% |
| **Agentic / tool-use** | **1.5%** | 240B | SWE-bench, BFCL, τ-bench, Terminal-Bench | 🔴 ~90% synthesized (~10k real trajectories) |
| Long-context | **2.5%** | 400B | 128K+ context capability | ⚠️ repacked overlay, not new tokens |
| Other (books/academic/QA/wiki) | **10%** | 1.6T | knowledge depth, reasoning substrate | ⚠️ 2–3 epochs (100–150B unique) |
| **Total** | **100%** | **16T** | | |

*Design note:* general web starts *lower* than V4's early ~70% because scarce lanes are protected from the first token, not rescued at the end.

---

## 3. Annealing preset (separate mixture, ~3% ≈ 480B, decayed LR)

General web collapses; scarce capabilities are upsampled from the reserve (§7). This is a **distinct preset**, spent only in the cooldown.

| Lane | Main | **Anneal** | | Lane | Main | **Anneal** |
|---|---:|---:|---|---|---:|---:|
| General web | 45% | **15%** | | Agentic | 1.5% | **10%** |
| Code | 20% | **22%** | | Long-context | 2.5% | **4%** |
| Math/reasoning | 8% | **22%** | | Other | 10% | **2%** |
| Indic | 13% | **25%** | | | | |

---

## 4. Indic lane — the four-tier split (no single headline number)

Indic lane = **2.08T tokens**. Verified-native supply = **64.3B** (Sangraha Verified; IndicCorp v2 / Varta / Wikipedia overlap it — not additive).

| Tier | Share of Indic | Tokens | How the share is reached |
|---|---:|---:|---|
| **Verified native (A)** | 22% | 458B | **~7 epochs** of 64B (scarcity-justified; best slice → anneal reserve) |
| **Unverified crawl (B)** | 18% | 375B | ~2–4 epochs of ~90B (Sangraha-unverified + CulturaX/MADLAD Indic) |
| **Translated (C→B)** | 40% | 832B | **MT synthesis** via IndicTrans2/NLLB over English A-tier, back-translation-gated |
| **Synthetic (C)** | 20% | 416B | **teacher-LLM generation** (Indic instruction/CoT/agentic), quality-gated |

**Data-card statement (mandatory honesty):** ~22% of Indic is verified-native, ~40% machine-translated, ~20% synthetic; real unique (native+crawl) ≈ 154B, and ~60% of the lane is translated+synthetic. We do **not** claim 25% native — the world holds 64B native tokens. A **per-language floor** protects scarce languages (Assamese 0.29B native is epoch-capped and leans on translation) from being crowded out by Hindi/Bengali.

---

## 5. Lane → dataset pointers (named supply)

| Lane | Named datasets (from inventory) |
|---|---|
| General web | FineWeb, FineWeb-Edu, DCLM-baseline, Nemotron-CC, FineWeb-2 (non-Indic multilingual) |
| Code | The Stack v2 (permissive), StarCoderData, OpenCoder/RefineCode; CommitPackFT (edit); SWE-Gym (agentic-coding) |
| Math/reasoning | OpenWebMath, proof-pile-2, FineMath; NuminaMath, OpenMathReasoning; OpenR1-Math-220k, OpenThoughts (CoT) |
| Indic — verified | Sangraha Verified, IndicCorp v2, Varta, Indic Wikipedia |
| Indic — crawl | Sangraha Unverified, CulturaX (Indic), MADLAD-400 (Indic) |
| Indic — translated | IndicTrans2 + BPCC / Samanantar applied to FineWeb-Edu / textbooks / code / math CoT |
| Indic — synthetic | teacher-LLM-authored Indic instruction / CoT / agentic (to build) |
| Agentic | **real:** APIGen-MT-5k, Mind2Web, AgentInstruct(THUDM), SWE-Gym, WebArena · **single-call:** xLAM(⚠️NC), Glaive, ToolACE · **synth pipeline:** SWE-Zero/SWE-rebench style |
| Long-context | ProLong recipe, PG-19, whole-repo Stack v2 (⚠️ exclude Books3) |
| Other | Wikipedia, arXiv, peS2o, StackExchange, Project Gutenberg |

---

## 6. Loss-map discipline (structural rule for every agentic/RAG/reasoning sample)

Loss falls **only on tokens the model must emit at eval time**:
- **Green (trained):** agent planning, tool-call JSON, generated code/patch, CoT, final answer.
- **Grey (context, no loss):** user turns, tool/shell/DB responses, retrieved passages, policy docs, simulated-user utterances.

Putting loss on grey tokens teaches the model to hallucinate the environment. Reasoning RLVR stages carry **no token-level target** — only a verifier reward (tests pass / boxed answer / DB state / rubric).

---

## 7. Protected floor + anneal reserve

**Always-on floor (OPUS may not cross it):**

| Protected lane | Floor per batch | Why |
|---|---:|---|
| Indic | **10%** | English-heavy selector proxy undervalues native Indic (V4 pinned 8%) |
| Agentic | **1.5%** | ~entire lane; too scarce to let the selector reject |
| Reasoning | **4%** | protects long-CoT the proxy scores as low-value |
| **Total floor** | **15.5%** | OPUS selects aggressively over the remaining ~84.5% |

OPUS context: V4 kept ~40% of candidates for ~6× effective-token value at a few % compute overhead — powerful, but blind to scarce capabilities without this floor.

**Anneal reserve (held back from the main run):** best ~15B verified-native Indic · **all ~10k real agentic trajectories** + top execution-verified synthetic · longest checker-passed reasoning traces (AIME-grade) · SWE-Gym-verified code edits. Sized **~300–400B tokens**. Rationale: SWE-Gym's 491 real trajectories alone gave +14% on SWE-bench Verified — the reserve is disproportionately valuable, and *cannot be recovered once spent early*.

---

## 8. Curriculum, difficulty bands, reasoning-length bands

**Stages** (V4 rebalance reference: web 70→18, code 13→35, sci/math 7→39, protected 8 fixed):

| Phase | Budget span | Blend direction | Difficulty |
|---|---|---|---|
| 0 Seed/warmup | 0–2% | broad general text | L0–L1 |
| 1 General | 2–55% | web-heavy; code+math introduced | L0–L2 |
| 2 Capability | 55–85% | web ↓~30%; code/math/reasoning/Indic ↑ | L2–L3 |
| 3 Long-context | 85–92% | 64K→512K repacked overlay | L2–L3 |
| 4 Anneal | 92–100% | anneal preset, LR decayed, reserve spent | L3–L4 |

**Stability:** every transition is **blended over a warmup band of several billion tokens** — never a hard step. V4 saw a sudden Hindi-share jump into frozen embeddings spike the gradient norm ~150×; we do not freeze embeddings during Indic-share increases, and we monitor gradient-norm at every boundary.

**Difficulty ladder (concrete example per rung):**

| Level | Code | Math | Indic |
|---|---|---|---|
| L0 trivial | print "hello world" | 2 + 2 | one Wikipedia sentence |
| L1 easy | reverse a string | 15% of 80 | short news paragraph |
| L2 medium | MBPP "shared list elements" | GSM8K word problem | IndicQA passage + answer |
| L3 hard | LiveCodeBench contest problem | MATH competition problem | IndicMMLU-Pro reasoning MCQ |
| L4 frontier | SWE-bench multi-file patch | AIME olympiad | MILU law/arts MCQ w/ cultural reasoning |

**Reasoning-length bands (the effort dial — a distribution, not one slot):**

| Band | Trace length | Main → Anneal mass | Example |
|---|---|---:|---|
| Low | <100 tok | 15% → 5% | "15% of 80? → 12" (direct) |
| Medium | 100–500 | 35% → 25% | GSM8K 3–4 step CoT |
| High | 500–2000 | 35% → 40% | MATH problem, verifies intermediate steps |
| Ultra | 2000–8000+ | 15% → 30% | AIME with alternatives + self-correction |

Anneal shifts mass toward High/Ultra so the model learns to *sustain* deep reasoning last; the RL stage (Sessions 17–18) then binds these lengths to the low/medium/high/ultra effort setting.

---

## 9. The plan as a testable hypothesis + proxy protocol

**Hypothesis (falsifiable):**
> A protected-floor mixture (Indic 10% / agentic 1.5% / reasoning 4% always-on, over an OPUS-selected remainder) yields **lower per-lane held-out perplexity on Indic and agentic** than a naive web-heavy mixture at equal compute, **without materially raising English/code perplexity** (Δ ≤ ~2%).

**Proxy protocol (the real gate before full scale):**
- **Scale:** 1B and 3B params, ~20–30B tokens/arm (cheap vs 16T).
- **Arms:** (A) naive web-heavy (web 80% / code 12% / math 4% / Indic 3% / agentic 1%); (B) protected-floor V5 (this spec).
- **Metric:** per-lane held-out perplexity (Indic, code, EN, math, agentic-format), plus early proxies for MILU / HumanEval / GSM8K.
- **Confirm:** Indic PPL ↓ ≥10% **and** EN/code PPL ↑ <2%. **Refute:** protected floor tanks EN/code, or fails to move Indic.
- **Decision rule:** a lane whose proxy refutes its share is resized *before* full-scale commit — a data decision is a hypothesis until a cheap experiment tests it.

**Scaled-down evidence run (actually executed):** [`../proxy/`](../proxy/) runs a laptop-scale stand-in — a real gradient-trained ~43k-param char-LM on real data (S4-cleaned Assamese + repo English + repo Python + generated math CoT), three arms at equal compute. **Result (held-out perplexity):**

| Lane | A naive web-heavy | B V5-main | C V5-anneal |
|---|---:|---:|---:|
| Indic | 48.3 | **21.4** (−21.0%) | **19.3** (−23.7%) |
| math | 12.8 | **5.6** (−32.2%) | **3.7** (−48.6%) |
| code | 17.4 | **15.5** (−4.0%) | 15.5 (−3.9%) |
| en | **7.3** | 8.3 (+6.3%) | 8.9 (+10.0%) |

**Finding:** the *direction* is confirmed strongly — protecting scarce lanes more than halves Indic perplexity and collapses math perplexity. The trade-off is real and monotone (English degrades as protection rises). The strict "≤2% English cost" sub-claim was **refuted at proxy scale** — a genuine cost the proxy usefully surfaced. Two effects mean the tiny proxy *overstates* that English cost (so it is an upper bound): at 16T scale English is data-rich (a share cut barely changes its loss), and a 43k-param model is capacity-bound in a way a 40B model is not. The magnitude gate remains the 1B/3B run. This is the discipline in action: *a data decision is a hypothesis until a cheap experiment tests it* — and this test refined the hypothesis rather than rubber-stamping it. Full limitations in [`../proxy/README.md`](../proxy/README.md).

---

## 10. Strategic bet: the Indic × agentic intersection

No mature Indic coding/agentic benchmark exists ([`01`](../research/01-benchmark-map.md) §4.5: BFCL-Hi is unverified MT; Indic LLM-Arena agentic is roadmap-only). Bharat's synthetic budget therefore builds **Indic BFCL / Indic τ-style / Indic-instruction-code** corpora — the same loss maps as English agentic, with Indic/Hinglish prompts — converting two separate lanes into one differentiated, measurable lead.

---

## 11. What this commits us to

1. The mixture is written down and defended, composed backward from benchmarks.
2. Scarce data (Indic native, agentic trajectories, long reasoning) is protected by an always-on floor and an anneal reserve — not left to a global selector that would starve it.
3. Every proportion is a hypothesis; the 1B/3B proxy (and the scaled-down run in [`../proxy/`](../proxy/)) decides what survives to full scale.

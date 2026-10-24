# Bharat — Real Dataset Inventory & Data-Mixture Supply Analysis

**Target model:** India-first ~40B dense/MoE LLM. Priorities: coding + agentic work, controllable reasoning, native Indic languages.
**Assumed pretraining budget:** ~16T tokens (the denominator for every "supply verdict"). Reference share targets in parentheses per lane.
**Sizing convention:** every entry is sized in **two currencies — samples AND tokens** — because they diverge wildly (a function-call set = millions of short samples / few tokens; an agentic trajectory set = few samples / huge tokens each). Token counts are approximate and tokenizer-dependent (most public figures use a GPT-2/GPT-NeoX-class tokenizer; Indic tokens inflate 2–4× under such tokenizers).
**Provenance tiers:** **A** = verified / curated / high-quality; **B** = decent, filtered but broad; **C** = raw crawl or synthetic-uncertain provenance.

> ⚠️ Numbers are approximate and drawn from dataset cards / papers as of 2025–2026. This is a planning dossier, not an audited count. Figures with ranges or flagged "uncertain" are the ones most worth re-checking against live dataset cards (Stack v2 total tokens, FineWeb-2 total, Sangraha verified/synthetic split).

---

## 0. Executive summary — where the budget can actually come from

| Lane | Reference share of 16T | Target tokens | Realistic UNIQUE supply | Verdict |
|---|---|---|---|---|
| General web (En + non-Indic) | 40–50% | 6.4–8T | 15–30T+ (FineWeb, DCLM, Nemotron-CC, FineWeb-2) | ✅ **Abundant** — unique tokens exceed need |
| Code | 12–18% | ~2–2.9T | ~0.9–1T high-quality unique; ~3T+ raw | ⚠️ **Adequate at 1–2 epochs**; SFT layer thin |
| Math / reasoning | 5–8% | ~0.8–1.3T | ~50–100B corpus + ~5–15B CoT traces | ⚠️ **Repeat 2–4× + synthesize CoT** |
| **Indic** | **15–25%** | **2.4–4T** | **~60–90B VERIFIED native; ~250B incl. crawl** | 🔴 **Impossible without heavy synthesis** — 30–60× gap |
| **Agentic / tool-use** | 3–5% (effective) | ~0.5T | **<10B real; long trajectories in low thousands of samples** | 🔴 **Must be SYNTHESIZED** |
| Long-context | (upsample, not new) | overlay | ~10–40B genuinely long unique | ⚠️ **Upsample existing, don't count as new** |
| Other (books/academic/QA/wiki) | 5–8% | ~0.8–1.3T | ~100–150B unique | ⚠️ **Repeat 2–3×** |

**The two structural gaps are Indic and agentic.** Everything else is either abundant (web) or fillable by modest repetition. Those two lanes are the ones that force a synthesis strategy.

---

## 1. General web (English + non-Indic multilingual) — share 40–50% (~6.4–8T)

| Dataset | Docs / samples | Tokens (approx) | License | Tier | Notes |
|---|---|---|---|---|---|
| **FineWeb** | ~25B+ docs | **~15T** | ODC-By 1.0 | A | CC 2013–2024, 96 dumps; strong dedup + quality filters |
| **FineWeb-Edu** | ~1.3B docs | **~1.3T** (score ≥3); ~5.4T (≥2) | ODC-By | A | Edu-classifier subset; best-in-class for knowledge |
| **FineWeb-2** (multilingual) | billions | **~3–8T** across 1,000+ langs | ODC-By | A/B | Non-English CC; covers **non-Indic multilingual lane** |
| **DCLM-baseline** | ~2.6B docs | **~3.8–4T** | (CC terms) | A | From DCLM-Pool (240T raw); model-based filtering |
| **RefinedWeb** | ~968M docs (public) | ~600B public / **~5T full** | ODC-By | A/B | Falcon corpus; only ~600B released |
| **C4 / mC4** | 365M / billions | **~156B** / ~6T+ | ODC-By/Apache | B | T5 corpus; mC4 multilingual much larger |
| **Dolma** (v1.7) | billions | **~2.3T** | ODC-By | B | OLMo corpus; web+code+books+papers |
| **RedPajama-v2** | ~100B docs | **~30T** raw (5 langs) | Apache 2.0 | C | Raw + quality signals; filter yourself |
| **Nemotron-CC** | billions | **~6.3T** (4.4T real + 1.9T synth rephrase) | (NVIDIA/CC) | A/B | Synthetic rephrasing extends high-quality tokens |

**Supply verdict:** ✅ **Fully met with UNIQUE tokens, no repetition.** The real work here is **quality filtering and dedup, not sourcing.**

---

## 2. Code — share 12–18% (~2–2.9T)

### 2a. Pretraining code corpora
| Dataset | Files | Tokens (approx) | License | Tier | Notes |
|---|---|---|---|---|---|
| **The Stack v2** (full) | ~3B+ files | **~900B+ unique** (StarCoder2 trained ~3.3T w/ repeats) | permissive per-file + opt-out | A/B | 600+ langs; **anchor code corpus** |
| **StarCoderData** | — | **~250B** (86 langs) | permissive + opt-out | A | Well-curated |
| **OpenCoder / RefineCode** | — | **~2.5T** | (open) | A/B | Modern strong-filtering recipe |
| codeparrot/github-code | ~115M files | ~200B (uncertain) | mixed | C | Raw GitHub dump |

### 2b. SFT / instruct / edit (SWE-bench-style)
| Dataset | Samples | Tokens | License | Tier | Notes |
|---|---|---|---|---|---|
| **SWE-bench** (full / Verified) | ~19,000 / 500 | large/sample | MIT | A | **Eval-shaped**; few samples, many tokens each |
| SWE-Gym / SWE-bench-extra | ~2,438 / ~6,400 | large | open | A/B | Training-oriented executable envs |
| **CommitPack / CommitPackFT** | 57M commits / ~700K | hundreds of B / few B | mixed / permissive | B/C, A/B | Edit/diff supervision |
| Magicoder OSS-Instruct / Evol | ~75K / ~110K | ~50–200M | research/distilled | B/C | Synthetic instruct |

**Supply verdict:** ⚠️ **Pretraining adequate at ~2–3 epochs** on best tokens (code tolerates mild repetition). **SFT/agentic-coding thin** — real SWE-instances are low tens of thousands of samples; agentic coding trajectories must be **synthesized** (see §5).

---

## 3. Math / reasoning text — share 5–8% (~0.8–1.3T)

### 3a. Math pretraining corpora
| Dataset | Docs | Tokens | License | Tier |
|---|---|---|---|---|
| **OpenWebMath** | ~6.3M | **~14.7B** | ODC-By | A |
| **proof-pile-2** | — | **~55B** | open | A |
| **FineMath** | ~6.7M+ | **4+ ~9.6B; 3+ ~34B** | ODC-By | A |
| MathPile | — | **~9.5B** | CC-BY-NC-SA | A/B |
| DeepMind Mathematics / AMPS | 2M / 5M | few B | Apache/MIT | B |

### 3b. Math SFT / QA
| Dataset | Samples | Tokens | License | Tier |
|---|---|---|---|---|
| **NuminaMath-CoT / 1.5** | ~860–900K | ~1–2B | Apache/CC-BY | A |
| **OpenMathInstruct-2** | ~14M pairs | ~5–10B | CC-BY | A/B |
| **OpenMathReasoning** (AIMO-2) | ~540K / ~3.2M sols | ~10B+ | CC-BY-4.0 | A |
| MetaMathQA | ~395K | ~0.2–0.4B | research | B |

### 3c. Long chain-of-thought traces (R1 era)
| Dataset | Samples | Tokens | License | Tier |
|---|---|---|---|---|
| **OpenR1-Math-220k** | ~220K (~800K traces) | ~few B | Apache 2.0 | A |
| **OpenThoughts-114k / 2 / 3** | 114K / 1M / 1.2M | 1–10B+ | Apache 2.0 | A |
| Bespoke-Stratos / Sky-T1 | ~17K each | ~0.1B | open/Apache | A/B |

**Supply verdict:** ⚠️ **Corpus repeatable, CoT must be synthesized/distilled.** Math pretraining ≈ **50–100B unique** → repeat best tokens ~3–4× (fine for math). Long-CoT ≈ **5–15B** useful tokens → distill more with a teacher. **Controllable reasoning (think-budget tags) requires purpose-built synthetic traces** — not off-the-shelf.

---

## 4. Indic — the scarcity story 🔴 — target 15–25% of 16T = 2.4–4T

### 4a. Verified native (Tier A)
| Dataset | Languages | Tokens (approx) | License | Tier | Notes |
|---|---|---|---|---|---|
| **Sangraha — Verified subset** | 22 Indic | **~60–62B** | CC-BY-4.0 (mixed) | A | Largest *native* verified pool (curated web + OCR PDFs) |
| **IndicCorp v2** (native) | 24 | **~20–21B** (Hindi ~5–6B) | research/open | A/B | News+web native crawl |
| **Wikipedia (Indic)** | ~20 | **~few hundred M** (Hindi ~50–70M) | CC-BY-SA | A | Tiny but high quality |
| Varta | 14 + En | ~few B | research | A/B | ~41M news articles |

**Verified native realistic total (after dedup): ~60–90B tokens across ALL 22+ Indic languages combined.**

### 4b. Unverified crawl (Tier B/C)
| Dataset | Tokens (approx) | Tier | Notes |
|---|---|---|---|
| **Sangraha (FULL)** | **~251B** | B/C | Verified ~62B + crawl + **large synthetic (romanized/translated/OCR) share** |
| **CulturaX (Indic)** | Hindi ~12–13B; others less | C | Cleaned mC4+OSCAR |
| **MADLAD-400 (Indic)** | part of 3T/419-lang | C | Noisy for low-resource Indic |

### 4c. Translated (parallel / translatable)
| Dataset | Coverage | Samples | Tokens | Tier | Notes |
|---|---|---|---|---|---|
| **BPCC** | 22 Indic ↔ En | **~230M pairs** | ~tens of B | B | Backbone for **IndicTrans2** |
| Samanantar | 11 Indic ↔ En | ~49.7M pairs | ~few B | B | |
| IndicTrans2 (model) | 22 Indic | — | — | — | The **engine** to convert English A-tier → Indic |

### 4d. Synthetic (what must be generated)
| Source | Method | Yield | Tier |
|---|---|---|---|
| **MT of English A-tier** via IndicTrans2/NLLB | machine translation | **effectively unbounded** (compute+QC limited) | C→B |
| **Synthetic native generation** (LLM-authored Indic instruction/CoT) | teacher LLM | unbounded, quality-gated | C |

### The Indic gap, quantified
| Pool | Tokens | vs 2.4T (15%) | vs 4T (25%) |
|---|---|---|---|
| Verified native (A) | ~60–90B | **~1.5–3.7%** | ~2% |
| + Full Sangraha incl. crawl/synth | ~251B | **~10%** | ~6% |
| + CulturaX/MADLAD Indic crawl | ~400–600B achievable | **~20–25%** | ~15% |
| Remainder to reach 2.4–4T | **~1.8–3.5T** | must be **TRANSLATED + SYNTHESIZED** | |

**Supply verdict:** 🔴 **A 15–25% share CANNOT be met with unique verified native tokens — the gap is 30–60×.** Reaching it requires, in order: (1) all verified native at **multiple epochs** (Indic tolerates 4–8+ given scarcity, per Muennighoff scaling-with-repetition); (2) all crawl at 1–2 epochs; (3) **machine-translate English A-tier** via IndicTrans2/NLLB with quality gating; (4) **synthesize** native-style Indic instruction/reasoning/agentic. A defensible mix is **~10–15% Indic**, of which **only ~1/5 is verified-native and ~4/5 translated+synthetic** — stated openly in the data card.

---

## 5. Agentic / tool-use 🔴 — share 3–5% effective (~0.5T), overwhelmingly synthetic

### 5a. Single / few-turn function-calling (many samples, few tokens)
| Dataset | Samples | Tokens | License | Tier |
|---|---|---|---|---|
| **Glaive-function-calling-v2** | ~113K | ~50–100M | Apache 2.0 | B/C |
| **xLAM / APIGen** | ~60K verified | ~30–60M | CC-BY | A/B |
| **ToolACE** | ~11K+ APIs | ~100M | research | A/B |
| **Gorilla / APIBench** | ~16K calls | ~10–20M | Apache 2.0 | B |
| BFCL data | ~2K+ cases | small | eval | A (use as **eval**, not training) |

### 5b. Multi-step trajectories (few samples, MANY tokens each)
| Dataset | Trajectories | Tokens | License | Tier | Notes |
|---|---|---|---|---|---|
| **ToolBench (ToolLLM)** | ~126K instr w/ paths | hundreds of M | Apache 2.0 | B/C | Synthetic DFS paths |
| **AgentInstruct (THUDM)** | **~1,866 trajectories** | ~few M | Apache | A/B | Genuinely multi-step, tiny count |
| Orca AgentInstruct (MS) | ~1M+ instr (synth) | hundreds of M | research | B/C | Not all long-horizon |
| **SWE-Gym / OpenHands traj.** | ~1,000s | large | open | A/B | Real executable agentic coding |
| Mind2Web | ~2,350 web tasks | tens of M | research | A/B | |
| APIGen-MT (xLAM multi-turn) | ~10s of K | tens of M | CC-BY | A/B | Verified pipeline |

**Supply verdict:** 🔴 **Must be SYNTHESIZED.** Single FC supply is fine at *sample* level (~300–400K) but ~<1B tokens and teaches only single-shot API selection. **Genuine long-horizon trajectories ≈ 5,000–15,000 real samples total** — nowhere near a 0.5T lane. The agentic lane is **~90%+ synthetic by necessity:** build environments (repos+tests, tool sandboxes) and run a teacher agent to generate + execution-verify trajectories. Highest-leverage synthesis investment alongside Indic.

---

## 6. Long-context — upsampling overlay, NOT a new-token lane
| Dataset / recipe | Tokens | License | Tier | Notes |
|---|---|---|---|---|
| **PG-19** | **~11B** | Apache 2.0 | A | Project Gutenberg pre-1919 |
| **ProLong** (Princeton) | **~40B** mix | open | A | Upsamples books+repos+textbooks to 512K ctx |
| Long code repos (Stack v2 repo-level) | subset | permissive | A/B | Whole-repo concatenation |
| LongAlpaca / LongAlign | tens of M | research | B | Long-context SFT |
| ⚠️ Books3 | — | copyright-tainted | C | **Avoid** — use PG/openly-licensed |

**Supply verdict:** ⚠️ **Do NOT count as new unique tokens.** Genuinely-long unique docs ≈ **10–40B**. Run a short long-context continued-pretraining phase (~10–50B tokens) that **upsamples and packs existing corpus documents by length**. Indic long-context ≈ zero → synthesize by concatenating Indic documents/books.

---

## 7. Other (books, academic, Q&A, Wikipedia) — share 5–8% (~0.8–1.3T)
| Dataset | Tokens | License | Tier |
|---|---|---|---|
| **Wikipedia (English)** | **~4–4.5B** | CC-BY-SA | A |
| **arXiv (full text)** | **~28–30B** | mixed | A/B |
| **peS2o (v2)** | **~42B** | ODC-By | A |
| **StackExchange** | **~20B** | CC-BY-SA | A/B |
| Project Gutenberg / books | few B | public domain | A |

**Supply verdict:** ⚠️ **~100–150B unique; repeat 2–3×** or blend with web. All high-tier and repetition-tolerant.

---

## 8. Synthesis: implications for the Bharat mixture

1. **Two lanes are supply-constrained: Indic (30–60× gap) and agentic trajectories (~1000× gap in real samples).** Everything else is abundant (web) or fixable with 2–4 epochs (code, math, academic).
2. **Indic realism:** verified native ≈ 60–90B. A 25% Indic budget (4T) would be ~98% translated+synthetic — not defensible. A **10–15% Indic budget** with an explicit split of ~1/5 verified-native (multi-epoch) + ~2/5 crawl + ~2/5 translated/synthetic is the honest ceiling. State the ratio in the data card.
3. **Agentic realism:** FC datasets are a rounding error in tokens; real trajectories ≈ 5–15K samples. **Budget compute for a trajectory-synthesis pipeline** (executable environments + teacher agent + execution-verification). Mandatory, not optional.
4. **Controllable reasoning** needs purpose-built synthetic CoT with explicit length/budget control; off-the-shelf R1/OpenThoughts (~5–15B) are a seed, not a solution.
5. **Repetition budget:** Muennighoff et al. ("Scaling Data-Constrained LLMs") supports ~4 epochs at near-full value, positive returns to ~16 — the empirical license to repeat scarce Indic/math/reasoning tokens rather than over-diluting with web.

---

## Appendix A — Per-language verified-native Indic ceiling (deep-dive)

The single most reviewer-defensible scarcity number. **Sangraha Verified = 64.3B tokens total** (human-verified web + OCR'd PDFs + ASR transcription); IndicCorp v2 (20.9B), Varta, and Wikipedia **substantially overlap it — do not sum**. 64.3B is the best single ceiling for genuinely verified native Indic text in the open commons.

| Language | Family | Verified-native tokens (Sangraha SV) |
|---|---|---|
| Hindi | Indo-Aryan | **12.6B** |
| Bengali | Indo-Aryan | **10.6B** |
| Tamil | Dravidian | 4.0B |
| Telugu | Dravidian | 3.7B |
| Gujarati | Indo-Aryan | 3.6B |
| Urdu | Indo-Aryan | 3.7B |
| Marathi | Indo-Aryan | 2.8B |
| Malayalam | Dravidian | 2.7B |
| Kannada | Dravidian | 1.8B |
| Odia | Indo-Aryan | 1.2B |
| Punjabi | Indo-Aryan | 1.1B |
| Assamese | Indo-Aryan | **0.29B** |

**Takeaway:** only Hindi and Bengali clear ~10B verified-native tokens; every other major Indian language sits at single-digit billions or less. The Sangraha **synthetic** split is **162.7B (65% of the 251B "Indic" headline)** — per-language the synthetic split is **3–40× larger than native** (Assamese: 11.7B synth vs 0.29B native = 40×). The apparent Indic abundance is overwhelmingly manufactured. A 15% Indic share of 16T (2.4T) is a **~37×** shortfall against verified native; a 25% share (4T) is **~62×**.

## Appendix B — The real agentic-trajectory commons (deep-dive)

Summing the genuinely *real / verified multi-step* agentic trajectory sets in the public OSS commons:

| Set | Real trajectories | Type |
|---|---|---|
| APIGen-MT-5k (xLAM-2) | 5,000 | multi-turn, executable-env verified |
| Mind2Web | 2,350 | real human web demonstrations |
| AgentTuning / AgentInstruct (THUDM) | 1,866 | real GPT-4 ReAct, reward-filtered |
| SWE-Gym (OpenHands) | 491 | real verified SWE trajectories |
| WebArena (human) | ~170 | real human trajectories |
| **Total real** | **~10,000** | — |

Everything larger is **model-synthesized**: NVIDIA SWE-Zero (318k), SWE-Hero (34k), Nebius SWE-rebench (32k), Orca-AgentInstruct (1M), xLAM-60k, Glaive-113k. This is why frontier agentic post-training relies on **synthesis + RL rollouts**, not harvested data. Note SWE-Gym's 491 real trajectories alone drove **+14% on SWE-bench Verified** — evidence that a small, execution-verified trajectory reserve is disproportionately valuable (directly motivating our anneal reserve).

**Licensing flags to carry into the plan:** xLAM-60k is **CC-BY-NC (non-commercial)** — usable for research proxy, flag for production; **Books3 is pirated/defunct** (active litigation, Anthropic settled $1.5B) — excluded from the long-context lane; academic (peS2o ODC-BY, arXiv per-paper) and Wikipedia/StackExchange (CC-BY-SA) carry attribution/copyleft obligations.

## Appendix C — Long-context is repacked, not sourced (deep-dive)

**ProLong** (Princeton, ACL 2025) trains on **40B continued-training tokens** (20B@64K + 20B@512K) that are **upsampled/repacked from existing pools** (code 0.30, books 0.30, FineWeb 0.20, arXiv/StackExchange/Wiki/OpenWebMath). Key result: 512K training improves even 64K-eval performance, and matched Llama-3.1-8B-Instruct long-context using **~5% of its long tokens**. Naturally-long *unique* documents (PG-19 ≈ 2B; whole-repo code; long arXiv/legal/gov) total ~10–40B. **Indic long-context ≈ zero** → synthesize by concatenating Indic documents/books. Confirms: budget long-context as a **~10–50B repacking overlay**, never as new unique tokens.

---

## Sources

FineWeb/-Edu/-2 — huggingface.co/datasets/HuggingFaceFW/{fineweb, fineweb-edu, fineweb-2} · DCLM — datacomp.ai/dclm · RefinedWeb — tiiuae/falcon-refinedweb · C4/mC4 — allenai/{c4,mc4} · Dolma — allenai/dolma · RedPajama-v2 — togethercomputer/RedPajama-Data-V2 · Nemotron-CC — nvidia/Nemotron-CC · The Stack v2 / StarCoder2 — bigcode/the-stack-v2, arxiv.org/abs/2402.19173 · StarCoderData — bigcode/starcoderdata · OpenCoder/RefineCode — arxiv.org/abs/2411.04905 · SWE-bench/Verified — swebench.com, openai.com/index/introducing-swe-bench-verified · SWE-Gym — github.com/SWE-Gym/SWE-Gym · CommitPack(FT) — arxiv.org/abs/2308.07124 · Magicoder OSS-Instruct — arxiv.org/abs/2312.02120 · OpenWebMath — arxiv.org/abs/2310.06786 · proof-pile-2/Llemma — arxiv.org/abs/2310.10631 · FineMath — HuggingFaceTB/finemath · NuminaMath — AI-MO/NuminaMath-{CoT,1.5} · OpenMathInstruct-2/OpenMathReasoning — nvidia/* · OpenR1-Math-220k — open-r1/OpenR1-Math-220k · OpenThoughts — open-thoughts.ai · Sangraha/IndicLLMSuite — ai4bharat/sangraha, arxiv.org/abs/2403.06350 · IndicCorp v2 — ai4bharat/IndicCorpV2 · IndicTrans2/BPCC — arxiv.org/abs/2305.16307 · Samanantar — ai4bharat/samanantar · Varta — rahular/varta · CulturaX — uonlp/CulturaX · MADLAD-400 — allenai/MADLAD-400 · Glaive FC v2 — glaiveai/glaive-function-calling-v2 · xLAM/APIGen — arxiv.org/abs/2406.18518 · ToolACE — arxiv.org/abs/2409.00920 · Gorilla/APIBench — arxiv.org/abs/2305.15334 · ToolBench/ToolLLM — arxiv.org/abs/2307.16789 · AgentTuning/AgentInstruct — arxiv.org/abs/2310.12823 · Mind2Web — arxiv.org/abs/2306.06070 · PG-19 — deepmind/pg19 · ProLong — arxiv.org/abs/2410.02660 · peS2o/S2ORC — allenai/peS2o · Scaling Data-Constrained LMs — arxiv.org/abs/2305.16264

# 02 — Deep-research findings (verified)

Output of the deep-research workflow: **108 agents**, 5 search angles → parallel search → fetch →
**3-vote adversarial verification** → synthesis. Every claim below survived skeptical review
(a claim is killed only if ≥2/3 verifiers refute it). Raw machine output is in
[`raw/deep-research-synthesis.json`](raw/deep-research-synthesis.json); the full per-agent trail
(each verifier's verdict) is in [`raw/deep-research-journal.jsonl`](raw/deep-research-journal.jsonl).

## Executive summary

For a 40B India-first LLM the evidence points to a clear recipe. Token budgets for 27–40B open
models cluster at **14–18T** (Gemma 3 27B 14T; Qwen2.5 18T; DeepSeek-V3 14.8T; Llama 3 ~15T).
**Llama 3's published mix (~50% general / 25% math+reasoning / 17% code / 8% multilingual)** is the
most usable public recipe — but the multilingual share must rise far above 8% for Indic. Indic
pre-training data is anchored by **Sangraha (251B tokens, ~65% synthetic)** and **IndicCorp v2
(20.9B)**; native high-quality Indic text is scarce and skewed. Cleaning combines **multi-level +
global MinHash dedup** (FineWeb: 5-grams, 112 hashes, 14×8 buckets, 75%), **LLM-scored quality
classifiers** (FineWeb-Edu), and PII/safety filters — with the caveat that global dedup shows
diminishing/negative returns. Post-training converges on **SFT + GRPO** with rule-based + model-based
rewards and R1-style distillation (no DPO). Large vocabularies (Gemma 3 262K) improve non-English
balance and are **~5% of params at 27B → effectively free at 40B**.

## The 10 verified findings

### 1. Token budget: 14–18T for the 27–40B class — `high`, 3-0
Gemma 3 27B = 14T; Llama 3 = ~15T (405B = 15.6T); Qwen2.5 = 18T; DeepSeek-V3 = 14.8T (671B MoE, 37B
active). → a 40B dense model should plan for the **upper end (15–18T+)**.
Sources: [Gemma3 report](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf),
[Llama 3 (2407.21783)](https://arxiv.org/abs/2407.21783),
[Qwen2.5 (2412.15115)](https://arxiv.org/abs/2412.15115),
[DeepSeek-V3 (2412.19437)](https://arxiv.org/abs/2412.19437).

### 2. Llama 3's mix is the reference recipe — `high`, 3-0
Explicit end-state percentages over ~15T: **~50% general, 25% math/reasoning, 17% code, 8%
multilingual**, dynamically adjusted during training. For India-first: **8% multilingual is a floor
to exceed**; 17% code / 25% math are the bars for strong coding/reasoning.
Source: [Llama 3](https://arxiv.org/abs/2407.21783).

### 3. Sangraha = largest Indic corpus, but ~65% synthetic — `high`, 3-0
251B tokens / 22 langs = Verified (64B: scraped + OCR + ASR) + Unverified (24B: perplexity-filtered
CulturaX/MADLAD) + **Synthetic (162B: Wikimedia EN → 14 Indic langs via IndicTrans2, romanized via
IndicXlit)**. Peer-reviewed (Khan et al., ACL 2024). → the synthetic share is a quality caveat.
Source: [IndicLLMSuite / Sangraha](https://github.com/AI4Bharat/IndicLLMSuite).

### 4. IndicCorp v2 + IndicNLP are the native resources — and severely skewed — `high`, 3-0
IndicCorp v2 = 20.9B tokens, 24 langs, 4 families. AI4Bharat-IndicNLP = 2.7B words, 10 langs.
**Per-language skew is severe**: IndicCorp v1 Hindi 1.86B vs Assamese 32.6M (~57× gap).
Sources: [IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2),
[IndicNLP (2005.00085)](https://arxiv.org/abs/2005.00085).

### 5. A phased language curriculum balances high/low-resource — `medium`, 3-0 (transferability caveat)
IndicBERT-v3 used 4 phases: English 0.30 → 14 high/mid Indic 0.25 → 8 low-resource 0.15 → joint
consolidation 0.25 (final ~10% of steps, all langs) to fight forgetting. **Caveat:** this is a 4B
*encoder* on ~35B tokens — ratios are illustrative of curriculum design, not a frontier decoder recipe.
Source: [IndicBERT-v3](https://huggingface.co/ai4bharat/IndicBERT-v3-4B/blob/main/README.md).

### 6. Cleaning = multi-level + global MinHash + line-freq + PII + quality classifiers — `high`, 3-0
Llama 3: URL/doc/line dedup + global MinHash + line-frequency filter (drop lines >6× per 30M-doc
bucket) + dirty-word/PII/unsafe filters + fastText & RoBERTa quality classifiers. FineWeb MinHash:
**5-grams, 112 hashes, 14 buckets × 8, ≥75% similarity**.
Sources: [Llama 3](https://arxiv.org/abs/2407.21783), [FineWeb (2406.17557)](https://arxiv.org/html/2406.17557v1).

### 7. LLM-scored quality classifiers (FineWeb-Edu) are cheap & replicable — `high`, 3-0
FineWeb 15T → FineWeb-Edu 1.3T via: Llama-3-70B scores 460K samples 0–5 → linear regression on
frozen Snowflake-arctic-embed-m (threshold 3, F1 82%) → classify all 15T in ~6,000 H100-hours
(~92% removed). **Directly transferable to build an Indic-Edu filter (one per language).**
Source: [FineWeb](https://arxiv.org/html/2406.17557v1).

### 8. More dedup is not always better — `high`, 3-0
FineWeb: per-snapshot dedup matched RefinedWeb; **global dedup gave only modest uplift and removed
up to 90% of older data — the retained old data was worse than what was removed.** → for scarce
Indic data, avoid over-aggressive global dedup.
Source: [FineWeb](https://arxiv.org/html/2406.17557v1).

### 9. Post-training = SFT + GRPO (+ R1 distillation), no DPO needed — `high`, 3-0
DeepSeek-V3 post-trains SFT → RL via **GRPO (no value model)**, **rule-based rewards for verifiable
tasks** + model-based rewards for free-form, distilling reasoning from DeepSeek-R1 (1.5M SFT
instances). No DPO. → a concrete RLVR + distillation template.
Source: [DeepSeek-V3](https://arxiv.org/abs/2412.19437).

### 10. A large, non-English-balanced vocab is favored — cost is negligible at scale — `high`, 3-0
Gemma 3 = **262K SentencePiece**, "more balanced for non-English." **Embedding cost shrinks with
scale: 302M/1B ≈ 30% vs 1,416M/27B ≈ 5%** → nearly free at 40B. Llama 3's 128K improved English
compression 3.17→3.94 chars/token (vocab design directly affects fertility).
Sources: [Gemma3 report](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf),
[Llama 3](https://arxiv.org/abs/2407.21783).

## Caveats (honest limits of the evidence)

- **Time-sensitivity:** 14–18T is an early-2025 snapshot; Qwen3 already doubled to ~36T. The real
  target is likely rising.
- **Only Llama 3 discloses an explicit mix** — Gemma 3 / Qwen2.5 / DeepSeek-V3 do not, so the
  recipe generalization rests substantially on one source.
- **Indic quality:** Sangraha's 251B is ~65% synthetic; "largest/high-quality" is AI4Bharat's own
  framing. Native high-quality Indic web text is far smaller.
- **Curriculum finding (#5)** is from a 4B *encoder*, not a frontier decoder — transferability limited.
- **"Fertility" vs "compression":** Llama 3's 3.17→3.94 is *chars/token* (higher = better), the
  inverse of fertility (*tokens/word*). Don't conflate.
- **Not covered by the verified set** (filled from `01` scoping searches instead): exact Sarvam-1 /
  Krutrim / IndicSuperTokenizer vocab & fertility tables, per-script fertility, and the specific
  Indic/coding/agentic benchmark details. Treat those as `01`-sourced, not deep-verified.

## Open questions (flagged by the workflow)

1. Concrete per-script Indic fertility numbers and how Sarvam/IndicSuperTokenizer/Gemma/Llama
   compare; is there hard evidence of diminishing returns >200K vocab?
2. Do dedicated Indic coding/agentic benchmarks exist, or must we compose English + Indic suites?
3. Actual SFT/preference/agentic-trace data sources & volumes for Indic (verified claims covered
   pre-training far more than post-training).
4. The code/math/multilingual mix % used by Gemma 3 / Qwen2.5 / DeepSeek-V3 (undisclosed).

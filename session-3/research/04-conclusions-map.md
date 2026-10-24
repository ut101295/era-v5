# 04 — Conclusions map (evidence → decision)

How each decision in [`../model-design-report/index.html`](../model-design-report/index.html) traces
to the research. Use this to audit any claim in the report.

| # | Report decision | Driven by | Confidence |
|---|---|---|---|
| Data — budget | ~16T pre-training tokens (upper end of 14–18T class) | `02` #1 (Gemma/Llama/Qwen/DeepSeek cluster) | high |
| Data — mix | 32% EN web / 22% Indic / 18% code / 14% math / 8% India-EN / 6% other = **36% India+multilingual** | `02` #2 (re-weight Llama 3's 8% multilingual floor) | high |
| Data — scarcity handling | upsample native (3–5 epochs) + IndicTrans2 translation + OCR/ASR; cap synthetic | `02` #3, #4, #8 (Sangraha 65% synthetic; 57× skew; over-dedup harms) | high |
| Data — curriculum | 4-phase: EN/code → +T1 Indic+math → +low-resource → joint consolidation | `02` #5 (IndicBERT-v3 curriculum) | medium (encoder caveat) |
| Post-train — SFT | Indic (native+translated+Hinglish), code, agentic traces, R1-distilled reasoning, India-domain | `02` #9 + `01` (BFCL/agentic) | high |
| Post-train — RL | **GRPO** + verifiable (rule-based) + model-based rewards; no DPO | `02` #9 (DeepSeek-V3 template) | high |
| Alignment | constitutional / India-first values, native-speaker preference data | assignment objective + `02` #9 | reasoned |
| Cleaning — dedup | URL→doc→line + MinHash (5-grams/112/14×8/75%), **conservative global** | `02` #6, #8 (FineWeb/Llama params; global-dedup caveat) | high |
| Cleaning — quality | FineWeb-Edu-style classifier, **one per Indic language** | `02` #7 (transferable pipeline) | high |
| Cleaning — Indic specifics | NFC normalization, script purity, MT-artifact removal, Hinglish romanization | Session 2 lesson + `03` | high |
| Eval — Indic | MILU, IndicMMLU-Pro, IndicGenBench, Indic-QA | `01` (benchmark landscape) | high |
| Eval — code/agentic/math | HumanEval/MBPP/LiveCodeBench/SWE-bench; BFCL/τ-bench/Terminal-Bench; GSM8K/MATH/GPQA | `01` | high |
| Eval — gap | propose **IndiaBench-Code / IndiaBench-Agent** | `01` (no Indic coding/agentic benchmark exists) | reasoned |
| Tokenizer — targets | fertility targets per bucket | `01` (Sarvam 1.4–2.1; generic 4–11) | high |
| Tokenizer — size | **262,144 (2¹⁸)** budget re-pointed to Brahmic + code | `03` derivation + `02` #10 + `01` (>200K plateau) | high |
| Tokenizer — cost argument | embedding ≈ 4–5% at 40B → free | `02` #10 (Gemma 30%@1B → 5%@27B) | high |

## What is *reasoned* vs *sourced*

- **Sourced & verified:** budgets, Llama-3 mix, corpus sizes/skew, cleaning params, GRPO template,
  embedding-cost scaling, Gemma vocab.
- **Sourced (scoping only, not deep-verified):** Sarvam/IndicSuperTokenizer fertility & vocab,
  benchmark specifics — see `02` caveats.
- **Reasoned by us (design choices, not facts):** the exact mix percentages, the vocabulary bucket
  split, the 4-phase ratios adapted to a decoder, and the IndiaBench proposal. These are engineering
  decisions grounded in the evidence, not quoted from a paper.

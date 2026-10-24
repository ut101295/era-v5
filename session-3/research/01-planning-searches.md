# 01 — Scoping searches (anchor numbers)

Before the deep-research pass, I ran targeted web searches to fix the "anchor" numbers the whole
design hinges on. These are summarized here with their sources. (The deep-research pass later
independently re-verified the frontier-model and corpus numbers.)

## Gemma 3 tokenizer & training

- **Gemma 3 vocab = 262,144 (2¹⁸)** SentencePiece, shared with Gemini 2.0 — "more balanced for
  non-English languages." (Gemma 2 was 256,000.) Split digits, preserved whitespace, byte-level.
- **Gemma 3 27B trained on ~14T tokens** (12B→12T, 4B→4T, 1B→2T); data = web + code + math +
  multilingual, with **2× the multilingual data of Gemma 2**.
- Sources: [Gemma 3 report (arXiv 2503.19786)](https://arxiv.org/html/2503.19786v1),
  [Google Developers blog](https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/).

## Indic tokenizer fertility (the core driver)

- **English fertility ≈ 1.4 tokens/word**; generic multilingual models hit **4–8 tokens/word on
  Indic scripts** (Telugu: Llama-3 / Qwen ≈ 10–11 tok/word).
- **Sarvam-1**: custom **68,096-token** Indic vocab → fertility **1.4–2.1** across supported
  languages (Gemma-3 ≈ 2.2 on Telugu).
- **Krutrim**: per-language SentencePiece-BPE up to **100K/language**, >99.7% coverage; MinHash-LSH
  dedup pipeline over petabyte-scale crawl+news+books.
- **IndicSuperTokenizer** (Nov 2025): pre-tokenization + multi-word tokenization beats naive vocab
  scaling; **diminishing returns for vocab > ~200K**; corpus gains plateau past ~10GB.
- Sources: [Sarvam-1](https://www.sarvam.ai/blogs/sarvam-1),
  [IndicSuperTokenizer (arXiv 2511.03237)](https://arxiv.org/html/2511.03237v1),
  [Token fertility explainer](https://medium.com/@biswanai92/understanding-token-fertility-why-it-matters-for-multilingual-llms-38c0b9f20da2).

## Evaluation benchmarks

- **Indic understanding:** MILU — 11 Indic langs, 8 domains, ~80K India-centric MCQs; GPT-4o tops
  it at **74%**. IndicMMLU-Pro, Indic-QA.
- **Indic generation:** IndicGenBench — 29 languages, 13 scripts, 4 families, 5 tasks
  (summarization, MT/Flores-In, QA).
- **Coding/agentic (English):** HumanEval, MBPP, LiveCodeBench, SWE-bench; Terminal-Bench, BFCL,
  τ-bench.
- **Gap:** no standard **Indic coding or agentic** benchmark exists (closest: a Hindi BFCL-Hi
  effort). → motivated the proposed IndiaBench suite.
- Sources: [MILU (AI4Bharat)](https://github.com/AI4Bharat/MILU),
  [IndicGenBench (arXiv 2404.16816)](https://arxiv.org/abs/2404.16816).

## Training-token budgets (~40B class)

- Gemma 3 27B **14T**; the deep pass extended this: Qwen2.5 **18T**, DeepSeek-V3 **14.8T**,
  Llama 3 **~15T** → a 40B model should target the **upper end, 15–18T+**.
- Source: [Gemma 3 report](https://arxiv.org/html/2503.19786v1) (+ frontier reports in `02`).

# Session 3 — "India-first 40B model" design report

## Context

Session 3 is **not a coding assignment** — it's a research + design report (worth 500 pts) judged on *depth of thinking*, with an explicit penalty for length ("longer submissions → lower scores"). The scenario: design the **data + tokenizer strategy** for a **40B model** that rivals "Gemma 4," is excellent at **coding, agentic work, and Indic languages**, and is **India-first** (views the world from an Indian perspective).

The report must answer four questions:
1. **Data** — what to collect and why, split across **pre-training / post-training (SFT) / RL & alignment**.
2. **Data cleaning** — how to clean/filter for those objectives.
3. **Evaluation** — how to test against each objective (Indic, coding, science, math, agentic, India-first).
4. **Fertility & tokenizer size** — target tokens/word per language + for code/science/math/agentic, and **derive the vocab size** from those targets.

Deliverable: a **self-contained `index.html`** report → deployed to **Netlify** → link shared. This matches the Session 1 & 2 pattern (zero-build single-file HTML deployed via Netlify Drop).

## Deliverable & location

- New folder: `session-3/model-design-report/` containing `index.html` (self-contained, no build, no npm).
- Style: tight, dense, table-driven. Target ~1500–2500 words of body text; let tables/diagrams carry the reasoning so we score high on "thought-through per word."
- Reuse the self-contained-HTML approach from `session-1/assignment-playground/index.html` and `session-2/tokenizer-widget/index.html` (inline CSS/JS, dark clean aesthetic).

## Step 1 — Deep research pass (before writing)

Invoke the **deep-research skill** to build a cited evidence base. Target these questions:
- Real data recipes & token budgets for ~30–40B frontier-open models (Gemma 3 = ~14T tokens; Llama 3, Qwen, DeepSeek mixes; % code/math/multilingual).
- Indic corpora sources: Sangraha (AI4Bharat), IndicCorp v2, Bhasha-wiki, Sarvam/Krutrim pipelines, digitized govt/books, transliteration/Hinglish, parallel data.
- Data cleaning at scale: dedup (MinHash/LSH), perplexity/quality classifiers, PII & safety filtering, decontamination, quality weighting.
- Post-training & RL: SFT instruction data (Indic + code + agentic/tool-use traces), preference data, RLHF/RLVR/RLAIF, DPO/GRPO; India-first alignment (cultural values, constitutional data).
- Evaluation: MILU, IndicGenBench, IndicMMLU-Pro, Indic-QA; coding (HumanEval, MBPP, LiveCodeBench, SWE-bench), agentic (Terminal-Bench, BFCL, τ-bench), math/science (GSM8K, MATH, GPQA). Note the **Indic coding/agentic benchmark gap** → propose one.
- Tokenizer fertility: Sarvam-1 (68K, 1.4–2.1), Gemma-3 (262K), IndicSuperTokenizer, per-script fertility, diminishing returns >200K vocab, embedding-parameter cost of large vocab at 40B scale.

Anchors already gathered (to verify/extend): Gemma 3 = 262,144-vocab SentencePiece, ~14T tokens, web+code+math+multilingual; generic tokenizers hit 4–8 tokens/word on Indic vs ~1.4 English; Sarvam-1 hits 1.4–2.1 with a 68K Indic vocab.

## Step 2 — Report structure (the HTML)

1. **Thesis / TL;DR** — one-paragraph statement of the model + the headline tokenizer decision.
2. **Language scope** — tiered set (English + Hindi/Bengali/Marathi/Telugu/Tamil/Gujarati/Kannada/Malayalam/Odia/Punjabi/Assamese/Urdu + Hinglish/romanized), with the *why* (speakers × digital presence × script family), plus code as a first-class "language."
3. **Data strategy** — table with columns: stage (pre-train / SFT / RL-alignment) × source × purpose × rough token/example budget. Emphasize India-first sourcing and the code/agentic/math mix.
4. **Data cleaning** — pipeline diagram: ingest → language ID → dedup → quality/perplexity filter → PII/safety → decontamination → quality weighting; per-objective specifics (code compile/lint filters, math verification, Indic script normalization).
5. **Evaluation** — table: objective × benchmark(s) × pass criteria; call out the Indic-agentic gap and propose a small held-out "India-first" eval.
6. **Fertility & tokenizer size (the quantitative core)** — see Step 3.
7. **One-line trade-off note** per major decision (why not 128K, why not per-language tokenizers, etc.).

## Step 3 — Tokenizer-size derivation (must be shown, not asserted)

Present an explicit budget table so the vocab number is *derived*:
- **Target fertility per bucket** (tokens/word): English ≤1.3; code ~1.3–1.5; high-resource Indic (Hindi/Bengali/Marathi) ~1.7–2.0; Dravidian/others ~2.0–2.5; math/science symbols low.
- **Vocab allocation** across script families to hit those fertility targets, e.g. Latin+code, Devanagari cluster, Dravidian scripts, other Brahmic scripts, math/symbols, reserved — summing to a total.
- **Land on a number** (expected ~200–256K, likely **256K / round 2^18**) and justify against: 128K (too small for 12+ scripts → Indic fertility blows up), >262K (diminishing returns + embedding-parameter cost at 40B), and Gemma's 262K (we bias allocation more toward Brahmic/code). Tie back to the Session 2 finding that `use_regex=False` / script-aware handling matters more than raw vocab.

## Step 4 — Build & deploy

- Write `session-3/model-design-report/index.html`.
- Preview locally: `cd session-3/model-design-report && python3 -m http.server 8000`.
- Deploy: Netlify Drop (drag folder) or `netlify deploy`; capture the URL.
- Update `session-3/session_notes.md` and paste the Netlify link into the assignment.

## Verification

- Open the served page in a browser; confirm all sections render, tables are readable, and the tokenizer budget table arithmetic sums to the stated vocab number.
- Sanity-check every quantitative claim against a cited source from Step 1 (no unsourced numbers).
- Word-count the body; trim to the tight target if over.
- Confirm the four assignment questions are each unambiguously answered.

## Notes / decisions

- No shared build system; keep the report self-contained per repo convention (see `CLAUDE.md`).
- Netlify deployment is outward-facing — I'll confirm the deploy step with you before pushing it live if credentials/manual drop are involved.

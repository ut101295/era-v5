# Session 3 — Research dossier

This folder documents **how the Bharat‑40B design brief was built** — the raw research, the
verified evidence, the reasoning, and the arithmetic — so the conclusions in
[`../model-design-report/index.html`](../model-design-report/index.html) can be traced end‑to‑end.

## The assignment (recap)

Design the **data + tokenizer strategy** for a 40B, "India‑first" model that rivals "Gemma 4" and
is excellent at coding, agentic work, and Indic languages. Answer four questions — data
(pre‑train / post‑train / RL), data cleaning, evaluation, and **fertility → tokenizer vocab size** —
then publish a *concise* report (longer = lower score) on Netlify.

## How the research was done

1. **Scoping searches** (manual, in plan mode) — a handful of targeted web searches to get anchor
   numbers (Gemma vocab, Indic fertility, benchmarks, training‑token budgets). → `01-planning-searches.md`
2. **Deep‑research pass** — a background workflow that fanned out **108 agents**: 5 search angles →
   parallel web search → fetch & extract falsifiable claims → **3‑vote adversarial verification per
   claim** (a claim dies only if ≥2/3 skeptics refute it) → synthesis. Result: **0 refutations** on
   the retained claim set. → `02-deep-research-findings.md`
3. **Derivation** — turned the verified evidence into fertility targets and a vocabulary budget that
   sums exactly to 262,144. → `03-tokenizer-derivation.md`
4. **Write + verify** — wrote the self‑contained HTML report, checked every table's arithmetic and
   that all four questions are answered.

## Files

| File | What it is |
|---|---|
| `01-planning-searches.md` | Initial anchor findings from scoping web searches |
| `02-deep-research-findings.md` | The 10 verified findings + caveats + open questions (human‑readable) |
| `03-tokenizer-derivation.md` | The fertility→vocab reasoning and the budget arithmetic |
| `04-conclusions-map.md` | Which evidence drove which decision in the final report |
| `raw/deep-research-synthesis.json` | Machine output of the workflow (findings, sources, caveats) |
| `raw/deep-research-journal.jsonl` | **Raw artifact** — one line per agent (searches, claims, each verifier's verdict) |
| `raw/implementation-plan.md` | The approved plan the report was built from |

## The bottom-line conclusions

- **Budget:** ~16T pre‑training tokens (upper end of the 14–18T open‑frontier class).
- **Mix:** re‑weight Llama‑3's recipe toward India — ~36% India/multilingual (vs Llama‑3's 8%),
  while holding code (18%) and math (14%) high.
- **Cleaning:** FineWeb/Llama‑style dedup + per‑language quality classifiers, but *conservative*
  global dedup because Indic data is scarce.
- **Post‑training:** SFT + **GRPO** with verifiable (rule‑based) + model‑based rewards, R1‑style
  reasoning distillation; no DPO required.
- **Evaluation:** MILU / IndicGenBench / IndicMMLU‑Pro (Indic) + HumanEval/SWE‑bench (code) +
  BFCL/τ‑bench (agentic); propose **IndiaBench‑Code / ‑Agent** to fill a real benchmark gap.
- **Tokenizer:** **262,144 (2¹⁸)** — same size as Gemma 3, budget re‑pointed from CJK to
  Brahmic scripts + code; embedding cost ≈ 4–5% of params at 40B, effectively free.

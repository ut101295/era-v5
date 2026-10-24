# Session 4 — Research dossier

This folder documents **how the Session 4 cleaning work was researched and built** —
the strategy count, the dataset investigation, the per-stage design decisions, and the
measured results — so everything in [`../cleaning-widget/`](../cleaning-widget/) and
[`../pipeline/`](../pipeline/) can be traced back to evidence rather than assertion.

The narrative version (read top-to-bottom) is [`../WALKTHROUGH.md`](../WALKTHROUGH.md).
This folder is the **artifact** version — the individual research outputs.

## The assignment (recap)

Work with your agent to (1) find how many **cleaning strategies** the session lists,
(2) pick a real **10–100M dataset** (biased toward the Session 3 corpora), (3) actually
**apply the cleanups**, and (4) build a **widget** covering the strategies, the dataset,
what was cleaned/why/how, other concerns, and final statistics — then deploy to Netlify.

## How the research was done

1. **Read the whole session** and enumerated the cleaning strategies from three
   independent places to defend a number. → `01-strategy-count.md`
2. **Probed the HuggingFace hub live** to find a real in-band Indic dataset from the
   Session 3 family and confirm its structure. → `02-dataset-selection.md`
3. **Designed one pipeline stage per strategy**, getting the Indic-specific details
   right, and recorded the design decisions. → `03-pipeline-design.md`
4. **Ran the pipeline for real**, debugged three Indic-bias bugs, and captured the
   measured before/after numbers, extra defects, and the reproducibility proof.
   → `04-run-findings.md`

## Files

| File | What it is |
|---|---|
| `01-strategy-count.md` | The enumeration of cleaning strategies and the defended count (8) |
| `02-dataset-selection.md` | Live HF hub probes, the candidate landscape, and why Assamese was chosen |
| `03-pipeline-design.md` | Per-stage design decisions and the Indic-specific correctness details |
| `04-run-findings.md` | Measured results, the three-bug debugging log, extra defects, determinism proof |
| `raw/dataset-probes.md` | Raw output of the HuggingFace probing commands |
| `raw/fetch-summary.json` | What `fetch_data.py` actually pulled |
| `raw/stats.json` | The full measured pipeline output (copy of the widget's data) |

## Bottom-line answers

- **Strategies:** **8** — normalization · format discipline · quality filtering ·
  deduplication · language validation · PII removal · decontamination · manifest.
- **Dataset:** AI4Bharat **IndicCorpV2 / Assamese** (~10M tokens) + a 300-doc Bengali
  mislabel probe + 2,000 Hindi instruction pairs.
- **Result:** 111,759 → 48,526 docs; 10.06M → 9.53M tokens; deterministic content hash;
  an English-tuned filter would have dropped **100%** of the good Assamese corpus.

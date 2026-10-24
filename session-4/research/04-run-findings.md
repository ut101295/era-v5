# 04 — Run findings (measured)

The pipeline was run for real on the Assamese corpus. This file records the measured
results, the debugging that got there, the extra defects, and the reproducibility proof.
The full machine output is in [`raw/stats.json`](raw/stats.json).

## The first run was wrong — three Indic-bias bugs (in my own code)

The first full run removed **98.9% of tokens**, which is absurd. The striking part:
all three bugs were *the exact Indic-bias traps the session warns about*, committed by
the cleaner itself.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Quality dropped 41,594 good docs as "symbol walls" | `not c.isalnum()` counts Indic **combining marks** (matras/viramas) as symbols | Classify symbols by `unicodedata.category` (marks = content) |
| 2 | ~9,800 genuine Assamese docs flagged as Bengali | Compared ব/র counts, which exist in **both** scripts | Disambiguate on the Assamese-only ৰ/ৱ |
| 3 | 82% of the corpus flagged "contaminated" | Used the 50 **longest** docs as eval with 8-grams → fingerprints covered common phrasing | Small random eval set + **13-grams** |

Two smaller bugs: the **canary** contains `::` which the tokenizer splits, so it must be
checked against raw text (not tokens); and a 300 KB `keep_mask` array was being
serialized into `stats.json` (dropped it → 828 KB → 11.6 KB).

## The clean run — per-stage numbers

Runtime **~37 s**. Docs flow:

| Stage | docs in → out | key measured detail |
|---|---|---|
| Normalization | 111,759 → 111,759 | 1,752 Brahmic joiners **kept**; 37 garbage chars removed; 1 HTML entity |
| Format discipline | 2,000 convos | **5,000 ghost markers** across 4 formats unified → 1; 0 leaks in pretraining |
| Quality filtering | 111,759 → 51,062 | 57,011 too-short/empty; 3,642 no-terminal-punct; avg edu 0.86 |
| Deduplication | 51,032 → 50,748 | **284 near-dups; 262 cross-shard** (global-only), 22 local; 43.6 MB index |
| Language ID | 50,748 → 50,452 | **246 injected Bengali caught**; 50 genuine false-positives |
| PII removal | 50,452 → 50,452 | 174 structured masks (156 phones, 13 URLs, 4 emails, 1 ID); name layer 73% FP |
| Decontamination | 50,452 → 48,526 | 1,926 contaminated removed; **canary planted → detected ✓** |
| Manifest | 48,526 docs | 9,525,589 tokens; content-hash id; **deterministic ✓** |

## Headline statistics

- **111,759 → 48,526 documents** (56.6% removed)
- **10,058,642 → 9,525,589 tokens** (5.3% removed — the dropped docs are mostly
  empty/short, so they hold few tokens; this is the honest shape of the reduction)
- **5,000** ghost markers unified · **284** near-dups (**262 cross-shard**) ·
  **246** mislabels caught · **1,926** contaminated docs removed

## The single most important result — filter bias quantified

An **English-tuned quality filter would have dropped 100% (51,062 / 51,062)** of the
good Assamese documents this pipeline kept. This is the session's warning made concrete:
*"filtering is not a neutral cleaning step, it is a decision about which languages the
model will be able to speak."*

## Reproducibility proof

Two back-to-back runs (`PYTHONHASHSEED=0`) produced the **identical** content hash:

```
run1 = 8a4100c95affc433f56831a16ea4cb6bf396adf9177554650ecf967c67d608b6
run2 = 8a4100c95affc433f56831a16ea4cb6bf396adf9177554650ecf967c67d608b6
DETERMINISTIC ✓
```

## Extra defects caught (the honest gallery)

1. **Half the raw crawl is empty** — 55,729 of 111,759 rows (49.9%) are blank.
2. **Doc-clean but char-dirty** — like Dolma, IndicCorpV2 is a good corpus at the
   document level, yet the normalizer still found HTML entities, replacement characters,
   and control characters to strip.
3. **Internal repetition = leakage risk** — 1,926 docs shared a 13-word passage with a
   30-doc held-out sample, i.e. real train/eval contamination the firewall removes.

## PII caveat (reported honestly, not hidden)

The name layer masked 33,207 spans of which **24,325 (73%) are false positives** —
place-names and common words like ভাৰত (India) / অসম (Assam). This is deliberately
surfaced as the precision/recall tension the session describes, and the widget separates
"structured PII" (174 clean masks) from the lossy name layer so the headline isn't
misleading.

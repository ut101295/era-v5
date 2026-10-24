# 01 — How many strategies? (the first deliverable)

The assignment's first task is to "find in total *how many strategies* are listed in
this session." The session frames data cleaning as **an ordered pipeline, not a menu**
(§2: *"the stages are not a menu to pick from, they are an ordered sequence where each
one depends on the one before it"*). So each pipeline stage is a cleaning *strategy*.

I enumerated them from **three independent places** in the session notes so the count
is defensible rather than a guess.

## (a) The narrative sections §3–§11 + the widgets

| # | Stage / strategy | Session § | Widget | V4 defect it fixes |
|---|---|---|---|---|
| 1 | **Extraction** (raw page → clean prose) | §2 | widget_1 | *known from Session 3; treated as given* |
| 2 | **Normalization** (NFC, strip invisibles/controls, unescape HTML, collapse whitespace; keep ZWNJ/ZWJ) | §3 | widget_2 | 46 garbage tokens in the vocab |
| 3 | **Format discipline / ghost-tag unification** (one canonical special-token conversation format) | §4 | widget_3 | P0: 4 sources, 4 formats, ghost markers in pretraining |
| 4 | **Quality filtering** (heuristic cascade → edu-value classifier; script-aware) | §5 | widget_4 | filter bias against low-resource languages |
| 5 | **Deduplication — the mechanism** (shingles → MinHash → LSH) | §6 | widget_5 | Indic crawl had *no* dedup |
| 6 | **Deduplication — at scale** (global cross-shard dedup on one large-memory machine) | §7 | widget_6 | duplication is global; local passes miss it |
| 7 | **Language identification & validation** (detect at runtime, don't trust the folder) | §8 | widget_7 | Telugu 2-char vs 3-char code silent near-miss |
| 8 | **PII removal** (regex for structured IDs + ML for names) | §9 | widget_8 | personal data + legal usability |
| 9 | **Decontamination / held-out firewall** (fingerprint eval, scan, remove, canaries) | §10 | widget_16 | train/test leakage inflates scores |
| 10 | **Reproducibility & the manifest** (content-hash IDs + provenance manifest) | §11 | widget_9 | copy-pasted file sizes, non-deterministic IDs, wrong Indic token counts |

That is **10 distinct pipeline stages** if you count extraction and split deduplication
into its two sections.

## (b) The closing "commitments" (§14) — the canonical list

§14 enumerates the strategies explicitly:

> *"…the pipeline stages we built today, **normalization, format discipline, quality
> filtering, deduplication, language validation, PII removal, decontamination, and the
> manifest**, are the standing definition of what turns collected data into training
> data…"*

That is a list of exactly **8**. Here deduplication is counted **once** (the two
sections are one strategy seen at two scales), and **extraction is excluded** because it
was already built in Session 3 and is "treated as known here" (§2).

## (c) The Minor (§12) — two operational artifacts

§12 adds two things that are part of the machinery but are **not cleaning transforms**:

1. the **live cleaning-numbers tracker** (who has shipped how many clean tokens), and
2. the **central dedup machine** (a provisioned, single-owner, large-memory,
   checkpoint-resumable instance that runs the one global dedup pass).

## The number I defend: **8**

The canonical, session-authored list in §14 names **8 cleaning strategies**:

> normalization · format discipline · quality filtering · deduplication ·
> language validation · PII removal · decontamination · manifest / provenance

I present this as the headline answer, with the honest nuance that:

- it is a **10-stage pipeline** if you add **extraction** (Session 3) and split
  **deduplication** into its local-mechanism and global-scale halves (the session gives
  each its own section and its own widget), and
- there are **2 operational artifacts** from the Minor (collection tracker + central
  dedup machine).

The widget headlines "**8 strategies**" and then *shows* the full 10-stage pipeline and
the 2 artifacts, so the reasoning behind the number is visible rather than asserted.

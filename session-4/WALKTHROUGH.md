# Session 4 — Data Cleaning & Deduplication · Walkthrough

> This document is the running narrative of how I read the Session 4 assignment,
> what it actually asks for, how I researched it, what dataset I picked, how I
> cleaned it, and how the final widget was built and deployed. It is written to be
> read top-to-bottom as one continuous story. Nothing is abbreviated — later
> sections are appended as the work happens, so the "flow" of the project is
> preserved and auditable end to end.
>
> The **artifact** version of this research (individual dossier files, raw probe
> outputs, measured stats) lives in [`research/`](research/) — see
> [`research/README.md`](research/README.md) for the index. This document is the
> narrative; that folder is the evidence.

---

## Part 0 — Where this session sits

Session 3 answered a *strategy* question: for a 40B, India-first model
("Bharat-40B"), **what** data to collect, how much, which data buys which
capability, and what tokenizer size (we landed on 262,144). Its output lives in
[`../session-3/model-design-report/index.html`](../session-3/model-design-report/index.html)
with a full research dossier in [`../session-3/research/`](../session-3/research/).

Session 3 *ended* on an honest gallery of the defects our own audit caught in a
pipeline that still shipped a 120B model. **Session 4 is where each of those
defects becomes a stage we build correctly.** The one sentence to hold onto:

> **Raw data is not training data.** A web page, an Indic crawl, or a dump of
> instruction pairs is *raw material*. It only becomes a training corpus after it
> has been extracted, normalized, filtered, deduplicated, scanned for
> contamination, and stamped with a manifest of where it came from — and doing
> all of that *reproducibly*.

So Session 4 is the **engineering** counterpart to Session 3's **strategy**.

---

## Part 1 — What the assignment actually asks

The assignment (verbatim, from [`assignment.md`](assignment.md) and §13 of the
session notes) has four moving parts:

1. **Count the strategies.** "Work with your agent, and find in total *how many
   strategies* are listed in this session." → We must read the whole session,
   enumerate the cleaning strategies/stages, and defend a number.

2. **Pick a real dataset.** "Find a **10–100M** dataset like
   [this](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled),
   and then apply these cleanups." The linked example is a *reasoning-distilled
   conversation* dataset. The "10–100M" is a size band (rows or tokens — we treat
   it as tokens/rows in the tens-of-millions range, i.e. a dataset small enough
   to actually download and clean, large enough for the stats to mean something).
   "Find interesting datasets, **specially from the Session 3** that you may have
   come across" → bias toward the Indic / India-first corpora we already
   researched (Sangraha, IndicCorp v2, IndicNLP, FineWeb-Edu, Dolma).

3. **Actually apply the cleanups.** Not describe them in the abstract — *run*
   them on the chosen dataset and measure what happens.

4. **Build a widget and deploy it.** A self-contained page (same house style as
   Session 2's `tokenizer-widget` and Session 3's `model-design-report`) that
   explains:
   - how many strategies there are, and what they are (described)
   - what dataset was picked (and why)
   - what was cleaned, why, and how
   - any *other* strategy or concern that was cleaned up
   - **final statistics** (before/after)

   Then **upload to Netlify** and share the link.

Grading note carried from Session 3's rubric: *"our evaluation is based on how
much you have thought through."* For Session 4 the equivalent signal is
**showing real, reproducible before/after numbers** from a real pipeline, not a
hand-wave. That pushes us toward actually running the cleaning code.

---

## Part 2 — Counting the strategies (the first deliverable)

The session presents the pipeline as **an ordered sequence of stages, not a menu
to pick from** (§2). Each stage is a cleaning *strategy*. I enumerated them from
three independent places in the notes so the count is defensible:

**(a) The narrative sections (§3–§11):**

| # | Stage / strategy | Widget | V4 defect it fixes |
|---|---|---|---|
| 1 | **Extraction** (raw page → clean prose) | widget_1 (pipeline map) | *known from Session 3; treated as given* |
| 2 | **Normalization** (Unicode NFC, strip control chars, unescape HTML, collapse whitespace — while **keeping** ZWNJ/ZWJ in Brahmic scripts) | widget_2 clean_text_live | 46 garbage tokens in vocab |
| 3 | **Format discipline / ghost-tag unification** (one canonical conversation format with real special tokens) | widget_3 special_tokens_ghost_tags | P0: 4 sources, 4 formats, ghost markers in pretraining |
| 4 | **Quality filtering** (heuristic rules → trained edu-value classifier; script-aware for Indic) | widget_4 quality_filter_cascade | filter bias against low-resource langs |
| 5 | **Deduplication — the mechanism** (shingles → MinHash → LSH near-dup detection) | widget_5 minhash_lsh_simulator | Indic crawl had *no* dedup |
| 6 | **Deduplication at scale** (global cross-shard dedup on one large-memory machine) | widget_6 global_dedup_scale | duplication is global; local passes miss it |
| 7 | **Language identification & validation** (detect language at runtime, don't trust the folder path) | widget_7 language_id_validation | Telugu 2-char vs 3-char code silent near-miss |
| 8 | **PII removal** (regex for structured IDs + ML for names; precision/recall tension on Indic names) | widget_8 pii_scrubber | personal data + legal usability |
| 9 | **Decontamination / held-out firewall** (fingerprint eval sets, scan shards, remove overlaps, plant canaries) | widget_16 contamination_heldout | train/test leakage inflates scores |
| 10 | **Reproducibility & the manifest** (deterministic content-hash IDs + provenance manifest per shard) | widget_9 lineage_manifest_builder | copy-pasted file sizes, non-deterministic IDs, wrong Indic token counts |

**(b) The closing "commitments" (§14)** enumerates the canonical list
explicitly: *"normalization, format discipline, quality filtering,
deduplication, language validation, PII removal, decontamination, and the
manifest"* → **exactly 8 strategies.** (Here deduplication is counted once, and
extraction is treated as already-known from Session 3.)

**(c) The Minor (§12)** adds two *operational* artifacts that are not cleaning
transforms but are part of the machinery: the **live cleaning-numbers tracker**
and the **central dedup machine** (a provisioned, single-owner, large-memory,
checkpoint-resumable instance).

### The answer I will defend

> **8 core cleaning strategies** (§14's canonical list):
> normalization · format discipline · quality filtering · deduplication ·
> language validation · PII removal · decontamination · manifest/provenance.
>
> Presented honestly with the nuance that the pipeline has **10 stages** if you
> (a) count **extraction** (carried over from Session 3) and (b) split
> deduplication into its **local mechanism** and **global-scale** halves — which
> is how the session itself devotes two separate sections + two widgets to it.
> Plus **2 operational artifacts** from the Minor (collection tracker + central
> dedup machine).

The widget will state "**8 cleaning strategies**" as the headline number and show
the full 10-stage pipeline + 2 operational artifacts so the reasoning behind the
number is visible rather than asserted.

---

## Part 3 — The dataset shortlist (candidates)

The example dataset is a **reasoning-distilled conversation** set. Session 3
surfaced a specific family of Indic / open corpora. Good candidates in the
10–100M band that let us *actually demonstrate* the cleaning strategies:

| Candidate | Why it fits | What cleaning it exercises well |
|---|---|---|
| **AI4Bharat Sangraha** (verified subset, per-language) | The flagship Indic corpus from Session 3; ~65% synthetic so quality/dedup matter | normalization (Brahmic joiners!), quality filter bias, dedup, language-ID mislabels |
| **AI4Bharat IndicCorp v2** (single language slice, e.g. Telugu/Hindi/Assamese) | Native Indic web text, severely skewed per Session 3 | Indic normalization, dedup, language validation (the Assamese-is-really-Bengali case) |
| **A reasoning-distilled conversation set** (like the linked example) | Directly mirrors the assignment's example; conversation data | **ghost-tag / format discipline**, PII in transcripts, decontamination |
| **FineWeb-Edu sample** | The quality-classifier recipe from Session 3 | quality filtering cascade, dedup, PII |
| **Dolma sample** | §3 calls out Dolma as "clean at doc level, dirty at char level" | normalization (char-level garbage tokens), dedup |

**Trade-off to decide:** an **Indic corpus** best demonstrates the *sovereign
thread* (Brahmic joiners, script-aware filters, language-ID mislabels, Indic PII
false-positives) which is the session's emphasis — but a **conversation/reasoning
dataset** best demonstrates the **ghost-tag / format-unification** strategy,
which is the P0 finding and the most visually compelling widget.

A strong compromise: pick **one primary dataset** (an Indic web slice, to honor
"specially from Session 3") and, if it lacks conversation structure, add a small
**secondary conversation sample** purely to demonstrate the ghost-tag stage. This
is captured as an open decision in Part 6.

---

## Part 4 — How the cleaning will actually be run

The plan is to build a small, real, **reproducible Python pipeline** under
`session-4/pipeline/` that ingests the chosen dataset and runs the 8 strategies,
emitting a `stats.json` of real before/after numbers that the widget loads. This
mirrors how Session 2's widget loaded a real `tokenizer.json`. Each stage:

1. **Normalization** — `unicodedata.normalize("NFC", …)`, strip C0/C1 control
   chars **except** ZWNJ (U+200C) / ZWJ (U+200D), unescape HTML entities,
   collapse whitespace. Count garbage chars removed; assert joiners preserved.
2. **Format discipline** — detect literal conversation markers
   (`[INST]`, `<|user|>`, `### Human:`, `<start_of_turn>`, …) and rewrite to one
   canonical special-token format. Count ghost markers found/fixed.
3. **Quality filtering** — heuristic cascade (mean word length, symbol ratio,
   line-ending punctuation, duplicate-line fraction, stop-word presence) with
   **script-aware thresholds** for Indic, then a cheap length/heuristic proxy for
   the edu-value classifier. Count docs dropped per rule.
4. **Deduplication (local)** — shingles (word 5-grams) → MinHash signatures →
   LSH banding (FineWeb params: 112 hashes, 14 bands × 8 rows, ~75% threshold).
   Count near-duplicate clusters collapsed.
5. **Deduplication (global note)** — since one dataset ~ one shard, we document
   why global dedup needs one large-memory machine and simulate a two-shard
   cross-overlap to show the effect (mirrors widget_6).
6. **Language ID & validation** — run a detector (e.g. `fasttext` lid.176 or
   `langdetect`) per doc, compare against the claimed language, flag mismatches.
7. **PII removal** — regex for emails / phones / IPs, plus a name pass; report
   precision/recall tension and Indic false-positives.
8. **Decontamination** — fingerprint a small held-out eval set, n-gram scan the
   corpus, remove overlapping docs, plant a canary string.
9. **Manifest** — emit a provenance JSON per shard with a **content hash**
   computed *after* cleaning, cleaning-script hash, token count, language
   breakdown, license — and prove determinism by re-hashing.

Every stage prints `docs in → docs out → tokens in → tokens out` so the
"surviving-token bar" in the widget is backed by real numbers.

---

## Part 5 — The widget & deployment

- **Location:** `session-4/cleaning-widget/index.html` — self-contained, zero
  build, loads a `stats.json` produced by the pipeline (same pattern as
  `session-2/tokenizer-widget`).
- **House style:** match Session 2/3 (dark, premium, single SPA).
- **Content sections:** (1) the 8 strategies described; (2) the dataset picked +
  why; (3) what was cleaned, why, how — one card per stage with real numbers;
  (4) "other concerns" (the Minor: tracker + dedup machine, and any extra defect
  we caught); (5) final statistics — before/after tokens, docs, garbage removed,
  dups collapsed, PII masked, contamination removed.
- **Deploy:** Netlify Drop (drag the folder), same as Session 2. Share the link.
  Corpus/stats live *outside* any `data/` folder so they escape `.gitignore`
  (Session 2 learned this).

---

## Part 6 — Open decisions (to confirm before building)

1. **Primary dataset** — Indic web slice (Sangraha/IndicCorp, honors "from
   Session 3" + sovereign thread) vs. a reasoning/conversation set (mirrors the
   linked example + best ghost-tag demo). Possibly one primary + one small
   secondary conversation sample.
2. **Execution depth** — run a *real* Python pipeline that downloads and cleans
   the dataset and emits real stats (recommended, matches the grading signal),
   vs. a lighter representative-sample approach.
3. **Scope of "other strategy/concern"** — how much of the Minor (collection
   tracker, central dedup machine) and any extra self-found defect to include.

### Resolved (confirmed with the user)

1. **Primary dataset = an Indic web slice** (Sangraha / IndicCorp v2, honoring
   "specially from Session 3" and the sovereign thread) **+ a small secondary
   conversation sample** purely to demonstrate the ghost-tag / format-discipline
   stage. Together they exercise all 8 strategies.
2. **Execution depth = real pipeline, real stats.** We download the data and run
   a real Python pipeline for all 8 stages, emitting genuine before/after numbers
   that the widget loads. No simulation of the headline numbers.
3. **"Other concerns" scope = the full Minor + an extra self-found defect.** The
   widget covers the collection tracker and the central dedup machine *and*
   surfaces at least one extra defect our own pipeline run catches — mirroring the
   session's "honest gallery" spirit.

---

## Part 7 — The concrete build plan

**Step 1 — Choose & fetch the exact datasets.**
- Primary: a single-language Indic slice in the 10–100M band from **Sangraha** or
  **IndicCorp v2** (e.g. Telugu or Assamese — Assamese lets us reproduce the
  "Assamese folder is really Bengali" language-ID case). Fetch via the HF
  `datasets` API (streaming to cap size).
- Secondary: a few thousand rows from a reasoning/instruction conversation set to
  demo ghost tags.
- Write a `fetch_data.py` (mirrors Session 2) that is deterministic and records
  what it pulled.

**Step 2 — Build `session-4/pipeline/` (real, reproducible).**
One module per strategy (Part 4 spec), a `run.py` orchestrator that streams docs
through all 8 stages in order and writes `stats.json` (per-stage docs/tokens
in→out) + a sample of struck-out characters, ghost markers, dup clusters, PII
masks, and one full provenance manifest. Cleaning happens **before** the content
hash. Determinism proven by re-running.

**Step 3 — Capture the extra defect.** Whatever our own run surfaces (e.g. a
mislabeled language slice, a garbage-token family, a dedup surprise) is logged
honestly into the walkthrough and the widget's "other concerns" panel.

**Step 4 — Build `session-4/cleaning-widget/index.html`.** Self-contained SPA in
the Session 2/3 house style, loading the real `stats.json`. Sections: 8
strategies described · dataset picked + why · what/why/how per stage with real
numbers · other concerns (tracker + dedup machine + extra defect) · final
before/after statistics. Stats live outside any `data/` folder so they commit.

**Step 5 — Deploy to Netlify Drop** and share the link; append it to Part 8.

---

## Part 8 — Execution log

This is the real, blow-by-blow record of building and running the pipeline.

### 8.1 Environment

The machine runs **Python 3.14.5** — new enough that many ML wheels don't exist yet.
I created `session-4/.venv` and found that `datasets==5.0.1` (with `pyarrow` 25 and
`numpy` 2.5.1) *does* install cleanly, so fetching via the HuggingFace `datasets` API
works. Every cleaning stage was then written against the **standard library only**
(`unicodedata`, `html`, `hashlib`, `re`, `zlib`) plus numpy for the MinHash inner
loop, so the pipeline is portable and reproducible with almost no dependencies.

### 8.2 Choosing the exact datasets

I probed the HuggingFace hub live:

- The dataset the assignment links as an example
  (`lordx64/Qwen3.6-…-Opus-Reasoning-Distilled`) **does not exist** on the hub — it's
  illustrative. So I chose from the Session 3 family instead.
- `ai4bharat/IndicCorpV2` exposes each language as a **split**: `asm_Beng`, `ben_Beng`,
  `hin_Deva`, `tel_Telu`, … This is a gift: Assamese (`asm_Beng`) and Bengali
  (`ben_Beng`) **share the Bengali script block**, which is exactly the language-ID
  confusion the session describes. I picked **Assamese** as the primary corpus so the
  language-ID stage would have a genuine, hard case.
- `ai4bharat/indic-align` config `Dolly_T` gives real **Indic instruction pairs**
  (per-language columns), which I used for the ghost-tag stage so even the conversation
  sample honors the India-first theme.

A 5,000-row throughput probe of `asm_Beng` immediately surfaced a defect:
**~50% of the raw rows are empty** (2,500 of 5,000 blank), and each non-empty row is a
short web sentence (~42 words). That became "extra defect #1".

### 8.3 Fetching (`fetch_data.py`)

I sized the fetch to land in the assignment's **10–100M token band** while staying
laptop-runnable: stream `asm_Beng` until ~10M tokens or 400k rows, whichever first.
Result:

- **111,459** raw Assamese rows ≈ **10,000,072 tokens**, 49 MB — right at the band's
  lower edge.
- **300** Bengali docs from `ben_Beng`, deliberately stamped `claimed_lang="asm"` (the
  mislabel probe).
- **2,000** Hindi instruction pairs from `Dolly_T`.

### 8.4 Building the eight stages

One module per strategy (see [`pipeline/README.md`](pipeline/README.md)), each written
to mirror the session's description and — importantly — to get the **Indic details**
right: normalization keeps ZWNJ/ZWJ; quality rules are script-aware and treat the danda
as sentence punctuation; language-ID disambiguates Assamese from Bengali on the
Assamese-only letters ৰ/ৱ; PII has an Indic false-positive probe. `run.py` streams the
corpus through the stages in order and writes `stats.json`.

### 8.5 The first run was wrong — and the bugs were instructive

The first full run removed **98.9% of tokens**, which is absurd. Debugging surfaced
three bugs, and the striking thing is that all three were *the exact Indic-bias traps
the session warns about — committed by my own code*:

1. **Quality filter counted Indic combining marks (matras/viramas) as "symbols."**
   The naive `not c.isalnum()` symbol test treats Bengali vowel signs as punctuation,
   so 41,594 perfectly good Assamese docs were flagged as "symbol walls." Fixed by
   classifying symbols with `unicodedata.category` (marks count as content).
2. **Assamese/Bengali disambiguation was backwards.** I compared counts of ব/র (which
   exist in *both* scripts) — so genuine Assamese was misread as Bengali en masse. The
   real signal is the presence of the Assamese-only ৰ/ৱ. Fixed.
3. **Decontamination used the 50 *longest* docs as the eval set with 8-grams**, whose
   fingerprints covered so much common phrasing that 82% of the corpus looked
   "contaminated." Fixed to a small random eval set with **13-grams** (the GPT-3
   convention).

I also hit a **canary bug**: the canary string contains `::`, which the tokenizer
splits, so checking it against the *tokenized* text failed. Fixed by checking the raw
text. And I removed a 300 KB `keep_mask` array that was being serialized into
`stats.json` by accident (shrank it from 828 KB → 11.6 KB).

### 8.6 The clean run (final numbers)

After the fixes, the pipeline runs in **~37 s** and produces sensible, honest numbers:

| Stage | docs in → out |
|---|---|
| Normalization | 111,759 → 111,759 (1,752 Brahmic joiners **kept**; 37 garbage chars removed) |
| Format discipline | 2,000 convos → **5,000 ghost markers** across 4 formats unified → 1 |
| Quality filtering | 111,759 → 51,062 (57,011 too-short/empty; avg edu 0.86) |
| Deduplication | 51,032 → 50,748 (**284 near-dups; 262 cross-shard**, invisible to local passes) |
| Language ID | 50,748 → 50,452 (**246 injected Bengali caught**, 50 genuine FP) |
| PII removal | 50,452 → 50,452 (174 structured masks; name layer 73% Indic FP — the tension, shown) |
| Decontamination | 50,452 → 48,526 (1,926 contaminated removed; **canary planted → detected ✓**) |
| Manifest | 48,526 docs, 9,525,589 tokens, content-hash id, **deterministic ✓** |

**Headline:** 111,759 → **48,526 docs** (56.6% removed), 10.06M → **9.53M tokens**
(5.3% removed — the dropped docs are mostly empty/short, so they hold few tokens).

The single most powerful result is the **filter-bias probe**: an English-tuned quality
filter would have dropped **100%** (51,062/51,062) of the good Assamese docs this
pipeline kept. That is the session's warning — "filtering is a decision about which
languages the model can speak" — made concrete in one number.

**Reproducibility** was verified for real: two back-to-back runs produced the
*identical* content hash
(`8a4100c95affc433f56831a16ea4cb6bf396adf9177554650ecf967c67d608b6`).

### 8.7 The widget

[`cleaning-widget/index.html`](cleaning-widget/index.html) is a self-contained SPA in
the Session 2/3 house style that loads the real `stats.json`. It has: an interactive
pipeline map (click a stage → surviving-docs bar collapses + stage detail), the eight
strategies described with live numbers, the dataset rationale, per-stage cards, the
"other concerns" section (collection tracker + central dedup machine + the extra
defects), a final-statistics KPI grid and before/after tables, and the provenance
manifest with the reproducibility proof. I verified it renders by driving it headless
in Chrome and confirming the dynamic values (raw doc count, content hash, cross-shard
sentence, the 100% bias readout, determinism ✓) all populate from `stats.json`.

### 8.8 Deploy

The widget folder is self-contained (`index.html` + `stats.json`), so deployment is a
drag of `cleaning-widget/` onto <https://app.netlify.com/drop>. The Netlify CLI is not
installed locally and a deploy publishes under a Netlify account, so the actual upload
is the one manual step — the resulting URL is the submission link and gets pasted here.

**Netlify link:** _(to be added after upload)_

### 8.9 Extra defects caught (the honest gallery)

1. **Half the raw crawl is empty** — 55,729 of 111,759 rows (49.9%) are blank; pure
   wasted tokens only the pipeline removes.
2. **The corpus is clean at the document level but not the character level** — like
   Dolma, IndicCorpV2 is a good corpus, yet the normalizer still found HTML entities,
   replacement characters, and control characters to strip (small counts, but real).
3. **Internal repetition** — 1,926 docs shared a 13-word passage with a 30-doc held-out
   sample, i.e. real train/eval leakage risk the decontamination firewall removes.

# 03 — Pipeline design decisions

One module per strategy, deliberately dependency-light (stdlib + `datasets` for
fetching + numpy for the MinHash inner loop). Below is *why* each stage is built the way
it is, with emphasis on the **Indic-specific correctness details** that separate a
cleaner built for these languages from a generic English one.

## Stage 2 — Normalization (`normalize.py`)

- **Order:** HTML-unescape → NFC → strip noise/controls → collapse whitespace.
- **The sovereign detail:** the zero-width **non-joiner (U+200C)** and **joiner
  (U+200D)** are *kept*, while every other invisible (ZWSP, BOM, bidi overrides,
  replacement char, soft hyphen) is stripped. ZWNJ/ZWJ carry real linguistic meaning in
  Brahmic scripts; a cleaner that strips all invisibles mangles Indic text while
  believing it's helping. The stage counts joiners kept as a correctness assertion.
- All special characters are written as escapes/named constants, never invisible
  literals in source.

## Stage 3 — Format discipline (`formats.py`)

- Renders each real instruction pair into one of **four incompatible source formats**
  (`[INST]…`, `<|user|>…`, `### Human:…`, `<start_of_turn>…`), simulating "four sources,
  four formats," then **unifies** to one canonical special-token schema.
- Also scans the **pretraining** corpus for leaked conversation markers (the session's
  "ghost markers in pretraining shards" finding).

## Stage 4 — Quality filtering (`quality.py`)

- Heuristic cascade (min length, mean word length, symbol ratio, duplicate lines, script
  purity, terminal punctuation) → an educational-value gate.
- **Script-aware by construction:** the Indic danda (।) counts as terminal punctuation,
  and — the detail that bit me in the first run — **combining marks (matras/viramas,
  Unicode category M\*) count as content, not symbols.** The naive `not isalnum()`
  symbol test treats Bengali vowel signs as punctuation and wrongly flags good text.
- **Bias probe:** every kept Indic doc is *also* run through an English-tuned variant to
  count how many good docs an English filter would have dropped. (Answer: 100%.)

## Stages 5 & 6 — Deduplication (`dedup.py`)

- **Mechanism:** word-5-gram shingles → **112**-permutation MinHash → LSH with **14
  bands × 8 rows** (the FineWeb parameters cited in Session 3; implied similarity
  threshold ≈ (1/14)^(1/8) ≈ 0.72). Union-find rolls candidate pairs into clusters;
  the lowest-index member survives.
- **Global/at-scale:** each doc is assigned a pretend "student shard"; the dedup reports
  how many duplicates were **cross-shard** (caught only by a global pass) vs within-shard
  (a local pass would catch). This makes the "duplication is global" point measurable.
- **Reproducibility fix:** shingles are hashed with `zlib.crc32`, a *stable* hash —
  Python's built-in `hash()` is per-process randomized and would make the corpus
  irreproducible, defeating Stage 10.

## Stage 7 — Language ID & validation (`langid.py`)

- Script-based detection over Unicode blocks (no fasttext build needed — and script
  detection is ideal for Indic, where languages map to distinct blocks).
- **Assamese vs Bengali:** both share the Bengali block, so a block check can't separate
  them. The disambiguation is on the **Assamese-only letters ৰ (U+09F0) / ৱ (U+09F1)**,
  which Bengali never uses — their presence ⇒ Assamese, their absence within the block ⇒
  Bengali. (The first run compared ব/র, which exist in *both* — a bug; see `04`.)
- The V4 **Telugu 2-char↔3-char code bug** is fixed by a `normalize_lang_code` that
  **fails loud** on an unmappable code instead of silently falling back.

## Stage 8 — PII removal (`pii.py`)

- Regex layer: emails, phones (incl. +91 forms), IPv4, URLs, Aadhaar-like IDs — these
  mask cleanly and safely.
- Name layer: a deliberately aggressive Indic name list that *also* contains place-names
  and common words (ভাৰত/অসম/গুৱাহাটী…), so the **precision/recall tension** shows up as
  real false-positive numbers rather than an abstract caveat.

## Stage 9 — Decontamination (`decontam.py`)

- Fingerprints a held-out eval set as **word-13-grams** (the GPT-3 contamination
  convention — long enough that an overlap means a genuine shared passage), removes any
  training doc that overlaps, and plants a **canary** string to detect future leaks.
- The eval set is held out **before** dedup (a sourcing-time firewall), and is a small
  random sample, not the longest docs (using the longest docs over-contaminates — see
  `04`).

## Stage 10 — Manifest & reproducibility (`manifest.py`)

- Content-addressed IDs (a hash of the *cleaned* text, not a running counter), the
  content hash computed **after** cleaning, the exact cleaning-script hash recorded, and
  an **unsafe/unknown license blocks the shard**.
- Determinism is proven by recomputing the hash in-run and (externally) by two
  back-to-back runs producing the identical hash.

## Orchestration (`run.py`)

Streams the corpus through the stages **in the session's order**
(normalize → format → quality → dedup → langid → PII → decontam → manifest), tracking
docs-in/out and surviving tokens at each step, and writes `stats.json` for the widget.
Run with `PYTHONHASHSEED=0` for byte-for-byte reproducibility.

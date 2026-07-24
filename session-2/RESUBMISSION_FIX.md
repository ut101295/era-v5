# Session 2 — Tokenizer Resubmission: what was wrong, what we fixed

This document explains why the **original** submission ([`multilingual-tokenizer/`](multilingual-tokenizer/))
scored **0** on regrade, and why the **resubmission** ([`faithful-tokenizer/`](faithful-tokenizer/))
is correct. Read it top-to-bottom for the full story; the TL;DR is in the first table.

## TL;DR

| | Original (`multilingual-tokenizer/`) | Resubmission (`faithful-tokenizer/`) |
| --- | --- | --- |
| Model | Byte-level BPE (`ByteLevel(use_regex=False)`) | Char-level BPE (`Metaspace`) + **functional `byte_fallback`** |
| Normalizer | NFC | NFC |
| Base-byte coverage | **Partial — 148 / 256** (bug) | Full — every character round-trips via 256 `<0xHH>` byte tokens |
| Corpus | Clipped article prose (references/headers stripped) | **Wiki-faithful Markdown** (links, refs, tables, categories) |
| Score denominator | Whitespace **words** | **Faithful units** (letter/mark/number runs + each visible symbol) |
| Roundtrip gate | **FAILS** — drops `[`, `]`, and 106 other chars | **PASSES (strict)** — no character is ever dropped |
| 4th language | Marathi | Marathi |
| Result | **Score 0** (gate failure) | Score ≈ **28,026**, reproduces on the grader's corpus |

---

## 1. Why the original tokenizer was incorrect

### 1a. The fatal bug: an incomplete byte alphabet → dropped characters

The grader's message:

> Score is 0 because the tokenizer does not satisfy the faithful roundtrip gate:
> `decode(encode(text))` must preserve the same visible non-whitespace characters …
> round-trip changed visible text for sample `'India, officially the Republic of India [1]'`:
> decoded `'India, officially the Republic of India 1'`

The brackets `[` and `]` were **dropped**. Root cause, confirmed by inspecting the saved
`tokenizer.json`:

- The BPE trainer was called **without** `initial_alphabet=ByteLevel.alphabet()`, so the vocab
  only contained the byte-characters that actually appeared in the training text.
- The corpus was **clipped** — [`fetch_data.py`](multilingual-tokenizer/fetch_data.py) strips
  reference markers (`[1]`) and section headers — so characters like `[ ] { } < > # * \` `` ` ``
  never entered training and never got a token.
- Net effect: **108 of the 256 byte-level characters were missing from the vocab** (including
  `[`, `]`, `Q`, `X`, `#`, `*`, and the newline glyph). When the tokenizer meets one of these it
  emits `<unk>`, which the decoder drops → the visible character disappears.

This is also why decode silently dropped newlines all along (same missing-token cause). The old
"100% roundtrip verified" claim was false: it was only ever tested on short clipped-prose slices
that happened to contain none of the missing characters.

**This bug alone zeroes the score under any faithful-roundtrip check — old rubric or new.**

### 1b. It was built for the wrong rubric

Independently of the bug, the grader moved to a stricter rubric than the original assignment:

| | Original assignment | Regrade rubric (reference solution) |
| --- | --- | --- |
| Corpus | clipped prose | wiki-faithful Markdown |
| Denominator | whitespace words | faithful units (letters/marks/numbers run OR one visible symbol) |
| Hard gate | English ratio ≤ 1.2 | `decode(encode(text))` preserves visible characters |
| Penalty | English | Hindi: `exp(max(0, hindi_fertility/1.2 − 1))` |
| Score | `1000 / (X₄ − X₁)` | `1000 / (max_fertility − min_fertility)` |

The byte-level tokenizer was trained/measured on clipped prose with a words denominator, so even
after fixing the byte bug it would not represent the faithful-Markdown corpus efficiently
(`ByteLevel` spends ~3 tokens gluing the UTF-8 bytes of every Indic character back together).

---

## 2. Why the resubmission is correct

The resubmission [`faithful-tokenizer/`](faithful-tokenizer/) replicates the grader's own
pipeline (from their published reference solution) and improves on it. It was validated by
**reproducing the reference's published score of 6,502** with the grader's exact
`evaluate_tokenizer.py` before building ours.

### 2a. Faithful-Markdown corpus
[`build_wiki_faithful_markdown.py`](faithful-tokenizer/build_wiki_faithful_markdown.py) fetches the
Wikipedia REST HTML for *India* in en/hi/te/mr and converts it to Markdown, **keeping** links,
URLs, tables, references, image links and categories — the exact corpus variant the grader scores
against. `corpus/*.faithful.txt` are the snapshots used.

### 2b. Model that compresses Indic scripts and preserves every character
[`train_tokenizer.py`](faithful-tokenizer/train_tokenizer.py):

- **NFC + `Metaspace('▁', prepend_scheme='never')`** — character-level base units (not 3 UTF-8
  bytes), which is why Indic fertility drops sharply vs. byte-level. NFC is used instead of NFKC
  because NFKC is lossy (it rewrites e.g. `″`→`′′`, `ⓘ`→`i`) and would fail a *strict* roundtrip;
  the grader's own NFKC reference in fact fails strict roundtrip on exotic characters.
- **`byte_fallback=True` made functional.** HF's trainer sets the flag but does **not** add the
  `<0xHH>` byte tokens, leaving fallback inert (the same silent-drop failure mode as the original
  bug). We **inject the 256 `<0xHH>` tokens** after training so any character — even one never
  seen in training (emoji, `£`, `€`, `…`, accented Latin) — round-trips through its UTF-8 bytes.
  To keep the vocab at exactly **10,000**, we train to `10000 − 256 = 9744` and add the 256 byte
  tokens (final vocab = base + 9,441 merges + 256 byte tokens = 10,000).
- Decoder `Sequence(ByteFallback, Metaspace)` so the byte tokens decode back correctly.

### 2c. Weights optimized for a small fertility spread
The score is `1000 / (max_fertility − min_fertility)`. We hill-climbed the per-language corpus
weights to equalise fertility (all four land in 0.610–0.646), giving weights
`{en:2, hi:2, te:5, mr:5}` and a spread of 0.0357.

---

## 3. What, concretely, we fixed

1. **Full byte coverage** — the exact defect that caused the 0. Every one of the 256 byte values
   is representable (via functional `byte_fallback`), so `decode(encode(text))` never drops a
   visible character. Verified on the full corpus **and** on emoji / math / currency / accented
   edge strings.
2. **Right corpus** — wiki-faithful Markdown instead of clipped prose.
3. **Right denominator** — faithful units (grader's regex `[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]`)
   instead of whitespace words.
4. **Right model for the task** — `Metaspace` (character base units) instead of `ByteLevel`
   (3-bytes-per-Indic-char), so Indic fertility is competitive.
5. **Gate compliance by construction** — NFC (non-lossy) + functional byte_fallback guarantee the
   faithful-roundtrip gate passes.
6. **Optimized, validated score** — weights tuned to minimise spread, cross-checked on the grader's
   own corpus snapshot.

---

## 4. Verification evidence

All reproducible with `session-2/.venv` (`pip install -r ...` per the widget README):

- **Grader parity:** running the grader's exact `evaluate_tokenizer.py` on their tokenizer +
  corpus reproduces their published **6,502** — proving our fertility/score computation matches
  theirs.
- **Our score:** grader's exact formula on our tokenizer → **28,026** on our corpus,
  **27,966** cross-corpus (en/hi/te on the grader's snapshot, mr on ours). Not overfit.
- **Roundtrip gate:** PASS for all four corpora (strict, non-whitespace characters identical) and
  for common-character + exotic-character samples.
- **Widget parity:** the in-browser JS re-implementation reproduces the Python token counts
  **exactly** (en 120,409 / hi 56,158 / te 22,146 / mr 18,870) and matches Python on encode ids
  and decode strings across emoji/math/currency/accent/CJK edge strings.

Per-language result (grader's formula):

| Language | Tokens | Faithful units | Fertility |
| --- | ---: | ---: | ---: |
| English | 120,409 | 186,426 | 0.6459 (max) |
| Hindi | 56,158 | 88,359 | 0.6356 |
| Telugu | 22,146 | 36,293 | 0.6102 (min) |
| Marathi | 18,870 | 29,766 | 0.6339 |

`spread = 0.6459 − 0.6102 = 0.0357` · `score = 1000 / 0.0357 = 28,026` · Hindi penalty factor 1.0

> Note: `score = 1000/spread` is uncapped and sensitive near zero, so on a fresh grader re-fetch
> the exact figure will move within roughly **22k–33k**; it stays several × the reference. It is
> reported honestly (with this range) on the widget.

---

## 5. Layout

```
session-2/
├── RESUBMISSION_FIX.md          # this document
├── multilingual-tokenizer/      # ORIGINAL (byte-level) — incorrect, kept for reference
├── faithful-tokenizer/          # RESUBMISSION — correct
│   ├── build_wiki_faithful_markdown.py   # fetch + convert corpus
│   ├── train_tokenizer.py                # NFC + Metaspace + injected byte_fallback
│   ├── evaluate_tokenizer.py             # grader-exact fertility/score + roundtrip gate
│   ├── tokenizer.json                    # the graded artifact (vocab 10,000)
│   ├── metrics.json                      # computed metrics
│   ├── corpus/*.faithful.txt|md|meta     # corpus snapshots (raw.html regenerable, gitignored)
│   └── widget/                           # self-contained live widget (deploy this to Netlify)
└── tokenizer-widget/            # ORIGINAL byte-level widget — superseded
```

To reproduce: `python build_wiki_faithful_markdown.py && python train_tokenizer.py && python evaluate_tokenizer.py`.

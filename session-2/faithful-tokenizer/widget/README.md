# Faithful Multilingual BPE Tokenizer — Widget

A single, self-contained, zero-dependency web widget for the resubmitted Session 2
BPE tokenizer. It loads `tokenizer.json` and `corpus/*.faithful.txt` and **re-runs
the tokenizer live in the browser** (a JS port of HuggingFace `tokenizers`) to compute
every number on the page — nothing is hard-coded.

## What it shows

- **Per-language fertility table** — tokens, faithful units, and fertility for
  English / Hindi / Telugu / Marathi.
- **Score calculation** — sorted fertilities, spread = `max − min`,
  `score = 1000 / spread`, the Hindi penalty `exp(max(0, hi/1.2 − 1))`, and the
  adjusted score.
- **Roundtrip gate** — proves `decode(encode(text))` preserves every non-whitespace
  character for all four corpora plus grader-style common-character samples.
- **Download** button for `tokenizer.json` (the graded artifact) and the vocabulary.
- **Live playground** — paste text and see the pre-tokens/tokens, counts, and the
  roundtrip result.

## The tokenizer (HuggingFace standard format)

- **Model:** BPE, `byte_fallback: true`, `unk_token: [UNK]`, vocab size 10,000.
- **Normalizer:** NFC (canonical, non-lossy).
- **Pre-tokenizer:** Metaspace, replacement `▁` (U+2581), `prepend_scheme: never`.
- **Decoder:** `Sequence([ByteFallback, Metaspace(▁, never)])`.

Note: the vocab contains all **256 `<0xHH>` byte tokens**, so `byte_fallback` is
**functional** — any symbol not in the vocab is emitted as its UTF-8 bytes and
round-trips exactly on decode (emoji, math letters, £/€/…, accents, CJK). `[UNK]`
never appears for real text. Vocab of 10,000 = base + 9,441 merges + 256 byte tokens.

## Authoritative numbers (recomputed live; must match)

| Lang | Tokens | Faithful units | Fertility |
|------|-------:|---------------:|----------:|
| en   | 120409 | 186426 | 0.6459 |
| hi   |  56158 |  88359 | 0.6356 |
| te   |  22146 |  36293 | 0.6102 |
| mr   |  18870 |  29766 | 0.6339 |

`spread = 0.035678`, `score = 28026.4`, Hindi penalty `1.0`, vocab 10000 / merges 9441 / 256 byte tokens.

## Preview locally

`fetch()` is blocked on `file://`, so serve over HTTP:

```bash
cd session-2/faithful-tokenizer/widget
python3 -m http.server 8000
# open http://localhost:8000/
```

## Verify the JS engine matches Python exactly

```bash
cd session-2/faithful-tokenizer/widget
node verify.mjs                 # per-language token counts vs targets + score
node verify.mjs --emit-stats    # (re)write stats.json
```

`engine.mjs` is the shared BPE engine (source of truth); the identical code is inlined
into `index.html`. The engine has been cross-checked against Python `tokenizers`
(`session-2/.venv`) on all four corpora and 200+ random/synthetic strings — encode ids
and decode strings match exactly.

## Deploy to Netlify

The folder is fully static. Either:

- **Netlify Drop** — drag the `widget/` folder onto <https://app.netlify.com/drop>, or
- **Netlify CLI** — `netlify deploy --dir=. --prod` from inside `widget/`.

No build step, no dependencies. Keep `corpus/` alongside `index.html` so the live
re-computation works.

## Files

- `index.html` — the widget (self-contained, all CSS + JS inline).
- `engine.mjs` — shared JS BPE engine (dev/verification; mirrored inline in the HTML).
- `verify.mjs` — Node harness that reproduces the target counts and score.
- `stats.json` — authoritative computed stats (written by `verify.mjs --emit-stats`).
- `tokenizer.json` — the graded artifact.
- `metrics.json` — saved training metrics (shown as a cross-check).
- `corpus/*.faithful.txt` — the four evaluation corpora.

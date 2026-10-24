# 03 — Tokenizer size derivation

The assignment's most quantitative ask: pick target **fertility** (tokens/word) per language + for
code/science/math/agentic, and **derive the vocabulary size** from those numbers. This is the
reasoning, shown — not asserted.

## Why fertility is the objective

Fertility = tokens per word. Lower fertility means: fewer tokens per sentence → **cheaper training
& inference, faster generation, and longer effective context** for the same window. Generic
tokenizers are catastrophic on Indic — 4–11 tokens/word vs ~1.4 for English (`01`, `02`) — which is
the single biggest reason "just fine-tune an English model" fails for India.

## Step 1 — set target fertility per bucket

Targets are set *below* generic tokenizers but *realistic* per script (Dravidian scripts are
agglutinative → intrinsically higher, even with a great tokenizer).

| Bucket | Generic (Llama/Qwen) | Target | Note |
|---|--:|--:|---|
| English | 1.3 | ≤1.15 | reasoning backbone |
| Code | ~1.5 | ≤1.4 | indentation/identifiers |
| Math / symbols | high (digit-split) | ~1.3 | LaTeX + split digits |
| Hindi, Marathi (Devanagari) | 3–4 | 1.6–1.8 | largest user base |
| Bengali, Gujarati, Punjabi, Odia | 4–6 | 1.8–2.0 | |
| Tamil, Telugu, Kannada, Malayalam | 8–11 | 2.0–2.4 | agglutinative |
| Urdu (Perso-Arabic) | 3–4 | ~1.9 | distinct script |

Benchmark anchor: Sarvam-1 reaches 1.4–2.1 across Indic with only a 68K *Indic-only* vocab (`01`).
We want comparable Indic fertility **while also** covering English + code + math in one shared vocab,
which requires a larger total.

## Step 2 — allocate a vocabulary budget to hit the targets

Fertility drops as a script gets more merges, but with **diminishing returns past ~200K**
(IndicSuperTokenizer, `01`). So allocate deliberately by user base + script complexity, then stop at
a clean power-of-two boundary.

| Allocation bucket | Vocab | Rationale |
|---|--:|---|
| Bytes + special + reserved | 1,024 | byte fallback, control tokens |
| English + Latin + Hinglish | 45,000 | English ≤1.15 + romanized Indic |
| Code | 25,000 | syntax, indentation, identifiers |
| Math & science | 8,000 | symbols, LaTeX, digit-splitting |
| Devanagari (Hindi + Marathi) | 40,000 | largest Indic base |
| Bengali–Assamese | 20,000 | shared script |
| Tamil | 18,000 | agglutinative |
| Telugu | 18,000 | agglutinative |
| Malayalam | 18,000 | highest fertility script |
| Kannada | 16,000 | agglutinative |
| Gujarati | 12,000 | T2 |
| Perso-Arabic (Urdu) | 12,000 | distinct script |
| Gurmukhi (Punjabi) | 10,000 | T2 |
| Odia | 10,000 | T2 |
| Shared subwords + other world langs | 8,120 | cross-lingual transfer |
| Reserved for expansion | 1,000 | future languages |
| **Total** | **262,144** | **= 2¹⁸** |

The buckets are chosen to sum to exactly **262,144 = 2¹⁸** (verified arithmetically). This is the
same footprint as Gemma 3's tokenizer, but the budget is re-pointed: Gemma spends heavily on CJK,
we spend on Brahmic scripts + code.

## Step 3 — justify the number against alternatives

| Option | Verdict | Reason |
|---|---|---|
| 128K (Llama 3) | Too small | 12+ Brahmic scripts + code + English can't coexist; Indic fertility stays 2.5–4× |
| **262,144 (2¹⁸)** | **Chosen** | hits fertility targets; embedding ≈ 4–5% of params at 40B — effectively free (`02` #10) |
| >300K | Diminishing | gains plateau past ~200K (`01`); low-resource rows rarely activate → wasted params + softmax cost |
| Per-language tokenizers | Rejected | breaks cross-lingual transfer & Hinglish code-mixing; one shared vocab required |

## Why the embedding cost is negligible (the key enabler)

From finding #10: in Gemma 3 the 262K embedding table is ~30% of a 1B model but only **~5% of the
27B model**. At 40B it's ~4% — a rounding error against the compute saved by roughly halving Indic
sequence lengths on *every* forward pass, for the model's entire deployed life. That asymmetry is
what makes "go big on vocab, point it at India" the correct call.

## Ties back to Session 2

Our Session-2 tokenizer showed that **script-aware handling (NFC + not splitting on matras)** matters
more than raw vocab for Indic compression (`use_regex=False` was the whole ballgame). Same lesson
here: the 262K budget only pays off if pre-tokenization respects Brahmic structure — which is also
IndicSuperTokenizer's central finding.

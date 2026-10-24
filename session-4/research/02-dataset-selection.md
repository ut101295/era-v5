# 02 — Dataset selection (live HuggingFace probing)

The assignment: *"find a 10–100M dataset like [the example], … find interesting
datasets, specially from the Session 3 that you may have come across."* So the search
was biased toward the Indic / India-first corpora Session 3 surfaced, and every
candidate was **probed live on the HuggingFace hub** before committing.

## The example link is illustrative — it doesn't exist

The assignment links `lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled` as an
example of a reasoning-distilled dataset. Probing it returned:

```
DatasetNotFoundError("Dataset 'lordx64/Qwen3.6-…-Opus-Reasoning-Distilled'
doesn't exist on the Hub or cannot be accessed.")
```

So it's an illustration of the *kind* and *size* of dataset, not a literal target. That
freed the choice to come entirely from the Session 3 research family.

## Candidate landscape (from Session 3 + the cleaning fit)

| Candidate | From S3? | Cleaning strategies it best exercises |
|---|---|---|
| **AI4Bharat IndicCorpV2** (per-language) | ✅ (finding #4) | Indic normalization, quality-filter bias, dedup, **language-ID mislabels** |
| AI4Bharat Sangraha | ✅ (finding #3) | same, plus a large synthetic share to stress quality/dedup |
| A reasoning/conversation set | example | **ghost-tag / format discipline**, PII-in-transcripts |
| FineWeb-Edu sample | ✅ (finding #7) | quality-classifier cascade, dedup |
| Dolma sample | ✅ (§3 callout) | char-level normalization (doc-clean but char-dirty) |

## The key discovery: IndicCorpV2 exposes languages as *splits*

Probing `ai4bharat/IndicCorpV2`:

```
configs: ['indiccorp_v2']
splits : ['asm_Beng', 'ben_Beng', 'brx_Deva', 'guj_Gujr', 'hin_Deva',
          'kan_Knda', 'mal_Mlym', 'mar_Deva', 'ory_Orya', 'pan_Guru',
          'tam_Taml', 'tel_Telu', 'urd_Arab', …]   # 23 splits
schema : {'text': <str>}
```

This is decisive: **Assamese (`asm_Beng`) and Bengali (`ben_Beng`) share the Bengali
script block.** That is exactly the language-ID confusion the session dwells on (§8: a
document "sitting in the Assamese folder that is really Bengali, which lights up as a
mismatch"). Picking Assamese as the primary corpus gives the language-ID stage a
*genuine, hard case* to catch instead of a toy one — and lets us reproduce the
sovereign thread end to end.

## A 5,000-row probe of `asm_Beng` — and the first defect

```
docs=5000  empty=2500  words=105,400
avg_words/non-empty-doc ≈ 42
throughput ≈ 888 docs/s
```

**~50% of the raw rows are empty.** That became "extra defect #1" — pure wasted tokens
that only the cleaning pipeline removes — and it confirmed the corpus is short
sentence-level web text, which the quality-filter and dedup stages then have to handle.

## The conversation secondary — keeping it India-first

For the ghost-tag stage we need conversation data. Rather than an English set, probing
`ai4bharat/indic-align` found config **`Dolly_T`** with real per-language instruction
pairs (`hin_Deva`, `ben_Beng`, `tel_Telu`, …). Using the Hindi column keeps even the
conversation demo inside the India-first theme.

## Final selection

| Role | Dataset | Size | Purpose |
|---|---|---|---|
| **Primary** | `IndicCorpV2 / asm_Beng` | 111,459 rows ≈ **10.0M tokens**, 49 MB | the corpus we clean (in the 10–100M band) |
| **Mislabel probe** | `IndicCorpV2 / ben_Beng` | 300 docs, stamped `claimed_lang="asm"` | give language-ID a real mismatch to catch |
| **Conversation** | `indic-align / Dolly_T` (Hindi) | 2,000 pairs | demonstrate ghost-tag / format discipline |

Sizing rationale: stream `asm_Beng` until ~10M tokens or 400k rows (whichever first),
which lands the raw corpus at the **lower edge of the 10–100M token band** while staying
runnable on a laptop (~37 s end-to-end). The fetch is deterministic and recorded in
[`raw/fetch-summary.json`](raw/fetch-summary.json).

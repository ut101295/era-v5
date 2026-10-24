# Session 4 — the cleaning pipeline

A small, real, **reproducible** data-cleaning pipeline. It streams a raw Indic web
corpus and runs the eight Session 4 cleaning strategies in order, emitting the real
before/after numbers the widget renders. Deliberately dependency-light: only
`datasets` (for fetching); every cleaning stage uses the Python standard library, and
the language detector is script-based (no fasttext build).

## Layout

| File | Stage | What it does |
|---|---|---|
| `fetch_data.py` | — | Streams the raw corpora from HuggingFace (capped, deterministic). |
| `normalize.py` | 2 | NFC + strip invisibles/controls + unescape HTML + collapse whitespace, **keeping** Brahmic joiners ZWNJ/ZWJ. |
| `formats.py` | 3 | Detects ghost conversation markers (4 formats) and unifies to one canonical special-token format. |
| `quality.py` | 4 | Script-aware heuristic cascade + educational-value gate; measures English-filter bias. |
| `dedup.py` | 5+6 | MinHash (112 perms) + LSH (14×8, FineWeb params); reports cross-shard (global-only) duplicates. |
| `langid.py` | 7 | Script-based language ID; Assamese vs Bengali disambiguated on ৰ/ৱ; fail-loud code normalization. |
| `pii.py` | 8 | Regex layer (emails/phones/IPs/IDs) + name layer with an Indic false-positive probe. |
| `decontam.py` | 9 | Word-13-gram eval fingerprinting + contamination removal + canary planting. |
| `manifest.py` | 10 | Content-addressed IDs, provenance manifest, content hash **after** cleaning, license gating. |
| `run.py` | all | Orchestrates the stages in order and writes `stats.json`. |

Stage 1 (extraction) is the known upstream stage from Session 3 and is noted, not re-run.

## Run

```bash
cd session-4
python3 -m venv .venv && source .venv/bin/activate
pip install -r pipeline/requirements.txt
cd pipeline
python3 fetch_data.py                 # -> data/*.jsonl (gitignored, ~50 MB)
PYTHONHASHSEED=0 python3 run.py        # -> ../cleaning-widget/stats.json + ./stats.json
```

`PYTHONHASHSEED=0` is set so the whole run is byte-for-byte reproducible; the pipeline
already uses stable hashes internally, and the manifest's content hash is identical
across runs.

## The datasets (all AI4Bharat, from the Session 3 research family)

- **Primary:** `ai4bharat/IndicCorpV2` split `asm_Beng` — Assamese web text (~10M tokens).
- **Mislabel probe:** `IndicCorpV2` split `ben_Beng` — 300 Bengali docs stamped as
  Assamese, so language-ID has a real mismatch to catch.
- **Conversation sample:** `ai4bharat/indic-align` config `Dolly_T` — 2,000 real Hindi
  instruction pairs, for the ghost-tag / format stage.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

This is a coursework monorepo ("Axiom — Learning OS"). Each `session-N/` directory is a self-contained assignment with its own README, notes, transcript, and code — there is no shared build system, no root-level dependencies, and no cross-session imports. Treat each session as its own mini-project and work from within its directory.

Common per-session files: `assignment.md` (the task), `notes.md` / `session_notes.md` (working notes), `transcript.md` (session log), plus the code.

## Session 1 — Neural network playground (`session-1/assignment-playground/`)

A single self-contained `index.html` SPA (zero build, zero npm) that implements a neural-network engine **from scratch in vanilla JS** — `DenseLayer` (forward `z = xW+b` + activation, analytical backprop), `SequentialNetwork` (chains layers), and BCE loss with prediction clamping to `[1e-7, 1-1e-7]`. It visually proves four claims: (1) nonlinearity is needed to separate concentric rings, (2) stacked linear layers collapse to one linear map, (3) next-token training induces embedding clusters, (4) memorization vs. generalization gap shrinks with data. Any change to the network math lives entirely in this one file.

Run locally:
```bash
cd session-1/assignment-playground
python3 -m http.server 8000   # open http://localhost:8000/index.html
```

## Session 2 — Multilingual BPE tokenizer (`session-2/multilingual-tokenizer/`)

A byte-level BPE tokenizer built on HuggingFace `tokenizers` (Rust backend), trained on the "India" Wikipedia article in English/Hindi/Telugu/Marathi. Scripts import each other by module name (e.g. `from bpe_tokenizer import ...`), so **run them from inside the `multilingual-tokenizer/` directory**, not the repo root.

Pipeline (order matters — `train.py` exits if `data/*.txt` is missing):
```bash
cd session-2
python3 -m venv .venv && source .venv/bin/activate
pip install -r multilingual-tokenizer/requirements.txt
cd multilingual-tokenizer
python3 fetch_data.py   # downloads data/{en,hi,te,mr}.txt via MediaWiki API
python3 train.py        # optimizes weights, writes output/tokenizer.json + output/results.txt
```

Key architectural points:
- **`bpe_tokenizer.py`** — tokenizer factory (`NFC` normalizer + `ByteLevel(use_regex=False)` pre-tokenizer + `ByteLevel` decoder). The critical detail is `use_regex=False`: the default GPT-style regex splits Indic scripts on matras/vowel signs before BPE can merge them, wrecking compression. Disabling it (byte→visible-Unicode mapping only) preserves Indic word structure. Roundtrip is exact for characters/matras/spaces but **drops newlines** on decode — this does not affect the score.
- **`train.py`** — minimizes compression-ratio *spread*. Score = `1000 / (max_ratio − min_ratio)`, with a hard rule that **English ratio must be ≤ 1.2 or the score is 0.0**. `optimize()` does **multi-start hill climbing** on per-language corpus weights (single-coordinate nudges + pairwise weight trades, sweeping to convergence at a shrinking step), keeping the best across seeds. It optimizes English *up toward* a safety cap `EN_CAP = 1.185` (a deliberate buffer under 1.2, since graders re-run on drifting Wikipedia text). Corpus weighting = repeating each language's lines `int(weight)` times + a fractional tail. Current best: **score 1415.05** (en 1.1839 / hi 1.7644 / te 1.8708 / mr 1.8906), verified reproducible from the saved `tokenizer.json`.
- **`fetch_data.py`** — pulls plaintext extracts from `{lang}.wikipedia.org/w/api.php` and strips reference markers / section headers.

## Session 2 widget (`session-2/tokenizer-widget/`)

Self-contained zero-dependency `index.html` that loads `tokenizer.json` and **re-runs the tokenizer live in the browser** (a JS port of HF byte-level BPE — NFC → byte→unicode → rank-ordered merges with leftmost tie-breaking) to display ratios/score and offer downloads. The JS BPE is verified to match the Python `tokenizers` output exactly. Corpus is copied into `corpus/` (not `data/`) so it escapes `.gitignore`. Needs a server (`python3 -m http.server`) or Netlify — `fetch()` is blocked on `file://`. Deploy target: drag the folder onto Netlify Drop.

## Notes

- `data/*.txt`, `*.json`, and `*.csv` under any `data/` folder are gitignored — regenerate with `fetch_data.py` rather than expecting them in the repo. (The widget's `corpus/` copies are intentionally outside `data/` so they commit.)
- The `file:///Users/.../Personal/Projects/...` links inside the session READMEs are stale absolute paths from an earlier checkout location; ignore them.

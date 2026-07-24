# Multilingual BPE Tokenizer — Submission Widget

A self-contained, zero-dependency web widget for the Session 2 assignment. It loads the trained byte-level BPE tokenizer and, **entirely in the browser**, re-runs it on the source corpora to show:

- per-language **compression ratios** (tokens ÷ words) for English, Hindi, Telugu, Marathi,
- the sorted ratios, the spread, and the **self score** = `1000 / (X_max − X_min)`,
- the English ≤ 1.2 constraint check (with the safety margin),
- a **download** button for `tokenizer.json` (and a vocabulary `.txt` export),
- a live playground: paste any text and watch it tokenize.

Nothing is hard-coded — the widget contains a faithful JavaScript port of HuggingFace's byte-level BPE (NFC → byte→unicode mapping → rank-ordered merges with leftmost tie-breaking) and recomputes the numbers on every load. Its output matches the Python `tokenizers` library exactly (verified: 11,795 / 14,062 / 4,473 / 8,500 tokens → score **1415.05**).

## Files

```
tokenizer-widget/
├── index.html        # the widget (all logic + styling inline)
├── tokenizer.json    # trained tokenizer (loaded live + offered for download)
├── stats.json        # authoritative stats snapshot (reference/cross-check)
└── corpus/           # source texts the widget tokenizes live
    ├── en.txt  ├── hi.txt  ├── te.txt  └── mr.txt
```

The corpus lives in `corpus/` (not `data/`) on purpose so it is committed — the repo's `.gitignore` excludes `**/data/*.txt`.

## Preview locally

Browsers block `fetch()` on `file://`, so serve the folder:

```bash
cd session-2/tokenizer-widget
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy to Netlify

1. Go to [app.netlify.com/drop](https://app.netlify.com/drop).
2. Drag the **entire `tokenizer-widget/` folder** onto the drop zone (it must include `tokenizer.json` and `corpus/`).
3. Netlify returns a public URL — that is the Widget Link for submission.

To refresh after retraining, copy the new artifact and corpus in and redeploy:

```bash
cp ../multilingual-tokenizer/output/tokenizer.json .
cp ../multilingual-tokenizer/data/*.txt corpus/
```

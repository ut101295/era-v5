# Session 4 — Data Cleaning widget

A self-contained, zero-dependency `index.html` that reports the result of running the
eight Session 4 cleaning strategies on a real Indic web corpus (AI4Bharat IndicCorpV2,
Assamese). It loads `stats.json` — the **measured** output of the pipeline in
[`../pipeline/`](../pipeline/) — and renders:

- how many strategies there are (8) and what each one is
- the dataset that was picked, and why
- what each stage cleaned, why, and how — with real before/after numbers
- other concerns cleaned up (the Minor: collection tracker + central dedup machine,
  plus the extra defects our own run caught)
- final statistics and the provenance manifest (with reproducibility proof)

## Run locally

`fetch()` is blocked on `file://`, so serve it:

```bash
cd session-4/cleaning-widget
python3 -m http.server 8000   # open http://localhost:8000/index.html
```

## Deploy to Netlify

Drag this **folder** (`cleaning-widget/`) onto <https://app.netlify.com/drop>.
Both files (`index.html` + `stats.json`) upload together; no build step. The
deployed URL is the assignment submission link.

## Regenerating stats.json

`stats.json` is produced by the pipeline, not written by hand:

```bash
cd session-4
python3 -m venv .venv && source .venv/bin/activate
pip install -r pipeline/requirements.txt
cd pipeline
python3 fetch_data.py                 # downloads the raw corpora (gitignored)
PYTHONHASHSEED=0 python3 run.py        # writes ../cleaning-widget/stats.json
```

The run is deterministic: the same input yields the same content hash every time.

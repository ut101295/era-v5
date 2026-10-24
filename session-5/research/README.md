# Session 5 — research dossier (index)

The evidence behind [`../plan/mixture-and-curriculum-v5.md`](../plan/mixture-and-curriculum-v5.md).
Read the narrative in [`../WALKTHROUGH.md`](../WALKTHROUGH.md); this folder is the auditable
backing for every claim.

| File | What it establishes | Feeds |
|---|---|---|
| [`01-benchmark-map.md`](01-benchmark-map.md) | Each benchmark (SWE-bench, BFCL, τ-bench, Terminal-Bench, HumanEval/MBPP/LiveCodeBench, GSM8K/MATH/AIME/GPQA, MILU/IndicMMLU-Pro/IndicGenBench/IndicQA/IndQA) → the training-data **format**, **loss map** (green vs grey), and **stage** it needs. | which lanes exist, and the loss-masking rules |
| [`02-dataset-inventory.md`](02-dataset-inventory.md) | Real public datasets per lane with **samples + tokens + license + provenance tier**, and a **supply verdict** (unique / repeat / synthesize). Appendices A–C: per-language verified-native Indic ceiling, the ~10k real-agentic-trajectory commons, long-context repacking. | the honest supply ceiling for every share |
| [`03-mixture-derivation.md`](03-mixture-derivation.md) | The arithmetic reconciling each desired share against its real supply; the Indic four-tier split; floor + reserve; curriculum + bands; the hypothesis. | the numbers in the plan |

**Method:** compose backward from benchmarks (`01`) → size against real supply (`02`) →
shrink any lane the supply can't honestly cover, and state where a share needs repetition or
synthesis (`03`) → protect the scarce lanes → test as a hypothesis with a real proxy
([`../proxy/`](../proxy/)).

Sources are cited inline in each dossier (HuggingFace dataset cards, arXiv papers, official
leaderboards), gathered via web research in 2026.

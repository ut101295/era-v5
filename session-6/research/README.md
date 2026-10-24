# Session 6 — Research Dossiers

Grounded, web-sourced background for the **Training Data Execution System**. Each dossier
maps to a block of the assignment and ends with a "minimal implementation recipe" that the
plan (`../plan/`) turns into code.

| # | Dossier | Assignment sections | Graded block |
| --- | --- | --- | --- |
| 01 | [Deterministic, resumable, replayable dataloading](01-deterministic-resumable-dataloading.md) | §2, §9, §14 | End-to-end; Checkpoint/crash/resume/replay/fork (150) |
| 02 | [Packing, loss/attention masks, position ids](02-packing-masks-position-ids.md) | §3, §4, §5 | Packing, masks and batch correctness (150) |
| 03 | [Immutable shards, manifests, lineage](03-shards-manifests-lineage.md) | §6 | Shards, manifests, tokenizer integrity (100) |
| 04 | [Consumption/learning ledger; replay/fork/audit](04-ledger-replay-fork-audit.md) | §8–§12, §14 | Consumption & learning ledgers (150) |
| 05 | [Eval/contamination firewall; OPUS audit](05-eval-contamination-firewall.md) | §10, §13 | Eval/validation firewall (50); OPUS (in mixture 150) |
| 06 | [Throughput and useful tokens/sec](06-throughput-useful-tokens.md) | §15 | Throughput and packing efficiency (50) |

**How this research was produced:** five parallel deep-research fan-outs (each spawning its
own sub-searches) against primary sources — Megatron-Core, torchdata `StatefulDataLoader`,
HF streaming, MosaicML StreamingDataset, WebDataset, IPFS/DVC content-addressing, LakeFS /
Iceberg / Delta transaction logs, Event Sourcing, W3C PROV / OpenLineage / MLMD, BIG-bench
canaries, EleutherAI/Dolma/LMSYS decontamination. Dossiers 02 and 06 lost their consolidated
web pass to mid-stream API errors and are written from domain knowledge with canonical
citations (flagged inline for re-verification). The full narrative of how the research was
run is in [`../WALKTHROUGH.md`](../WALKTHROUGH.md).

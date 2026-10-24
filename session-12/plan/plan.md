# Session 12 — Plan

## Goal
Stand up 32 virtual GPUs, run a demo model on them, and show how memory and computation change
under plain data parallelism vs ZeRO stages 1/2/3 — as *measured* numbers plus a *correctness
proof*, shipped as a detailed README + support code + one executed notebook. Binding rule: don't
assert the savings, measure them from the actual per-rank tensors, and prove sharding doesn't
change the result.

## Constraints from the spec
- Submission = detailed README.md + support code.
- 32 GPUs; a demo model running on them.
- Simulate DP and ZeRO-1 / ZeRO-2 / ZeRO-3.
- Show the memory change (per-GPU state) and the computation change (communication) across stages.

## Steps
1. **The 16-byte picture** (§0): a width-256 depth-3 MLP (P≈83k); enumerate bf16 weight 2 + bf16
   grad 2 + fp32 master 4 + Adam m 4 + Adam v 4 = 16 bytes/weight that DP replicates on every GPU.
2. **Sharding machinery**: one fp32 master vector; contiguous shards across 32 ranks (`sizes`,
   `bounds`); `full_gradient()`; vectorized `adam_coord()` with a bf16 grad round-trip.
3. **§1 Memory** — `per_gpu_bytes(stage)` measured from the tensors each rank holds; table + stacked
   bar; confirm N=8 → 16/5.50/3.75/2.00 and N=32 → 16/4.375/2.438/0.500.
4. **§2 Correctness** — one real step under all four stages, reconstruct the full weight, assert
   bitwise identical (max |Δ| = 0.0). This is the anchor claim.
5. **§3 Communication** — 2P/2P/2P/3P with the reduce-scatter + all-gather decomposition; explain
   ZeRO-3's second all-gather (forward + backward) and the 1/N optimizer-compute drop.
6. **§4 Train under ZeRO-3** — 60-step fully-sharded loop; show loss falls and the reconstructed
   weight stays 0.0 from a replicated reference every step.
7. **§5 Memory ladder** — bytes/weight vs N (DP flat at 16, ZeRO ~1/N); concrete 7B-on-80GB fit table.
8. **Notebook** — `build_notebook.py` splits the script; execute with outputs + 3 plots embedded.
9. **Docs** — README (concept-first), requirements, session-notes, this plan, research, WALKTHROUGH.

## Risks / watch-items
- **bytes/weight off by shard size** → divide total per-GPU bytes by **P**, not by `sizes[0]`
  (this bug printed 511 for DP; fixed to 16).
- **"Simulation" read as fake arithmetic** → make each rank a real object holding real sharded
  tensors; the correctness proof is what earns trust in the memory table.
- **Over-claiming** → state scope honestly: single-process functional model (not NCCL),
  training-state bytes only (activations are a separate axis), gradients treated as pre-averaged.
- **Script/notebook drift** → one source of truth (`zero_sim.py`), notebook generated from it.

## Definition of done
Notebook runs top to bottom with outputs + all three plots; memory numbers match the lecture at
N=8 (16/5.5/3.75/2.0); all four stages bitwise identical (max |Δ|=0.0); ZeRO-3 trains
(1.3924→0.0170) staying 0.0 from replicated; communication reported 2P/2P/2P/3P with reasoning;
7B-on-80GB fit table concrete. (User pushes to GitHub + pastes link.)

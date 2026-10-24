# Session 12 — Walkthrough

A guided tour of the submission, in the order it's easiest to understand and defend.

## The 60-second version

The deliverable is a **faithful simulation of ZeRO across 32 virtual GPUs** that doesn't just
*assert* the memory savings — it **measures** them and **proves** the sharding is correct.
Everything comes from one file ([zero-sim/zero_sim.py](zero-sim/zero_sim.py)) split into one
notebook that runs top to bottom with outputs and all three plots embedded. The anchor claim,
stated as a test: sharding changes *who stores and computes what*, never the result.

## 0. The 16-byte idea (§0)

Every weight drags 16 bytes of training state: bf16 weight (2) + bf16 grad (2) + fp32 master (4)
+ Adam `m` (4) + Adam `v` (4). Plain data parallelism puts all 16 on *every* GPU, so adding GPUs
never grows model capacity. That single fact is what ZeRO exists to fix.

## 1. Measured memory per GPU (§1) — `fig_memory_per_gpu.png`

Each rank is a real object holding only its shard; I measure the bytes it holds by category. The
stages shard one more thing each: ZeRO-1 the optimizer state, ZeRO-2 also gradients, ZeRO-3 also
weights → bytes/weight `16 → 4+12/N → 2+14/N → 16/N`. At N=8 these are exactly the lecture's
**16 / 5.50 / 3.75 / 2.00**; at N=32, **16 / 4.375 / 2.438 / 0.500**.

## 2. Correctness: DP ≡ ZeRO (§2)

The decisive check. One real optimizer step under DP / ZeRO-1 / ZeRO-2 / ZeRO-3, reconstruct the
full weight, compare. All four are **bitwise identical (max |Δ| = 0.0)**. This is ZeRO's promise
written as an assertion — the memory table is only trustworthy because the thing it describes
provably computes the same answer.

## 3. Communication per step (§3)

Memory isn't free. An all-reduce = reduce-scatter (P) + all-gather (P) = 2P. DP / ZeRO-1 / ZeRO-2
are all **2P**; **ZeRO-3 is 3P** because sharded weights must be all-gathered twice per step (once
for the forward, once for the backward). So ZeRO-1/2 are ~free memory wins; ZeRO-3 trades +50%
bandwidth for the biggest saving. (Per-GPU optimizer *compute* also drops to 1/N — each rank
updates only its shard.)

## 4. The model actually trains under ZeRO-3 (§4) — `fig_zero3_training.png`

To show §2 wasn't a lucky identity, train 60 steps fully sharded: all-gather weights → fwd/bwd →
reduce-scatter grad → each rank updates its shard → all-gather new weights. Loss falls
**1.3924 → 0.0170**, and the reconstructed weights stay **0.0** apart from a replicated reference
every step. A real model, really sharded, really training.

## 5. The memory ladder (§5) — `fig_memory_ladder.png`

Per-GPU bytes/weight vs N: DP is **flat at 16** (why it can't scale), ZeRO stages fall as **1/N**,
ZeRO-3 fastest. Made concrete: a **7B** model on **80 GB** cards at N=32 does *not* fit under DP
(16 B/wt → 5.0B max) but fits comfortably under every ZeRO stage (ZeRO-3 → 160B params of
headroom). Same hardware, same model — only the distribution of the 16 bytes changed.

## 6. One source of truth

`zero_sim.py` is authoritative. `build_notebook.py` splits it on the `# ===` separators into
`zero_sim.ipynb`, executed with outputs + plots embedded — script and notebook can't drift.

## 7. Reproduce

```bash
cd zero-sim
pip install -r requirements.txt
python zero_sim.py            # prints every number, writes the three PNGs
# or: open zero_sim.ipynb and Run All
```

Submit: push the repo, paste the GitHub README link into the course portal and `assignment.md`.

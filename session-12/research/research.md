# Session 12 — Research notes

Background facts I relied on, with the reasoning so the harness is defensible.

## The 16 bytes per weight
Mixed-precision Adam training keeps, *per weight*: a bf16 weight the forward pass uses (2 B), its
bf16 gradient (2 B), an fp32 "master" copy (4 B), and Adam's two fp32 moments `m` and `v` (4 B
each). That's **16 bytes**. The fp32 master exists because a bf16 weight has ~8 bits of mantissa;
a tiny update (`η·m̂/√v̂` can be ~1e-7 of the weight) would round away entirely if applied in
bf16. So the real update happens in fp32 and the bf16 copy is re-derived — this is why the master
+ moments (12 of the 16 bytes) dominate and are the first thing ZeRO shards.

## Why plain data parallelism can't scale model size
DP replicates all 16·P bytes on **every** GPU and only averages gradients (an all-reduce) each
step. Adding GPUs multiplies throughput but holds per-GPU state **constant** — so the largest
model you can train is capped by one GPU's memory no matter how many you own. For a 7B model,
16·7e9 = 112 GB of state, already past an 80 GB card. That ceiling is the entire motivation for
ZeRO.

## ZeRO = shard the redundant state, gather on demand
The N copies of optimizer state across N DP ranks are redundant. ZeRO keeps only **1/N** of it on
each GPU and communicates the missing pieces exactly when needed. The three stages shard one more
thing each, so per-GPU bytes/weight are:
- **ZeRO-1** (shard optimizer state): `4 + 12/N` — weight+grad (2+2) stay replicated, the 12 bytes
  of master+m+v divide by N.
- **ZeRO-2** (also shard gradients): `2 + 14/N`.
- **ZeRO-3** (also shard weights): `16/N` — everything divides by N.
At **N=8** these are 5.50 / 3.75 / 2.00 (vs DP 16), the exact lecture numbers; at **N=32** they're
4.375 / 2.438 / 0.500. The optimizer state is sharded first because it's the biggest (12/16) and
the easiest — only the optimizer step touches it, so it never needs to be present during
forward/backward.

## Communication: the 2P / 3P accounting
The unit is **P**, one pass over the parameters. An **all-reduce** (DP's gradient average) is a
*reduce-scatter* (P) followed by an *all-gather* (P) = **2P**.
- **DP, ZeRO-1, ZeRO-2 all cost 2P.** ZeRO-1 adds nothing — the sharded optimizer state is local.
  ZeRO-2 rearranges into reduce-scatter grads (P) + all-gather updated weights (P), still 2P.
- **ZeRO-3 costs 3P**, because the weights themselves are sharded and must be all-gathered
  **twice** per step — once to run the forward pass, once for the backward — plus the
  reduce-scatter of gradients. So ZeRO-1/2 are near-free memory wins; ZeRO-3 pays +50% bandwidth
  for the largest saving. Separately, per-GPU **optimizer compute** drops to 1/N under every ZeRO
  stage (each rank updates only its shard instead of redundantly updating all P weights).

## Why the correctness proof matters
ZeRO's defining promise is that it changes *where state lives and who computes the update*, **not
the result**. A memory table is only trustworthy if that holds, so the simulation runs one real
optimizer step under all four stages and checks the reconstructed weights are **bitwise identical
(max |Δ| = 0.0)**, then trains the demo model 60 steps under full ZeRO-3 sharding (all-gather
weights → fwd/bwd → reduce-scatter grad → per-shard update → all-gather) to show it actually
learns while staying 0.0 from a replicated reference. Passing both is the difference between
"plausible arithmetic" and "a working sharded trainer."

## Honest scope
- **Single-process functional model** of 32 ranks, not NCCL — chosen because it's perfectly
  reproducible and lets me *prove* DP≡ZeRO, which real multi-process timing would obscure. The
  ranks are real objects with real sharded tensors; only the data movement is emulated, and its
  cost is accounted separately (§3).
- **Training-state bytes only** (weights+grads+optimizer). Real ZeRO-3 also needs a transient
  all-gather buffer, and **activations** are a separate memory axis (activation checkpointing,
  sequence/tensor parallelism) — out of scope and flagged.
- Gradients are treated as **already averaged** across DP ranks; that averaging *is* the 2P
  all-reduce counted in §3, so using a single gradient doesn't understate communication.

## Production lineage
This is the idea behind **PyTorch FSDP** and **DeepSpeed ZeRO**. CPU/NVMe **offload** pushes the
sharded state off the GPU entirely for a further memory/speed trade, enabling very large models on
modest hardware.

## Sources (concepts)
- Rajbhandari et al. 2020 — "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
  (the three stages and the memory/communication analysis).
- Ren et al. 2021 — "ZeRO-Offload" / Rajbhandari et al. 2021 — "ZeRO-Infinity" (CPU/NVMe offload).
- Zhao et al. 2023 — PyTorch FSDP.
- Micikevicius et al. 2018 — mixed-precision training (the fp32 master-weight rationale).
- Course lecture (Session 12) — the 16-byte picture and the per-stage bytes/weight at N=8.

# Session 13 — Research notes

Background facts I relied on, with the reasoning so the harness is defensible.

## Why activations, not weights, are the memory wall
The forward pass stores intermediate results (activations) that the backward pass needs.
Korthikanti et al. (2022) counted them at ~34 bytes per token per unit of hidden size for a
transformer layer with FlashAttention. That total scales with **depth × tokens × hidden**, so
for a deep model one long sequence can store more activations than the whole training state —
the lecture's example is 127.5 GiB of activations for a single 8,192-token sequence through a
96-layer model, larger than an H100's 80 GiB. ZeRO (Session 12) shards weights/grads/optimizer
but **never divides activations**, so activations become the binding constraint. Reversibility
attacks exactly that term.

## The reversible rule (midpoint / leapfrog)
From Gal, Eliasof, Turek, Ascher, Treister & Haber, *Reversing Large Language Models for
Efficient Training and Fine-Tuning* (Nov 2025). Treat the layer index as time and advance the
residual stream with a rule runnable in both directions. With `p^ℓ` the stream entering layer
ℓ and `f_θℓ` the block (attention then FFN), the **midpoint** rule is

```
p^{ℓ+1} = p^{ℓ-1} + 2h · f_θℓ(p^ℓ)          (forward)
p^{ℓ-1} = p^{ℓ+1} − 2h · f_θℓ(p^ℓ)          (exact inverse)
```

It adds the block's output to the state **two layers back** and evaluates the block at the
state **in between** — the state the backward pass already holds — so recovery needs nothing
that depends on the value being recovered. I implement it as the two-register leapfrog
`(a,b) → (b, a + 2h·f(b))`, inverse `(a,b) → (b − 2h·f(a), a)`, which I verify reconstructs the
input to ~1e-7.

### Why "euler" (the ordinary residual) is NOT reversible
The standard block `p^ℓ = p^{ℓ-1} + f(p^{ℓ-1})` (a forward-Euler step) cannot be inverted:
recovering `p^{ℓ-1}` would need `f(p^{ℓ-1})`, computed from the very state being recovered. So
the answer to "which variant worked" is: **midpoint/leapfrog is the reversible one; Euler is
the non-reversible baseline.** This is why the assignment phrases it as "midpoint, euler, etc."

## Why the custom autograd is mandatory
Changing only the forward formula saves nothing: PyTorch's autograd records a graph and stores
every intermediate it needs for backward. The memory win requires a hand-written
`torch.autograd.Function` whose `forward` runs under `no_grad` and keeps only the two boundary
states, and whose `backward` walks down the stack, reconstructs each layer's input via the
inverse rule, recomputes that one block *with* grad, and accumulates parameter gradients. This
is the RevNet backward (Gomez et al. 2017) adapted to the leapfrog coupling. I check its
gradients against ordinary autograd on the identical forward and they agree to ~1e-6 — so the
memory trick is numerically free, not a different computation.

## The trade: compute for memory
Each block runs once in the forward pass and once more during backward (to recompute its
output for the inverse + gradients). The authors estimate 30–50% step overhead, reasoning that
backward already costs more than forward, so one extra forward is a modest add. My local
measurement reproduces this: the reversible stack runs at roughly two-thirds the tokens/s of
the standard stack — i.e. ~1.5× time per step, squarely in the 30–50% band.

## Why the saved memory becomes throughput
Activation memory that no longer grows with depth frees room for a **much larger batch**, and
on a memory-bound accelerator a larger batch means more tokens per step and higher throughput.
The paper measured 114.49 vs 56.96 samples/s at hidden 512 / 96 layers — a 2× throughput gain
that came entirely from fitting batch 874 instead of 52. On an 80 GB H100 their standard model
fit batch 26 and the reversible one batch 257 (~9.9×), and ~10× held across five cards.

## Two correctness constraints
- **Deterministic forward.** The backward recomputes each block; a random dropout mask would
  make the recomputation differ from the forward that produced the stored output. So dropout
  must be 0 (the Lightning-LM report treats this as a correctness requirement). My reversible
  runs use dropout 0.
- **Marginal stability.** The paper's analysis needs the step size `h` and blend coefficient in
  a narrow range; outside it reversible nets are hard to train. I use `2h = 1` (same update
  scale as a unit residual) and zero-initialise each block's output projection so the stack
  starts as the identity — without that, the leapfrog's additive accumulation blows the
  residual stream up at step 0 (I observed a loss of ~250 before adding it).

## What reversibility changes downstream (Section 17)
It doesn't make memory free; it **moves the constraint**. Once activations are tiny, the
weights and optimizer state become the term that decides the parallel layout — the things ZeRO
divides. So on a small/shallow model the *total* peak memory can be dominated by the optimizer
state and look similar with or without reversibility; the activation saving only dominates at
depth and long sequence. My depth sweep isolates the activation term to show the mechanism
cleanly, and I report total vs activation peak separately so this distinction is explicit.

## Sources (concepts)
- Gal et al. 2025 — *Reversing Large Language Models…* (arXiv:2512.02056): the midpoint rule,
  the overhead estimate, and the batch-size memory result.
- Gomez, Ren, Urtasun & Grosse 2017 — *The Reversible Residual Network*: the reconstruct-in-
  backward technique the custom autograd implements.
- Korthikanti et al. 2022 — activation-memory accounting (~34 bytes/token/hidden).
- Course lecture (Session 13), Sections 16–17 — reversibility and what it changes.

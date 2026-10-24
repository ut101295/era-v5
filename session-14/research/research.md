# Session 14 — Research notes

Background facts I relied on, with the reasoning so the harness is defensible.

## What a Mixture-of-Experts layer is
Replace a transformer layer's single feed-forward network (FFN) with **N** FFNs ("experts") plus
a small **router**. For each token the router scores all experts, keeps the **top-k** (here k=2 of
N=8), runs only those, and combines their outputs by the router's (renormalised) weights. So the
*total* parameters grow with N, but the *compute per token* grows only with k — capacity is
decoupled from cost. This is the Switch-Transformer / GShard idea: more parameters without
proportionally more FLOPs.

## The expert: SwiGLU
Each expert is the same FFN shape as the dense model — a SwiGLU block
`down( SiLU(gate·x) ⊙ up·x )`. An MoE layer is then a strict *generalisation* of a dense FFN: N
identical experts whose top-k combine-weights sum to 1 compute exactly the dense FFN's output.
That fact is what makes upcycling exact.

## Sparse upcycling — why loss continues instead of restarting
From Komatsuzaki et al. 2022 (*Sparse Upcycling: Training Mixture-of-Experts from Dense
Checkpoints*). Take a trained dense model and build an MoE by **copying the dense FFN into every
expert** and starting the router near zero. Then:
- every expert is (almost) the dense FFN, and
- the top-k weights are renormalised to sum to 1,

so the MoE's output at step 0 equals the dense output. I verify this two ways: the MoE's
cross-entropy at conversion matches the dense model's to a tiny `|Δ|` (controlled by the deliberate
symmetry-breaking noise — at noise 0 it is bit-for-bit exact; here |ΔCE| = 8.48e-06), and the MoE's
**first training step loss lands on the dense model's final loss** (≈0.049 vs 0.047). Training
therefore **continues from
the dense loss**, not from scratch — the whole point of the assignment ("show they continue to
train and reduce loss"). The contrast run (MoE trained from scratch under an equal step budget)
shows how much the warm start buys.

## Why a load-balancing loss is needed
Left alone, the router collapses onto a few favourite experts; the rest get no gradient and never
learn, so the extra capacity is wasted. The **Switch-Transformer auxiliary loss** fixes this:

```
aux = α · N · Σ_i  f_i · P_i
```

where `f_i` is the fraction of (token, slot) picks routed to expert i and `P_i` is the mean router
probability for expert i. It is minimised when both are uniform (1/N), so it pushes the router
toward even utilisation. I add it (α=0.01) to the loss and log the per-expert load; I report the
load coefficient-of-variation (0 = perfectly even) to show no expert is starved.

## Active vs total parameters
The headline number MoE papers quote is "X total params, Y active per token". For this config
(8 experts, top-2) the MoE has several× the dense model's FFN parameters but only ~2× the per-token
FFN compute (it runs 2 of 8 experts). I compute both in the notebook so the capacity-vs-cost trade
is explicit and defensible.

## Why "linear → MoE" is read as "dense FFN → MoE"
The assignment says "train a Linear model and convert that into an MoE." A single `nn.Linear` has
no nonlinearity to make into interesting experts and can't language-model, so I read "linear/dense"
as the ordinary **dense feed-forward** transformer (the standard baseline an MoE is built from in
every MoE paper) and upcycle its FFNs into experts. This is the faithful, defensible reading — it
is exactly the dense→sparse conversion the MoE literature calls upcycling.

## Correctness gate
Before trusting any training curve I check the conversion reproduces the dense prediction: compare
the **cross-entropy computed from the logits** (not `model(x,y).loss`, which for the MoE carries the
load-balance aux term). If the MoE didn't reproduce the dense output at init, "continues training"
would be a coincidence of the optimizer, not a property of the construction. Gating on the
equivalence makes the claim real.

## Sources (concepts)
- Komatsuzaki, Puigcerver, Lee-Thorp et al. 2022 — *Sparse Upcycling* (arXiv:2212.05055): copy a
  dense checkpoint into experts and continue.
- Fedus, Zoph & Shazeer 2021 — *Switch Transformers*: top-1/top-k routing and the load-balance
  auxiliary loss.
- Shazeer et al. 2017 — *Outrageously Large Neural Networks*: the original sparsely-gated MoE.
- Shazeer 2020 — *GLU Variants Improve Transformer* (SwiGLU expert).
- Course lecture (Session 14) — dense → MoE conversion.

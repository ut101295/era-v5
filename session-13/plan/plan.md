# Session 13 — Plan

## Goal
Train a ~20M LLM three ways — standard, reversible (midpoint), and reversible at the maximum
batch — and report final loss, throughput (tokens/s), peak memory and findings, as a detailed
README + runnable notebook(s). Binding rule: the memory saving must be *real and measured*,
which means a hand-written reversible backward, not just a changed forward formula.

## Constraints from the spec
- Train a 20M LLM for 50M tokens (Colab or similar). Fix a batch size you can run.
- Train again with reversibility; report which variant worked (midpoint, euler, …).
- Train again with reversibility pushed to the maximum batch size.
- Report final loss, speed (tok/s), memory peak, and other findings.
- Submit a detailed README (GitHub link); repo must contain the ipynb notebook(s).

## Steps
1. **Model** — GPT-style byte LM; one block `f` = attention then FFN written so the ordinary
   residual layer is exactly `p = p + f(p)`.
2. **Two stacks** — StandardStack (Euler residual, not reversible) and ReversibleStack
   (midpoint/leapfrog `p[l+1]=p[l-1]+2h·f(p[l])`) with a custom `autograd.Function` whose
   backward reconstructs each layer's input and recomputes one block at a time.
3. **Correctness** — forward-then-inverse reconstruction to ~0; custom backward vs ordinary
   autograd to several decimals. (Gate everything on this.)
4. **Memory vs depth** — measure peak activation memory for both stacks at growing depth; the
   shapes (standard linear, reversible flat) prove the claim hardware-independently.
5. **Three training runs** — standard@B, reversible@B, reversible@maxB; log loss, tok/s, and
   both the step high-water peak and the optimizer floor.
6. **Notebook** — build_notebook.py splits the script; execute with outputs + plots.
7. **Docs** — README (the three runs + findings), requirements, session-notes, plan, research,
   WALKTHROUGH. `FULL=True` is the 20M/50M-token Colab config; `FULL=False` is the local run.

## Risks / watch-items
- **Naive reversible saves no memory** — autograd stores everything unless the backward is
  hand-written. The custom Function is mandatory; verify its gradients.
- **Leapfrog instability** (lecture: "marginally stable") — zero-init block output projections
  so the stack starts as the identity; use `2h=1`. Without it the loss blew up (~250) at step 0.
- **Dropout must be 0** in reversible mode or the recomputed activations won't match the forward.
- **Small/shallow model is optimizer-bound**, so *total* peak can look similar; report activation
  high-water vs optimizer floor separately, and use the depth sweep to isolate the mechanism.
- **Batch→throughput payoff is GPU-bound** — on MPS a tiny model doesn't speed up with a bigger
  batch; report the memory headroom locally and attribute the throughput gain to a memory-bound
  GPU (with the paper's measured 2×).

## Definition of done
Notebook runs top to bottom with outputs + both plots; correctness PASS (recon ~1e-7, grads
~1e-6); depth sweep shows standard rising and reversible flat; three training runs all reduce
loss; reversible's activation peak clearly below standard's at the same batch; reversible overhead
lands in the lecture's 30–50% band; findings stated honestly incl. the MPS throughput caveat.
(User runs the FULL 50M-token job on Colab, pushes to GitHub, pastes the link.)

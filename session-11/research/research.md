# Session 11 — Research notes

Background facts I relied on, with the reasoning so the harness is defensible.

## Adam, and why the step is ~η
Adam keeps two running averages per weight: `m` (mean of the gradient, momentum) and `v`
(mean of the gradient *squared*, the second moment). The update is
`−η · m̂/(√v̂+ε)`. For a steady-sign gradient, `m̂ ≈ g` and `√v̂ ≈ |g|`, so `m̂/√v̂ ≈ 1`
and the step size is `≈ η` regardless of how big the gradient is. That normalization is the
whole reason Adam is robust to gradient scale — and it's why the hand-table's steps are all
≈ `−0.001` with `η = 1e-3`. The `ε` lives *outside* the square root in PyTorch's default
(`√v̂ + ε`), which the hand computation matches exactly.

## Bias correction and its timescale
At `t=0` both moments start at zero, so early `m`, `v` are biased *toward zero*. The
correction divides by `1−βᵗ`. The two corrections decay on very different clocks:
- **β₁ = 0.9**: `1−0.9ᵗ` reaches 0.99 at `t≈44` — the momentum correction is done fast.
- **β₂ = 0.999**: `1−0.999ᵗ` reaches 0.99 only at `t≈4603` — the second-moment correction is
  the long pole, because `v̂ = v/(1−0.999ᵗ)` stays a large multiplier for thousands of steps.
Without correction, `v` is *under*-estimated early, so `√v̂` is too small and the step is too
big: `(1−β₁)/√(1−β₂) = 0.1/√0.001 = 3.16` at `t=1` (the canonical 3.16η). This is why the
"stops mattering" answer is **thousands of steps**, not twenty — the slow moment sets it.
(Common lore says "a few hundred steps"; that's because warmup + noisy gradients mask the
effect visually, not because the correction factor itself is small.)

## Update-to-weight ratio
`‖update‖/‖weight‖` is the model-agnostic "how fast is this layer moving" diagnostic; a
healthy value is **~1e-3** (an update that nudges each weight by about a thousandth of its
magnitude). During warmup the LR ramps, so the ratio *rises* with it while the weights are
still near init; it **peaks when the ramp ends**, then drifts down as the weight norm grows
(numerator ~constant-η, denominator growing). So the step where "warmup stops changing it" is
the peak, not a plateau. A ratio pinned far from 1e-3 flags a mis-scaled layer, which is why
people watch it per-layer.

## Cosine vs WSD schedules
**Cosine** decays the LR along a cosine from the peak to ~0 over a *pre-committed* horizon T.
Its weakness: you must know T in advance, and if you stop early you're frozen mid-schedule
(LR still high, loss not settled) — and you can't cheaply "finish" without running to T.
**WSD** (Warmup-Stable-Decay) holds a flat peak for most of training and decays only the last
~10–20%. It **matches cosine's final loss** while letting you (a) stop or extend anytime and
(b) branch any checkpoint and decay it into a finished shorter run. That flexibility — one run,
a usable model at any budget — is why WSD is the pragmatic default. The lecture's framing:
cosine's single advantage (a slightly smoother final descent) rarely pays for its rigidity.

## Learning-rate transfer and width (muP)
Under the **standard parameterization**, the largest stable LR shrinks as the model gets wider
— roughly `η_opt ∝ 1/width` (a factor ~2 per doubling). So an LR tuned at one width is wrong
at another, and tuning directly at the target width is expensive. The lecture's reference
table: 256→3.0e-3, 512→1.5e-3, 1024→7.5e-4, 2048→3.8e-4, 4096→1.9e-4 (each ×0.5 per doubling).
**muP (maximal update parameterization)** rescales initialization, LR, and output multipliers
per-layer by width so that the optimal LR *stops moving with width* — you tune on a small proxy
(e.g. width 256) and the value transfers to a huge model unchanged. That turns a risky
extrapolation into a one-shot small sweep, which is why frontier labs use it for HP transfer.
On a toy MLP the measured per-doubling ratio (~0.32 here) is only approximately 0.5 — the rule
is a scaling *tendency*, not an identity, which is itself the argument for muP.

## The meta-point: tune both sides
The assignment's warning — *almost every optimizer claim that failed to replicate was a
well-tuned method measured against a badly-tuned one* — is the reason each task states its setup
(task choice, capacity, grid, parameterization) and names its caveats. A comparison where one
side is mistuned is not evidence; a fair comparison with stated confidence is.

## Sources (concepts)
- Kingma & Ba 2015 — Adam, the moment estimates and bias correction.
- Loshchilov & Hutter 2019 — AdamW (decoupled weight decay) and cosine annealing with warm restarts.
- Hu et al. 2024 (MiniCPM) / Hägele et al. 2024 — WSD schedules and checkpoint-and-decay.
- Yang et al. 2021 (muP / "Tensor Programs V") — hyperparameter transfer across width.
- Course lecture (Session 11) — the Adam 5-step reference table and the width→LR table.

# Session 11 — Walkthrough

A guided tour of the submission, in the order it's easiest to understand and defend.

## The 60-second version

The deliverable is **five optimizer claims, each measured fairly instead of asserted**.
Everything is generated from one file ([optimizers/adam_lab.py](optimizers/adam_lab.py))
into one notebook that runs top to bottom with outputs and all four plots embedded. The
theme — straight from the assignment — is *tune both sides before you trust a comparison*,
because almost every optimizer claim that failed to replicate was a well-tuned method
measured against a badly-tuned one.

## 1. Adam by hand == PyTorch (§1)

The floor check. One weight, five gradients, `η=1e-3`; compute `m, v, m̂, v̂, step` by hand
and race them against `torch.optim.Adam` with the same gradients fed in. They agree to
**~1e-8** and the final weight is exactly the lecture's **0.995031**. The thing to notice:
the step is ~`−η` every time — Adam normalizes the gradient away, so `η` sets the distance.

## 2. Bias correction, and the honest crossover (§2) — `fig_bias_correction.png`

Switch correction off: the first step jumps from `1.00η` to `3.16η` (the textbook number).
The assignment asks when it stops mattering, and the honest answer is **not within twenty
steps** — the fast moment (β₁) is done by ~step 44, but the slow moment (β₂=0.999) drags the
gap out to **~step 3925** at the 1% bar. The plot shows the two curves still far apart across
the first 60 steps. Reporting ~3925 instead of "about twenty" *is* the fair-measurement point.

## 3. Update-to-weight ratio (§3) — `fig_update_ratio.png`

Log `‖update‖/‖weight‖` per layer. It sits in the **healthy ~1e-3 band**, and — the detail —
it **rises through warmup and peaks at step ~43**, right where the 40-step LR ramp ends, then
drifts down as the weights grow. So the step where "warmup stops changing it" is the peak of
the curve. (Earlier I detected this wrong by assuming a flat plateau; the ratio actually
peaks-then-decays, so the detector is the argmax, not a plateau-crossing.)

## 4. Cosine vs WSD (§4) — `fig_schedules.png`

Same model, same data, both plans 300 steps, both stopped at 200. **Cosine (0.6349) is the
worst** — it committed its LR budget to a 300-step decay and at 200 is behind on both counts.
WSD (0.5378 flat, 0.5533 decayed-checkpoint) wins. **Keep WSD**: one long flat run yields a
usable model at *any* budget, because you can branch a checkpoint and decay it cheaply —
cosine makes you pre-commit to the horizon.

## 5. LR sweep across widths (§5) — `fig_lr_sweep.png`

Sweep LR at widths 256/512/1024 on a smooth surface (sharp optimum needed to see a subtle
shift). Minima drift **monotonically left**: 1e-2 → 3.16e-3 → 1e-3 (0.32 per doubling vs the
`1/width` ideal 0.50). Extrapolated **4096 ≈ 1e-4** (lecture table: 1.9e-4, same order), with
**LOW-to-MODERATE confidence** stated out loud — it's two doublings past the data under the
standard parameterization. The principled fix named: **muP** flattens the optimum so one sweep
at width 256 transfers exactly.

## 6. One source of truth

`adam_lab.py` is authoritative. `build_notebook.py` splits it on the `# ===` section
separators into `adam_lab.ipynb`, executed with outputs + plots embedded — script and notebook
can't drift.

## 7. Reproduce

```bash
cd optimizers
pip install -r requirements.txt
python adam_lab.py            # prints every number, writes the four PNGs
# or: open adam_lab.ipynb and Run All
```

Submit: push the repo, paste the GitHub README link into the course portal and `assignment.md`.

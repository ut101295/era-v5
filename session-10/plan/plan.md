# Session 10 — Plan

## Goal
Build a training loop that is made *observable and self-checking*: satisfy the six tasks
(shapes, hand-checked gradient, broken accumulation with a plotted gap, grad-norm-before-loss,
honest MFU, `0.1` in three float formats), and ship it as one notebook that runs top to bottom
plus a README with the numbers.

## Constraints from the spec
- Submission = GitHub repo + README + `.ipynb`.
- Task 2: finite-difference gradient must agree with `backward()` to several decimals.
- Task 3: must *plot both curves* so the gap is seen, not asserted; use uneven micro-batches.
- Task 4: log grad norm every step; find a step where it moved before the loss.
- Task 5: compute MFU, report honestly, name what's costing the distance to 40%.
- Task 6: show the actual bits for `0.1` in fp32 / bf16 / fp8 E4M3; pick a training format.
- Theme: a plausible number is not evidence — print / check / plot, never assert.

## Steps
1. **From-scratch tiny GPT** (causal attention + MLP, LayerNorm, tied-free head) + seeded
   Markov byte corpus. From scratch so shapes are mine to print and weights mine to probe.
2. **Task 1** — run one forward+loss, print every tensor shape with a dim-meaning line.
3. **Task 2** — float64 model, central difference on one `head.weight` element vs `backward()`.
4. **Task 3** — lecture's 4/4/2 single-step case (15.4%), then a real loop computing both loss
   combinations each step with a hard short batch; plot both; add an equal-count control.
5. **Task 4** — log pre-clip grad norm + clean eval loss each step; inject one anomaly batch;
   detect and report the step where the norm jumped while the loss stayed flat; plot twin axes.
6. **Task 5** — measure tokens/sec of the real loop (warmup + sync), compute `6·N·tok/s`, divide
   by a clearly-stated device peak; print MFU and the honest gap diagnosis.
7. **Task 6** — compute `0.1` bit patterns; verify against `torch.bfloat16` / `float8_e4m3fn`.
8. **Notebook** — `build_notebook.py` splits the script; execute with nbconvert (outputs+plots).
9. **Docs** — README (six answers), requirements, session-notes, this plan, research, WALKTHROUGH.

## Risks / watch-items
- **Position-embedding overrun** if corpus length ≠ model block → pass exactly `T`, keep block ≥ T.
- **Accumulation gap invisible** under AdamW if shown via trained models → show the *combination*
  gap each step instead (correct vs average-of-averages), with a hard short batch.
- **Gradient check only agrees to 3 decimals** in float32 → do the check in float64.
- **MFU peak unqueryable on MPS** → hard-code a labelled peak + caveat; emphasise the reasoning.
- **Notebook ≠ script drift** → generate the notebook, never hand-edit cells.

## Definition of done
Notebook runs top to bottom with outputs + both plots; gradient agrees to ~1e-8; accumulation
gap plotted (254% uneven / 0% equal); grad-norm spike localised to a step; MFU computed with an
honest diagnosis; `0.1` bits match the hardware. (User pushes to GitHub + pastes the link.)

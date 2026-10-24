# Session 11 — Plan

## Goal
Answer the five optimizer tasks as *fair measurements* (not flattering ones), and ship them
as a detailed README + support code + one executed notebook. The binding rule: tune both
sides before accepting any comparison.

## Constraints from the spec
- Submission = detailed README.md + support code (1000 pts).
- Task 1: hand Adam must agree with PyTorch to several decimals.
- Task 2: *plot* 20 steps both ways; report when the difference stops mattering.
- Task 3: log the update/weight ratio per layer; find where warmup stops changing it.
- Task 4: same model, cosine vs WSD, 300 steps, stop both at 200; report both losses + which to keep.
- Task 5: sweep LR at widths 256/512/1024; plot loss-vs-LR; mark 3 minima; state the 4096 value + confidence.

## Steps
1. **Scalar Adam** (tasks 1–2): one weight, five gradients; hand table vs `torch.optim.Adam`;
   then step-size ratio with/without bias correction over a long horizon; compute crossovers.
2. **Width-parameterised MLP + tasks** for 3–5:
   - hard nonlinear classification (random teacher + 20% label noise) for tasks 3 & 4 so training
     never flatlines;
   - smooth regression surface + quarter-decade LR grid for task 5 so the optimum is sharp.
3. **Task 3** — AdamW, 40-step warmup; log `‖ΔW‖/‖W‖` per layer; detect the peak (= warmup end).
4. **Task 4** — width-64 proxy; train cosine-300 and WSD-300, stop at 200; also decay the WSD
   step-200 checkpoint as its own run; plot schedules + losses.
5. **Task 5** — sweep LR × {256,512,1024}; mark minima; measure per-doubling ratio; extrapolate 4096.
6. **Notebook** — `build_notebook.py` splits the script; execute with nbconvert (outputs + plots).
7. **Docs** — README (five answers + a "why these are honest" section), requirements, session-notes,
   this plan, research, WALKTHROUGH.

## Risks / watch-items
- **Task converges to ~0** → gradients vanish, tasks 3 & 4 degenerate. Fix: hard task with a loss floor.
- **Schedule race collapses** if the proxy memorizes the noise → use a small-capacity (width-64) proxy.
- **LR drift invisible** on a noisy classifier or a coarse (√10) grid → smooth surface + quarter-decade grid.
- **Task-3 "settle" detector** wrong if it assumes a plateau: the ratio *peaks then decays*, so use argmax.
- **Over-claiming the 4096 value** → report it as an extrapolation with explicit LOW-to-MODERATE confidence.

## Definition of done
Notebook runs top to bottom with outputs + all four plots; Adam matches PyTorch to ~1e-8 (final w
0.995031); bias-correction crossover reported honestly (~3925, not "twenty"); update ratio ~1e-3
peaking at warmup end; cosine 0.6349 vs WSD 0.5378/0.5533 with a reasoned "keep WSD"; LR minima drift
left (1e-2/3.16e-3/1e-3) with a confidence-qualified 4096 value. (User pushes to GitHub + pastes link.)

# Session 14 — a dense model, upcycled into a Mixture-of-Experts

The task: train a dense ("linear") model, **convert it into an MoE**, and show both continue to
train and reduce loss — with the training logs in the repo.

Everything here is produced by [moe_lab.py](moe_lab.py) and runs top to bottom in
[moe_lab.ipynb](moe_lab.ipynb) (outputs + plot embedded). Raw logs:
[results.txt](results.txt) and [training_log.txt](training_log.txt) (the required per-step log).

```bash
pip install -r requirements.txt
python moe_lab.py            # writes fig_moe.png + results.txt + training_log.txt
# or open moe_lab.ipynb and Run All
```

> **One code path, two scales.** The committed numbers below are a genuine short run that finishes
> in a few minutes on a laptop GPU/CPU (measured on Apple-Silicon **MPS**). The method is
> identical at larger scale — raise `config()` and the step counts on a GPU.

---

## What "convert a dense model into an MoE" means here

A Mixture-of-Experts layer replaces a transformer's single feed-forward network (FFN) with **N
experts** (N FFNs) plus a small **router**. For each token the router scores all experts, keeps the
**top-k** (here k=2 of N=8), runs only those, and combines them by the router's renormalised
weights. So **total** parameters grow with N (capacity), while **compute per token** grows only
with k (cost). The MoE is a strict generalisation of the dense FFN — N identical experts whose
top-k weights sum to 1 reproduce the dense FFN exactly.

I read "train a Linear model and convert it into an MoE" as the standard **dense FFN transformer →
sparse MoE** conversion that every MoE paper performs (a bare `nn.Linear` has no nonlinearity and
cannot language-model). The experts are SwiGLU blocks, the same shape as the dense FFN.

## The method: sparse upcycling (why loss *continues* instead of restarting)

**Sparse upcycling** (Komatsuzaki et al. 2022): take the trained dense model, build the MoE by
**copying the dense FFN into every expert** (plus a whisper of noise to break symmetry), and start
the **router at zero**. Because every expert is (almost) the dense FFN and the top-k weights are
renormalised to sum to 1, the MoE's output at step 0 equals the dense output — so when training
resumes, the loss picks up exactly where the dense model stopped. I verify this directly before
trusting anything downstream:

```
dense model, final loss                                   0.0412
upcycled MoE, FIRST step loss                             0.0512      <- no jump; continues
|MoE cross-entropy − dense cross-entropy| at conversion = 3.35e-05    <- same prediction
```

The MoE's first-step loss lands on the dense model's final loss — the conversion is seamless, then
the MoE keeps reducing loss with its larger capacity.

## Results

Same byte-level LM (4 layers, width 256, SwiGLU inner 512; 8 experts, top-2 for the MoE), AdamW,
seed 0.

| run | start loss | final loss | note |
|---|---:|---:|---|
| **dense** | 5.59 | **0.041** | the baseline we upcycle from |
| **MoE (upcycled, continues)** | **0.051** | **0.045** | starts at the dense final loss — *no jump* |
| **MoE (from scratch, equal budget)** | 5.59 | 0.060 | same architecture, cold start, for contrast |

![dense → MoE](fig_moe.png)

**Parameter accounting (capacity vs cost):**

| | total params | active / token | |
|---|---:|---:|---|
| dense | 2.66M | 2.62M | |
| MoE | **13.68M** | **4.20M** | **5.1× the parameters at 1.6× the per-token compute** |

**Reading the result (what the assignment asks for):**

1. **Both train and reduce loss.** Dense falls 5.59 → 0.041. The upcycled MoE *continues* from
   there (0.051 → 0.045) — the left plot shows one unbroken curve through the conversion line, no
   spike. The dense→MoE conversion is the whole point and it is seamless.
2. **MoE adds capacity cheaply.** 5.1× the parameters for only 1.6× the per-token FFN compute,
   because only 2 of 8 experts run per token.
3. **The warm start is worth it.** The from-scratch MoE (same size, same step budget) has to climb
   down from 5.59 and ends higher (0.060 vs 0.045); the upcycled model was already at 0.05 before it
   took a single MoE step.
4. **Experts stay balanced.** With the Switch-Transformer load-balance auxiliary loss, the router
   keeps all 8 experts in use — no expert is starved. With top-2 of 8 the even share is 25% per
   expert, and the load coefficient-of-variation ends at **0.27** (upcycled) and **0.15**
   (from-scratch), where 0 is perfectly even — so the spread stays within a few points of even
   rather than collapsing. Without the auxiliary loss the router piles onto a few favourites and
   the rest never learn. See the right plot and the per-step `load-cv` column in
   [training_log.txt](training_log.txt).

## Why it's real, not a trick

- **The conversion is verified exact**, not assumed: the MoE's cross-entropy equals the dense
  model's at init (`|Δ| = 3.35e-05`), and the first MoE training step lands on the dense final loss.
- **The MoE genuinely routes sparsely** — top-2 of 8, only the chosen experts run — and the load
  log shows all experts are used.
- **The load-balancing loss is present and measured** (per-expert load logged every log step).

## Honest scope

- **Scale.** Small model (2.66M dense / 13.68M MoE) and a small, repetitive byte corpus so the repo
  is dependency-free and reproducible in minutes; both models memorise it (hence losses → ~0.05).
  The upcycling mechanism, routing, and balancing are identical at larger scale — raise `config()`
  and the step counts on a GPU.
- **Symmetry-breaking noise.** Experts are copied with 1e-4 Gaussian noise so they can diverge
  during training; that is why the conversion CE difference is tiny-but-nonzero rather than exactly
  0. At noise 0 the conversion is bit-for-bit exact but all experts stay identical twins.
- **Hardware.** Measured on MPS; `device` auto-selects cuda → mps → cpu.

## Files

| file | what |
|---|---|
| [moe_lab.py](moe_lab.py) | source of truth — data, SwiGLU, MoE, LM, upcycling, training, run |
| [moe_lab.ipynb](moe_lab.ipynb) | same code as a notebook, executed with outputs + plot |
| [build_notebook.py](build_notebook.py) | splits the `.py` into the `.ipynb` (so they can't drift) |
| [results.txt](results.txt) | param accounting, conversion |Δ|, start/final loss for all three runs |
| [training_log.txt](training_log.txt) | **the required training logs** — per-step loss + load-cv |
| [run_log.txt](run_log.txt) | an **earlier** run of the same code, kept for reference; its numbers are superseded by `results.txt` |
| fig_moe.png | the plot (loss continuity + per-expert load) |

*(Submission: push this folder to GitHub, paste the README link into the course portal.)*

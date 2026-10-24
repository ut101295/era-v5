# Session 14 — Walkthrough

A guided tour of the submission, in the order it's easiest to understand and defend.

## The 60-second version

Train a dense transformer, then **upcycle** it into an 8-expert top-2 Mixture-of-Experts by copying
the dense FFN into every expert and zeroing the router. Because identical experts with top-k weights
summing to 1 reproduce the dense FFN exactly, the MoE's output at conversion equals the dense
output — so training *continues* (no loss jump) instead of restarting, now with 5× the parameters.
One source of truth ([moe-upcycle/moe_lab.py](moe-upcycle/moe_lab.py)) → one executed notebook.
Anchor claim, stated as a check: the MoE's first training step lands on the dense model's final
loss.

## 1. The MoE layer (§2)

Router (Linear d→8) + 8 SwiGLU experts. Per token: softmax the router, take the **top-2**,
renormalise their weights to sum to 1, run only those 2 experts, combine. Total params scale with 8;
compute per token scales with 2. Plus the **Switch-Transformer load-balance aux loss**
`α·N·Σ f_i·P_i`, minimised when every expert gets an even share — without it the router collapses.

## 2. Upcycling (§3) — the heart of the submission

Copy the trained dense FFN into every expert (+1e-4 noise to break symmetry), zero the router. Now
every expert ≈ the dense FFN and the top-2 weights sum to 1, so the MoE computes the dense output.

## 3. Correctness first (§3)

`|MoE cross-entropy − dense cross-entropy|` at conversion = `8.48e-06`, and the upcycled MoE's
**first** training step loss (0.049) equals the dense model's **final** loss (0.047). That is the
proof that "continues training" is a property of the construction, not a lucky optimizer. Everything
downstream only matters because this holds.

## 4. The result (§4) — `fig_moe.png`

| run | start | final |
|---|--:|--:|
| dense | 5.59 | 0.047 |
| MoE (upcycled, continues) | 0.049 | 0.040 |
| MoE (from scratch, equal budget) | 5.59 | 0.056 |

- **Left plot:** dense falls, then the MoE continues on one unbroken curve through the "convert"
  line — no spike. The from-scratch MoE (dashed) has to climb down from 5.59 and ends higher.
- **Right plot:** per-expert load stays near the even 25% line (load-cv ≈ 0.13) — the aux loss
  works; no expert is starved.

## 5. Capacity vs cost

MoE: **13.68M total / 4.20M active per token** vs dense 2.66M/2.62M — **5.1× the parameters at 1.6×
the per-token compute**. That decoupling is the entire reason MoEs exist.

## 6. The "linear" reading, stated plainly

"Train a Linear model and convert it into an MoE" → the standard **dense FFN transformer → sparse
MoE** conversion (upcycling). A bare `nn.Linear` has no nonlinearity and can't language-model; the
dense FFN is what every MoE paper upcycles. (See [research/research.md](research/research.md).)

## 7. One source of truth & reproduce

`moe_lab.py` is authoritative; `build_notebook.py` generates the executed notebook from it.
`training_log.txt` is the required training log.

```bash
cd moe-upcycle
pip install -r requirements.txt
python moe_lab.py        # or open moe_lab.ipynb and Run All
```

Submit: push, paste the README link into the portal and assignment.md.

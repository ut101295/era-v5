# Session 13 — Walkthrough

A guided tour of the submission, in the order it's easiest to understand and defend.

## The 60-second version

A reversible transformer stores only the two boundary states of its layer stack and **rebuilds**
every layer's activations during the backward pass, so activation memory stops growing with
depth — in exchange for ~one extra forward per step. The deliverable *measures* that trade and
*proves* the backward is correct. One source of truth
([reversible-llm/reversible_lab.py](reversible-llm/reversible_lab.py)) → one executed notebook.
Anchor claim, stated as a test: the hand-written reversible backward computes the **same
gradients** as full autograd (to 1e-6), so the memory saving is free.

## 1. The rule (§2) — midpoint, not euler

`p[l+1] = p[l-1] + 2h·f(p[l])` adds the block output two layers back and evaluates `f` at the
state in between, so it inverts exactly: `p[l-1] = p[l+1] − 2h·f(p[l])`. The ordinary residual
(forward-Euler) can't be inverted — that's *why* the reversible variant is the midpoint rule.

## 2. Why the custom autograd is the whole point (§2)

Change only the forward and autograd still stores everything → zero saving. The real win is a
`torch.autograd.Function`: `forward` under `no_grad` keeps two tensors; `backward` reconstructs
each block's input, recomputes one block at a time, accumulates grads. Memory becomes O(1) in
depth.

## 3. Correctness first (§3)

Reconstruction `max|recon − input| = 7e-7`; custom-vs-autograd gradients agree to ~1e-6 → PASS.
Everything downstream is only trustworthy because this passes.

## 4. The core proof: memory vs depth (§4) — `fig_memory_vs_depth.png`

Standard activation memory rises linearly (88→676 MB over depths 4→32); reversible stays low
(14→103 MB, slope = parameter growth only). Hardware-independent — it's the *shape* that proves
the claim.

## 5. The three runs (§5–8) — `fig_training.png`

| run | batch | loss | tok/s | peak | floor |
|---|--:|--:|--:|--:|--:|
| standard | 16 | 0.059 | 8,695 | 552 | 204 |
| reversible | 16 | 0.067 | 5,877 | 218 | 203 |
| reversible max-batch | 64 | 0.666 | 6,560 | 257 | 199 |

- **Loss:** all fall; max-batch ends higher only because equal tokens → 1/4 the steps.
- **Memory:** 552→218 MB (2.5×) at the same batch; reversible is now *optimizer-bound* (+15 MB
  activations vs standard's +348). Max-batch fits 4× the batch in half standard's memory.
- **Speed:** reversible is ~48% slower/step — the lecture's 30–50% overhead, measured.

## 6. The variant answer, stated plainly

Midpoint/leapfrog reversibility worked; Euler (the ordinary residual) is the non-reversible
baseline. Stability needed `2h=1` + zero-init block outputs (identity start) and dropout 0.

## 7. One source of truth & reproduce

`reversible_lab.py` is authoritative; `build_notebook.py` generates the executed notebook from
it. `FULL=False` is the committed local verification; `FULL=True` is the 20M/50M-token Colab job.

```bash
cd reversible-llm
pip install -r requirements.txt
python reversible_lab.py        # or open reversible_lab.ipynb and Run All
```

Submit: run `FULL=True` on Colab, push, paste the README link into the portal and assignment.md.

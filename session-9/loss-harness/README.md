# Session 9 — the loss harness, made correct and observable

A cross-entropy next-token training harness with every step **made observable**, plus
a second (t+2) output head. One notebook that runs top to bottom.

- **Notebook (runs top to bottom):** [`loss_harness.ipynb`](loss_harness.ipynb) — committed *with outputs*.
- **Script form / logs:** [`harness.py`](harness.py), captured run in [`results.txt`](results.txt).

> The one rule of this assignment: a target shift in the wrong direction still
> produces a beautiful loss curve. So the harness **prints the token strings**, not
> the ids — you cannot eyeball an off-by-one in a wall of integers.

## How to run

```bash
pip install -r requirements.txt
python harness.py                 # prints every number below
# or open loss_harness.ipynb in Colab / Jupyter and Run All
```

Fully self-contained: a tiny word-level tokenizer, a ~105k-param causal transformer,
and a seeded first-order Markov corpus — no downloads. Numbers below were produced on
Apple MPS (`device` is auto-detected: cuda → mps → cpu); exact values shift slightly
by device but every relationship holds.

## Part 1 — the seven results

| # | Requirement | Result |
|---|-------------|--------|
| 1 | **Every tensor shape**, with each dim's meaning | `tokens [B,T]=(2,6)` → `hidden [B,T,d]=(2,6,64)` → `logits [B,T,V]=(2,6,14)` → shift to `(2,5,14)`/`(2,5)` → flatten to `(10,14)`/`(10,)` |
| 2 | **Verify the shift** with strings (input beside target) | `the→cat`, `cat→ran`, `ran→over`, … reads as a sensible next-word column; `assert target[i]==input[i+1]` passes |
| 3 | **Mask padding**, confirm contributing-token count changes | contributing tokens **10 → 7** (dropped 3 `<pad>`); loss 2.6629 → 2.6330 |
| 4 | **Pack two docs**, mask the boundary (trained model) | loss **5.6055 → 5.2413** when the boundary target is masked; the boundary term *alone* = **9.6114** (far above the ~5 within-doc terms) |
| 5 | **Perplexity**: untrained ≈ vocab | untrained CE **2.6573** → **PPL 14.26**, vocab = **14** (ln 14 = 2.639) ✓ |
| 6 | **Tied vs untied** head params | tied **105,088** / untied **105,984** → untied costs **896** extra (= one 64×14 matrix) |
| 7 | **Peak memory**: ordinary vs chunked CE | largest logits tensor **0.229 MB → 0.014 MB**, a **16.0×** smaller peak, *identical loss* |

### What each one is actually checking

- **(3) Padding.** Without `ignore_index`, `<pad>` targets are averaged into the loss,
  diluting the real signal and leaking gradient into padding. Masking makes the mean a
  mean over *real* tokens — and the contributing count visibly drops 10→7.
- **(4) Document packing.** When you pack `docA <eos> docB` into one sequence, the
  target that predicts docB's first word *from docA's `<eos>`* is a cross-document
  guess the model can never learn. On the trained model that single term is **9.61**
  nats — masking it removes a large, spurious contribution from the mean (5.61→5.24).
- **(5) Perplexity sanity.** A randomly-initialized LM should guess ~uniformly, i.e.
  loss ≈ ln(V) and PPL ≈ V. We get **14.26 vs 14** — the harness is wired correctly.
  (This only works because the model uses small-std init; default `nn.Embedding` init
  tied to the head makes an untrained model wildly over-confident and PPL explode —
  itself a bug this check would catch.)
- **(7) Chunked cross-entropy.** The memory villain is the full `[N, V]` logits
  activation. The hand-written chunked CE computes it `N/chunks` rows at a time, so the
  peak logits tensor is 16× smaller for the *same* loss. The win scales with vocab —
  tiny here (V=14), large for a real V≈50k. On CUDA the script also prints the real
  measured `torch.cuda.max_memory_allocated`.

## Part 2 — a second head predicting t+2

A second (untied) output head predicts token **t+2**; both losses are trained jointly
(sum). Per-head loss over 300 steps:

| step | loss1 (t+1) | loss2 (t+2) | sum |
|------|-------------|-------------|-----|
| 0   | 2.6301 | 2.6539 | 5.2840 |
| 50  | 1.0008 | 1.5179 | 2.5187 |
| 150 | 0.8136 | 1.2780 | 2.0917 |
| 300 | 0.6279 | 0.9613 | 1.5892 |

**What happens and why.** Both losses fall together, but **loss2 (t+2) stays clearly
above loss1 (t+1)**, and the gap *widens* as training sharpens the one-step predictor.
The training data is a first-order Markov chain, so the most recent token `x_t`
strongly pins `x_{t+1}` (low loss1), but `x_{t+2}` can only be reached by marginalizing
over the still-unknown `x_{t+1}`, which is more spread out (higher loss2). The far head
converges to a higher floor — the extra irreducible uncertainty of the longer jump.
This is the same reason speculative decoding and multi-token-prediction heads trust the
near token more than the far one. (It is the robust tendency for sequential data, not a
theorem for every distribution — on a bag of independent tokens the horizon wouldn't
matter, which is exactly why the corpus here has real sequential dependence.)

## Files
- `harness.py` — the harness as a runnable script (source of truth).
- `loss_harness.ipynb` — the notebook, generated from `harness.py`, executed with outputs.
- `build_notebook.py` — splits `harness.py` into the notebook (one source of truth).
- `results.txt` — captured console run (training logs).
- `requirements.txt` — just `torch`.

# Session 10 — the training loop, made to tell the truth about itself

One small model, one real step, and six checks that each produce a number which can
look perfectly plausible while being wrong. So we never trust the loss: we print the
shapes, hand-check a gradient, break the accumulation on purpose, watch the grad norm
move before the loss, compute our own MFU, and write `0.1` out to the bit.

- **Notebook (runs top to bottom):** [`train_loop.ipynb`](train_loop.ipynb) — committed *with outputs and plots*.
- **Script form / logs:** [`train_loop.py`](train_loop.py), captured run in [`results.txt`](results.txt).
- **Plots:** [`fig_accum.png`](fig_accum.png), [`fig_gradnorm.png`](fig_gradnorm.png).

## How to run

```bash
pip install -r requirements.txt
python train_loop.py                 # prints every number, writes both PNGs
# or open train_loop.ipynb in Colab / Jupyter and Run All
```

Self-contained: a from-scratch tiny GPT, a seeded Markov byte corpus, no downloads.
Numbers below are from Apple MPS (`device` auto-detects cuda → mps → cpu); exact
decimals shift by device and (for timing) by run, but every relationship holds.

---

## 1 · Every tensor shape, with what each dim means

`B` = batch (sequences), `T` = time (tokens per sequence), `D` = model width, `V` = vocab.

| Tensor | Shape | Meaning |
|---|---|---|
| `input_ids` | `(4, 16)` = `[B, T]` | B sequences, T tokens each |
| `targets` | `(4, 16)` = `[B, T]` | the next token for every position |
| token+pos embeddings | `(4, 16, 128)` = `[B, T, D]` | a D-vector per token |
| `logits` | `(4, 16, 256)` = `[B, T, V]` | a score per vocab item, per position |
| flattened logits | `(64, 256)` = `[B*T, V]` | rows for cross-entropy |
| flattened targets | `(64,)` = `[B*T]` | one target id per row |
| `loss` | `()` = scalar | the one number the optimiser minimises |

## 2 · One gradient, verified by hand

Nudge one weight by ±ε, measure how the loss moved (central difference), compare to what
`backward()` reported. Done in **float64** so they agree to many decimals — if they
didn't, the loss/target wiring would be suspect.

```
weight interrogated : head.weight[3,5]
loss(w-ε) = 3.3858608399      loss(w) = 3.3858608214      loss(w+ε) = 3.3858608029   (ε = 1e-6)
numeric  (ΔL / 2ε)  = -0.0184935556
analytic (backward) = -0.0184935553
relative difference = 1.3e-8      → they agree
```

## 3 · Breaking gradient accumulation on purpose

The lecture's exact single step first — three micro-batches holding **4, 4, 2** real
tokens with mean losses **2.0, 2.0, 5.0**:

```
correct  (Σloss / Σtokens) = (4·2 + 4·2 + 2·5) / 10 = 2.6000
naive    (mean of means)   = (2.0 + 2.0 + 5.0) / 3  = 3.0000   →  15.4% too high
```

Then the same error inside a real loop ([`fig_accum.png`](fig_accum.png)): two long
learnable micro-batches (120 tokens each) and one short, hard one (24 random tokens,
loss stays near `ln V`). The combined number — the one whose gradient drives the step —
is computed both ways every step:

| | final value |
|---|---|
| correct (Σloss / Σtokens) | **0.57** |
| naive (mean of means) | **2.00** |
| gap | **+254%** too high |
| **control: equal token counts** | correct 0.028 = naive 0.028, **gap 0.000** |

The short micro-batch holds one-tenth the tokens but gets a **full one-third vote**. The
last row is why the bug lived in every major framework until 2024: **make the token
counts equal and the error vanishes**, so casual testing never saw it.

## 4 · The grad norm moves before the loss

Log the pre-clip grad norm every step, and the loss on a fixed clean eval batch. At step
20 we feed one anomalous micro-batch (scrambled targets), no clipping
([`fig_gradnorm.png`](fig_gradnorm.png)):

```
grad norm at step 20:  0.935 → 4.155   (4.4× in one step)
eval loss step 20 → 21: 1.6992 → 1.7146  (+0.0154 — barely a twitch, a step later)
```

The **same** anomaly moved the norm 4.4× and the loss by 0.015. That is the whole reason
the grad norm earns a place on every dashboard: it is the sensitive early-warning trace;
the loss is nearly blind to the event that the norm screams about. (It's also what
gradient clipping watches — cap the norm and this spike never reaches the weights.)

## 5 · MFU, computed honestly

A forward+backward costs ≈ `6·N` FLOPs per token (`N` = parameters). MFU = achieved
FLOP/s ÷ machine peak.

```
parameters N        : 3,323,392
tokens / second     : ~38,500        (measured on this device, varies by run)
achieved            : ~0.77 TFLOP/s  (6·N·tok/s)
assumed device peak  : 28 TFLOP/s    (MPS — stated, not queried)
MFU                 : ~2.8%          (healthy is 35–50%)
```

**What's costing the distance to 40%**, honestly: the model is *tiny*, so the loop is
memory- and kernel-launch-bound, not compute-bound; it runs as an eager Python step loop
(no `torch.compile`); the batch is small; there is no fused/flash attention; and these are
generic kernels on a consumer device, not tuned datacenter GEMMs. None of that shows up in
the loss curve — which is exactly why MFU is worth logging. (For reference, the lecture's
9B-on-8×H100 example: achieved 648 TFLOP/s ÷ 7,912 peak = **8.2%**.)

## 6 · `0.1` written out in three formats

`0.1` has no exact binary form: `0.1 = 1.1001100110011…₂ × 2⁻⁴`, so sign = 0 and the
unbiased exponent = −4. Each format just divides its bits differently (verified against
the hardware):

| Format | sign · exp · mantissa (bits) | hex | actually stores |
|---|---|---|---|
| **fp32** (1·8·23) | `0 01111011 10011001100110011001101` | `0x3DCCCCCD` | 0.10000000149011612 |
| **bf16** (1·8·7) | `0 01111011 1001101` | `0x3DCD` | 0.10009765625 |
| **fp8 E4M3** (1·4·3) | `0 0011 101` | `0x1D` | 0.1015625 |

Note bf16 keeps **all 8 of fp32's exponent bits** (same `01111011`), trading mantissa for
range; fp8 E4M3 has only 4 exponent + 3 mantissa bits, so `0.1` rounds all the way to
`0.1015625`.

**Which would I train in? bf16.** It keeps fp32's full exponent range, so tiny
late-training gradients never flush to zero and there is no loss-scaling apparatus to tune
and get wrong (the trap fp16 sets). fp8 E4M3 is for throughput once a recipe is proven —
but only with per-block scaling and with attention left in higher precision, because
softmax amplifies the noise. fp32 stays as the master-copy / precision fallback, not the
per-step compute format.

---

## Files
- `train_loop.py` — the harness as a runnable script (source of truth).
- `train_loop.ipynb` — the notebook, generated from the script, executed with outputs + plots.
- `build_notebook.py` — splits `train_loop.py` into the notebook (one source of truth).
- `results.txt` — captured console run.
- `fig_accum.png`, `fig_gradnorm.png` — the two required plots.
- `requirements.txt` — `torch`, `matplotlib`.

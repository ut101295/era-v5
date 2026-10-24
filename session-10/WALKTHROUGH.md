# Session 10 — Walkthrough

A guided tour of the submission, in the order it's easiest to understand and defend.

## The 60-second version
The deliverable is a **training loop that tells the truth about itself**: one small
from-scratch GPT, one real step, and six checks where a plausible number could be hiding a
bug. Everything is generated from one file (`train_loop.py`) into one notebook that runs top
to bottom with outputs and both plots embedded. The theme throughout — straight from the
lecture — is *a plausible number is not evidence*, so nothing is asserted that isn't printed,
hand-checked, or plotted.

## 1. Shapes — know what every tensor is (§1)
The step prints each tensor and names its dims: `input_ids [B,T]` → embeddings `[B,T,D]` →
`logits [B,T,V]` → flattened `[B*T,V]` → scalar `loss`. If you can't say what each dimension
means, you can't debug the step; this is the floor everything else stands on.

## 2. One gradient, by hand (§2)
The headline check. Nudge `head.weight[3,5]` by ±ε, measure the central difference of the
loss, and compare to `backward()`. In float64 they agree to **1.3e-8 relative** — proof the
loss/target wiring is right. (The assignment warns a wrong-direction target still gives a
pretty loss curve; a hand-checked gradient is what actually rules that out.)

## 3. Break gradient accumulation on purpose (§3) — plot: `fig_accum.png`
First the lecture's exact case: micro-batches of 4, 4, 2 tokens → correct `2.6000` vs naive
average-of-averages `3.0000`, **15.4% wrong**. Then the same bug in a real loop: two long
learnable micro-batches and one short hard one, with the combined loss computed *both ways
every step* and plotted. The gap is **254%** — and the control with equal token counts is
**0.000%**, which is exactly why the bug survived in every major framework until 2024.

## 4. Grad norm moves before the loss (§4) — plot: `fig_gradnorm.png`
Log the pre-clip grad norm and a clean eval loss every step; inject one anomalous batch at
step 20 with no clipping. The norm jumps **4.4× in one step** while the loss twitches by
**+0.015** a step later. The plot makes the point visually: the orange norm spikes, the blue
loss glides on, nearly blind. That sensitivity is why grad norm is the earliest warning on a
dashboard (and what clipping watches).

## 5. MFU, honestly (§5)
`MFU = 6·N·tokens_per_sec / peak`. Measured on this device: ~0.77 TFLOP/s achieved against a
stated 28 TFLOP/s peak → **~2.8%** (healthy is 35–50%). The honest diagnosis of the gap to
40%: tiny model (memory-/launch-bound, not compute-bound), eager Python loop, small batch, no
fused attention or `torch.compile`, generic consumer-device kernels. The lecture's
9B-on-8×H100 reference (8.2%) is included for scale.

## 6. 0.1 to the bit (§6)
`0.1 = 1.1001100…₂ × 2⁻⁴`. Written out and **verified against the hardware**:
fp32 `0x3DCCCCCD`, bf16 `0x3DCD`, fp8 E4M3 `0x1D` (storing 0.1000000015, 0.10009766,
0.1015625). bf16 keeps all 8 of fp32's exponent bits — the reason to **train in bf16**: full
range, no loss-scaling apparatus to get wrong; fp8 only once proven, with block scaling and
attention in higher precision; fp32 as the master-copy fallback.

## 7. One source of truth
`train_loop.py` is authoritative. `build_notebook.py` splits it on section separators into
`train_loop.ipynb`, executed with outputs + plots embedded — script and notebook can't drift.

## 8. Reproduce
```bash
cd training-loop
pip install -r requirements.txt
python train_loop.py            # prints every number, writes both PNGs
# or: open train_loop.ipynb and Run All
```
Submit: push the repo, paste the GitHub README link into the README and `assignment.md`.

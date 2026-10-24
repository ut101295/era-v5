# Session 10 — Research notes

Background facts I relied on, with the reasoning so the harness is defensible.

## The gradient, and checking it numerically
Autograd computes `∂loss/∂w` analytically by the chain rule. The independent check is the
*definition* of the derivative: the central finite difference
`(L(w+ε) − L(w−ε)) / 2ε`, which has error `O(ε²)` (vs `O(ε)` for the one-sided form). Two
caveats make it agree to many decimals rather than a few:
- **Precision.** In float32 the loss itself carries ~7 significant digits, so differencing two
  nearly-equal losses loses most of them — agreement caps around 3 decimals. In **float64** the
  budget is ~16 digits, so central difference with ε≈1e-6 agrees to ~1e-8 relative.
- **Determinism.** No dropout, fixed data, same forward each call — otherwise the two loss
  evaluations differ for reasons unrelated to the weight nudge.

## Gradient accumulation and the average-of-averages bug
Gradient accumulation simulates a large batch that doesn't fit in memory: run several
micro-batches, let their gradients add up, step once. The **correct** reduction is total loss
over total tokens (`Σloss / Σtokens`), so every token has equal weight. The **bug**, present in
major frameworks until 2024, averaged the per-micro-batch *means* (`mean of means`), which gives
a short micro-batch the same vote as a long one. For the lecture's 4/4/2-token case that's
`2.6` vs `3.0` = 15.4% wrong. The error is **identically zero when all micro-batches hold the
same token count** — which is why it hid: casual tests use equal lengths. (Fixed across HF
Transformers / DeepSpeed / others in late 2024; the fix normalises by tokens, not by
micro-batch.) The gradient inherits whatever combination the loss used, so the wrong number is
also a wrong update direction, not just a wrong log line.

## Grad norm, clipping, and early warning
The **grad norm** is the L2 norm of all gradients concatenated. Gradient clipping
(`clip_grad_norm_`) rescales every gradient by `cap / norm` when `norm > cap`, preserving
direction and changing only length — a guard against one anomalous batch taking a huge step.
Beyond the clipping it enables, the norm is the **most sensitive trace on a dashboard**: an
instability shows up there, often thousands of steps before the loss visibly degrades (lecture
§12/§14). In this harness, under AdamW, a single scrambled batch spikes the raw norm 4.4× while
the eval loss barely moves — the norm sees what the loss can't. (Under plain SGD the same spike
would also take a large *step* and dent the loss shortly after; adaptive optimisers absorb the
step magnitude but not the norm signal.)

## MFU (Model FLOPs Utilisation)
`MFU = achieved_flops / peak_flops`, where a forward+backward pass costs ≈ `6·N` FLOPs per
token for `N` parameters (2·N forward multiply-adds ×1 + ×2 for backward ≈ 6·N; the standard
Chinchilla/Kaplan estimate, ignoring attention's sequence-length term, which is small for short
contexts). So `achieved = 6·N·tokens_per_sec`. A healthy large-scale run sits at **35–50%**; the
uncomfortable fact is the loss curve looks identical at 8% and 45% — you simply pay several times
over. Peak FLOP/s is device-specific (e.g. H100 ≈ 989 TFLOP/s bf16 dense); MPS doesn't expose a
clean peak, so I state one and foreground the reasoning. Common MFU killers on a toy/consumer
setup: tiny model (memory- and launch-bound, not compute-bound), eager execution (no
`torch.compile`/graph), small batch, unfused attention, and generic kernels.

## Floating-point formats
A float is `sign · exponent · mantissa`; exponent bits buy **range**, mantissa bits buy
**precision**, and the total is fixed. `0.1` is non-terminating in binary
(`1.100110011…₂ × 2⁻⁴`), so every finite format rounds it.
- **fp32** (8 exp, 23 mant): `0x3DCCCCCD` → 0.10000000149.
- **bf16** (8 exp, 7 mant): same exponent as fp32, fewer mantissa bits → `0x3DCD` → 0.10009766.
  Keeping fp32's **range** is why bf16 beat fp16 for training: fp16's 5 exponent bits underflow
  late-training gradients (< ~6e-8 → 0), forcing **loss scaling** (multiply loss by ~1024 before
  backward, divide after); bf16's floor (~1e-41) makes that apparatus unnecessary.
- **fp8 E4M3** (4 exp, 3 mant): `0x1D` → 0.1015625. ~1.2 decimal digits; usable in production
  (2026) with per-block scaling and attention kept in higher precision, since softmax amplifies
  quantisation noise.
Bit patterns here are verified against `torch.bfloat16` and `torch.float8_e4m3fn`, and fp32 via
`struct.pack('>f', …)`.

## The 16-bytes-per-weight memory picture (context for later sessions)
Training state per weight ≈ 16 bytes: bf16 weight (2) + bf16 grad (2) + fp32 master copy (4) +
Adam's two moments (8). So an 80 GB card holds ~5.4B params of *state* before activations —
the reason activation checkpointing (recompute in backward, ~30% more compute) and the
sharding of Sessions 12–13 exist.

## Sources (concepts)
- Kaplan et al. 2020 / Hoffmann et al. 2022 — the `6N` FLOPs-per-token estimate and MFU usage.
- Chowdhery et al. 2022 (PaLM) — popularised Model FLOPs Utilisation as a reported metric.
- Micikevicius et al. 2017 — mixed-precision training and loss scaling.
- bf16 (Google Brain) and the OCP FP8 (E4M3/E5M2) / NVFP4 format specs.
- The 2024 gradient-accumulation normalisation fix (HF Transformers / trainer ecosystem).

# Session 7 — Plan (Problem #3: dynamic-length Kronecker)

## Goal
Remove the Kronecker embedding's 32-byte wall so arbitrary-length words are
encoded without cropping or silent collisions, and prove it with a trained model.

## Decision log
- **Problem choice:** #3 (dynamic length). Avoided #4 (Fourier) deliberately — a
  public reference submission already does #4, so #3 keeps the work unmistakably ours.
- **Approach:** replace the discrete `onehot_32(pos)` byte-position factor with a
  continuous `phi(p)` (sinusoidal). Chosen over alternatives:
  - *chunked / hierarchical Kronecker* — handles length but reintroduces a window boundary.
  - *hash-folding positions mod N* — removes the wall but manufactures new collisions.
  - *learned position function* — viable, but a fixed sinusoid is the lecture's own
    §11 prescription and needs zero extra parameters, so it is the cleanest baseline to beat.

## Build order (done)
1. `embeddings.py` — baseline + functional codec, shared projection. ✅
2. `collision_test.py` — proof 1 (named pair) + proof 2 (script sweep). ✅
3. `model.py` — tiny causal transformer. ✅
4. `train.py` — proof 3 (long-word learnability) + proof 4 (accuracy vs length). ✅
5. `run_demo.py` + README + charts. ✅

## Proof strategy
Everything except the input codec is held identical (model, data, seed, optimizer),
so any measured gap is attributable to the codec alone. Three independent angles:
representation (cosine/collision counts), learnability (task accuracy), and
parameter cost (projection size).

## Result (reproducible)
Baseline collapses long Indic families to one code → 12.5% (chance) on >32-byte
words. Functional codec → 100%, PPL ~1.0, at half the projection parameters.

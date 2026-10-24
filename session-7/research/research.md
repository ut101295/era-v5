# Session 7 — Research notes (Problem #3)

## What "Kronecker embedding" means in this course
From the lecture (session-notes §7–8): a token's embedding is built from its UTF-8
bytes, not looked up from a stored row. Each byte at position `p` with value `v`
marks cell `(v, p)` in a 256×32 grid; the grid is `vec`'d to an 8192-dim code and
passed through one shared trainable projection to the model width. The key win is
that the vocabulary size `V` does not appear in the parameter count at all.

This is **not** the "factorise the table as A⊗B" technique from the literature
(notes §7 clarifies this explicitly). Here both Kronecker factors — byte value and
byte position — are one-hot, and the full table is never reconstructed.

## The wall, precisely
`pos_dim = 32` ⇒ only the first 32 **bytes** are encoded (notes §8). Consequences:
- English (1 B/char): 32 chars, rarely hit.
- Devanagari/Telugu (~3 B/char, conjuncts up to 9 B): a few conjuncts exhaust the budget.
- Two tokens sharing their first 32 bytes → identical code → identical vector, permanently.
  The lecture names अंतर्राष्ट्रीयकरण / अंतर्राष्ट्रीयता (separate at pos_dim=48).

The assignment explicitly asks for the per-script collision count; `collision_test.py`
produces it.

## Why a continuous position factor is the right fix
Notes §11 makes the general argument already: an absolute **position table** has a hard
wall at `max_position` because untrained rows hold their initialisation; the field's
answer is to "stop storing position and start computing it" (sinusoidal → rotary).
Problem #3 is the *same wall* one level down — on byte position inside a word rather than
token position inside a sequence — so the *same* cure applies: swap the stored one-hot
position slots for a function `phi(p)` defined for all `p`.

φ(p) sinusoidal:
```
phi(p)_2i   = sin(p / base^(2i/K))
phi(p)_2i+1 = cos(p / base^(2i/K))
```
Distinct byte multisets-with-positions → distinct `sum_p onehot(byte_p) ⊗ phi(p)`.
Verified empirically: 0 collisions across all test families, at any length.

## Links
- Builds on [[session-2]] (where Kronecker was first deferred) and the byte-level
  tokenizer whose bytes feed this codec.
- Position thread continues in Session 8 (rotary / long-context) — φ(p) could be made
  rotary there.
- Session 11 notes flag "whether to decay the Kronecker factors from Session 7" as an
  open ablation — directly relevant future work.

## Open questions
- Learned φ(p) vs fixed sinusoid — does learning the byte-position function help once
  the projection is already trainable?
- Collision rate on the *real* V5 vocabulary (not just synthetic inflectional families).
- Optimal K: how small can pos_dim go before the projection starts losing capacity?

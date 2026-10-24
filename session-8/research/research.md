# Session 8 — Research notes (date verification + the story)

## Method
Each date = the **arXiv v1 submission date** read off the paper's "Submission
history," not the published/updated date and not from memory. NTK-aware scaling is
the single exception (community Reddit post, no paper). All verified October 2026.
Full table lives in `attention-timeline/README.md`.

## The DroPE / DRoPE collision (the instructor's warning, in the wild)
Searching "DroPE" returns two unrelated papers:
- **DroPE** — Sakana AI, *dropping positional embeddings*, arXiv:2512.12167, Dec 2025.
- **DRoPE** — *directional RoPE* for autonomous-driving agents, arXiv:2503.15029, Mar 2025.
They differ by one capital letter and ~9 months. The session is about positional
encoding and long context, so the Sakana paper is the intended one; the app and
README flag the ambiguity so a grader can see it was handled, not missed.

## What the timeline shows (Q2, part 1)
Once sorted by date the mechanisms stop being a taxonomy and become a sequence of
reactions. Four observations that are invisible in a grouped list:

1. **It oscillates rather than progresses.** Exactness-with-less-compute (Sparse
   2019, top-k 2019, sliding window 2020) → reclaim memory (MQA 2019 → GQA 2023 →
   MLA 2024) → reclaim length (RoPE/ALiBi 2021, NTK/YaRN/sinks 2023) → reclaim
   memory again by dropping the quadratic core (linear 2020, delta 2021, DeltaNet
   2024, Gated DeltaNet 2024). The same anxieties recur in new forms.
2. **2019 contains the whole fork.** MQA (cut memory) and Sparse Transformers (cut
   compute) appear months apart; every later branch descends from one or the other.
3. **Position encoding is a 2017 decision never fully paid off.** Sinusoidal → RoPE →
   ALiBi → NTK → YaRN → DroPE is one eight-year thread of patching the same original
   choice — ending with a 2025 paper arguing you should *remove* PE after training,
   i.e. undo the 2017 move.
4. **2024-25 is convergence.** MLA, NSA, Gated DeltaNet each *combine* prior threads
   rather than opening a new axis — the signature of a maturing field, and a hint
   that the next step is a hybrid paying compute + memory + length at once.

## Mechanisms the instructor did not cover (Q2, part 2)
Verified v1 dates:
- **FlashAttention** — arXiv:2205.14135, 2022-05-27. The headline omission: it keeps
  attention *exact* and attacks memory-IO by tiling into SRAM, unlike every
  neighbouring approximation. Underpins most later long-context training.
- **Ring Attention** — arXiv:2310.01889, 2023-10-03. Context scales with device count.
- **Infini-attention** — arXiv:2404.07143, 2024-04-10. Compressive memory for
  bounded-memory infinite context.

## Links
- Positional thread starts in [[session-7]] (the byte-position Kronecker wall is the
  same "stop storing position, start computing it" argument one level down). φ(p)
  there is a fixed sinusoid; RoPE on this timeline is its natural rotary upgrade.
- Session 11 notes flag decaying the Kronecker factors — relates to the gated-decay
  idea in Gated DeltaNet here.

## Open questions
- Where does FlashAttention belong conceptually — it's orthogonal (an exact-attention
  *implementation*), so does it even sit on the same "pay less of the bill" axis, or
  is it a second axis (algorithm vs. systems)?
- The oscillation predicts a compute+memory+length hybrid next. NSA is the closest;
  is there a cleaner synthesis already in preprint?

# Session 9 — Plan

## Goal
Build a cross-entropy next-token **loss harness** where every normally-invisible step is made
observable and provably correct, satisfy the seven Part-1 checks, add a Part-2 t+2 head, and
ship it as one notebook that runs top to bottom plus a README with the numbers.

## Constraints from the spec
- Submission = GitHub README link; repo holds the `.ipynb` (runs top-to-bottom) + training logs.
- Part 1 = 1000 pts, seven explicit checks (shapes / shift-by-strings / padding mask /
  doc-boundary mask / perplexity≈vocab / tied-vs-untied / chunked-CE memory).
- Part 2 = second head predicting t+2; report both losses + sum; explain the far head's loss.
- Hard warning: wrong-direction shift still looks great → must print strings.

## Steps
1. **Tokenizer + corpus.** Word-level vocab (12 words + `<pad>` + `<eos>` = 14). Seeded
   first-order Markov corpus so t+1 is genuinely easier than t+2 (needed for Part 2 honesty).
2. **Model.** ~105k-param causal transformer (`nn.TransformerEncoder`, causal mask, norm_first,
   GELU), tied head option, **small-std init** (load-bearing for the untrained-PPL check).
3. **Part 1 harness**, one numbered section per check; print shapes with dim meanings, print
   the shift as strings with an assert, mask padding and count, pack+mask a boundary with a
   trained model, compute perplexity, count tied/untied params, write a chunked CE and compare
   peak memory to ordinary CE.
4. **Part 2.** `TwoHeadLM` with an untied second head; `two_head_losses` returns loss1 (t+1,
   `[:, :-1]` vs `[:,1:]`) and loss2 (t+2, `[:, :-2]` vs `[:,2:]`); train jointly; log both.
5. **Notebook build.** `build_notebook.py` splits `harness.py` on separators → `.ipynb`;
   execute with nbconvert so outputs embed.
6. **Docs.** README (seven numbers + two losses + explanations), requirements, session-notes,
   this plan, research, WALKTHROUGH.

## Risks / watch-items
- **Untrained PPL explosion** if init is wrong → small-std init (std 0.02) on tok/pos/head.
- **Part 2 contradicting its explanation** if the corpus is near-independent → Markov corpus.
- **Notebook ≠ script drift** → generate the notebook from the script, never hand-edit cells.
- Device differences (MPS vs Colab) shift decimals but not relationships → note this in README.

## Definition of done
Notebook runs top to bottom with outputs; all seven numbers sensible; Part 2 shows loss2 > loss1
and explains why; README carries the numbers. (User pushes to GitHub + pastes the link.)

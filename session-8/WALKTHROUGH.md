# Session 8 — Walkthrough

A guided tour of the submission, in the order it's easiest to understand and defend.

## The 60-second version
The app is a single `index.html` that lists **every attention mechanism from the
lecture in launch-date order**. Each card says what problem existed at that moment,
what the mechanism buys, what it costs, and when you'd pick it. Read top to bottom,
the timeline shows the field **oscillating** — exactness → memory → length → memory
again — which you cannot see from a grouped list. The hardest-graded part (dates) is
verified against each paper's arXiv v1 and sourced in the README.

## 1. Start where the spec says to — vanilla attention
`index.html` opens with a scaled-dot-product explainer and a small interactive
heatmap ("the cat sat on the mat", causal, temperature slider). The spec is explicit:
start with standard attention "because nothing after it makes sense without it."

## 2. The timeline — the actual deliverable
20 cards, chronological. Each is color-coded by **family** (baseline / compute /
memory / position / linear) and tagged in text. Filter chips isolate one family so
you can trace a single thread (e.g. just the position lineage: sinusoidal → RoPE →
ALiBi → NTK → YaRN → DroPE).

The content is the point, not the animation: every card has honest **buys / costs**.
If a technique had only pros, I wouldn't have understood it — e.g. ALiBi's recency
bias *is* its extrapolation trick and also its weakness on far lookback; MQA's memory
win *is* its quality loss.

## 3. The story — answers Q2 part 1
The closing section, "What the timeline shows," states the four things the date
ordering reveals (oscillation; the 2019 fork; the unbroken 2017 position thread
ending in "drop position"; 2024-25 convergence). This is the question the instructor
says is "the whole reason I am asking."

## 4. Dates — the part that's easiest to get wrong
Every date is the arXiv **v1** submission date, read off each paper's submission
history, tabulated in `README.md` with links. NTK-aware scaling is the one community-
sourced date (Reddit, no paper) and is flagged. The **DroPE vs DRoPE** ambiguity —
exactly the kind of trap the instructor warned about — is resolved explicitly in both
the app footer and the README.

## 5. The missed mechanism — answers Q2 part 2
**FlashAttention** (arXiv:2205.14135, 2022-05-27), with Ring Attention and
Infini-attention as extras, all with verified dates. FlashAttention is the
interesting omission because it's *exact* attention made cheap via memory-IO tiling —
a different axis from the approximations around it.

## 6. Color is computed, not eyeballed
Per the dataviz skill, the categorical palette was run through
`validate_palette.js` in **both** light and dark modes. The first attempt (violet
baseline) failed the dark CVD gate against blue; swapped to magenta (palette slot 5),
re-validated → all gates pass. Low-contrast light hues are covered by the relief rule
(text tag on every card).

## 7. Reproduce / deploy
```bash
cd attention-timeline
python3 -m http.server 8000     # http://localhost:8000/index.html
```
Deploy: drag `attention-timeline/` onto Netlify Drop; paste the URL into the README
and `assignment.md`.

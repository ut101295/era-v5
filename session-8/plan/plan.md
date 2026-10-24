# Session 8 — Plan

**Deliverable:** a web app showing every attention mechanism from the lecture,
**chronologically**, each framed as the answer to a problem that existed at that
moment, with honest pros/cons and "when you'd pick it." Live link + repo. (Q1 1000;
Q2 1000 bonus = the story + a mechanism he missed; Q3 250 optional social post.)

## The one thing that is graded hardest
Dates. The instructor got burned by a hallucinated date in Session 7 and corrected
it in front of the cohort. So: **every date verified against the paper's arXiv v1
submission history**, sources cited in the README. No date from memory.

## Steps
1. **Enumerate** the required mechanisms from the spec + anything missed.
2. **Verify each date** at its arXiv abstract page (v1 line), or the community post
   for NTK. Resolve the DroPE/DRoPE ambiguity explicitly.
3. **Write content** per mechanism: problem-at-that-moment, buys/costs (honest),
   pick-it-when. This is the real work — the trade-offs, not the animation.
4. **Build** a single self-contained `index.html`: vanilla-attention explainer at the
   top (the spec insists on starting there), then the dated timeline, then a
   synthesis section answering "what does the timeline show." Category filter + theme
   toggle + a small interactive attention heatmap.
5. **Color** via the dataviz skill: assign categorical hues in fixed order, then
   **run the validator** in both modes; fix until it passes. Relief rule for low-
   contrast hues (text tags on every card).
6. **Docs:** README with the date-source table + Q2 answer; this plan; research
   notes; WALKTHROUGH; fill session-notes with the Q2 narrative.
7. **Deploy** to Netlify, paste the live link into README + assignment.

## Scope decisions
- Keep it one file, no framework — matches the Session 1/2 pattern and makes "drag to
  Netlify" trivial; nothing to break at grade time.
- Interactive but not clever-for-its-own-sake (the spec: "a clear static page beats a
  broken clever one"). The attention heatmap is the one real interaction; filters and
  theme are utility.
- Color by *conceptual family* (baseline / compute / memory / position / linear),
  not by date — so the reader can isolate one thread, which is what makes the
  oscillation legible.

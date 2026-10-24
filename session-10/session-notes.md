# Session 10 — Notes

> Working notes: understanding of the task, design decisions, open questions, progress.

## Understanding the task

Session 10 is "take a small model and a real loop, and make it tell you the truth about
itself." The recurring warning from the lecture — **a plausible number is not evidence** —
is the whole assignment. Six checks, each one a place where training silently lies:

1. **Print every tensor shape** in a step, one line per dim meaning (B/T/D/V).
2. **Verify one gradient by hand** — nudge a weight, central finite difference, compare to
   `backward()`; they must agree to several decimals.
3. **Break gradient accumulation on purpose** — average-of-averages with uneven-length
   micro-batches, and *plot both curves* so the gap is visible, not asserted.
4. **Log the grad norm every step**, find a step where it moved before the loss did.
5. **Compute my own MFU**, report it honestly, say what's costing the distance to 40%.
6. **Write `0.1` by hand** in fp32, bf16, fp8 E4M3 (show the bits); say which I'd train in.

Submission = GitHub repo + README with the numbers + the `.ipynb`.

## Approach / design

- **One source of truth.** Everything is in `train_loop.py` (runnable script). The notebook
  is *generated* from it by `build_notebook.py` and executed with outputs + plots embedded,
  so script and notebook can't drift. Same pattern as Session 9.
- **From-scratch tiny GPT**, not `nn.Transformer`, so every tensor shape in the step is mine
  to print (task 1) and I can interrogate a specific weight (task 2).
- **Gradient check in float64** with central difference (ε=1e-6) → agreement ~1e-8 relative.
  Float32 would only agree to ~3 decimals and invite doubt.
- **Accumulation bug made visible, not asserted.** The faithful demonstration isn't two
  trained models (AdamW's per-coordinate normalisation washes the bias out — I tried, the
  gap was 0.2%). It's computing *both combinations of the same micro-batch losses every
  step* and plotting them: the quantity whose gradient drives the step is wrong by 254%
  when counts are uneven, and **exactly 0% when counts are equal** — which is why the bug
  hid in every framework until 2024.
- **Grad norm as the sensitive trace.** Under AdamW a one-off bad batch spikes the raw grad
  norm 4.4× while the loss barely twitches (+0.015) — the honest version of "moves before
  the loss": the norm is the early-warning trace, the loss is nearly blind to the same event.
- **MFU reported honestly** on local hardware (~2.8% on MPS) with a clearly-labelled stated
  peak, plus the lecture's 8.2% H100 reference, and a frank list of what's costing the gap
  (tiny model → memory/launch-bound, eager loop, small batch, no fused attention/compile).
- **Floats verified against the hardware** (`torch.bfloat16`, `torch.float8_e4m3fn`) so the
  by-hand bit patterns are provably correct, not just plausible.

## Progress log

- Wrote `train_loop.py` with six sections; first run hit index errors (passed `T+1` to the
  corpus helper, overran the position embedding) → fixed all call sites to pass exactly `T`.
- Task 3 first attempt (two trained models, AdamW) showed only a 0.2% gap → redesigned to
  plot both loss *combinations* each step with a deliberately hard short batch → 254% gap,
  plus an equal-count control at 0.000%.
- Task 4 first tried SGD to make the loss visibly jump after the spike; SGD trained the
  transformer poorly and the loss even dropped after the shock → reverted to AdamW and
  reframed around the norm being 4.4× more sensitive than the loss (faithful + clean plot).
- Verified the fp32/bf16/fp8 bit patterns match `torch` exactly before writing them down.
- Built + executed the notebook (8 code cells, all executed, 2 plots embedded, 0 errors).
- Wrote README (all six answers), requirements, this file, WALKTHROUGH, plan, research.

Remaining (mine): nothing code-side. **User's tasks:** push the repo to GitHub, paste the
repo/README link into the README + `assignment.md`.

## Open questions

- **MFU peak is a stated constant, not queried.** MPS doesn't expose a clean peak-FLOP/s the
  way CUDA does, so I hard-code a defensible 28 TFLOP/s with a visible caveat. On a grader's
  machine the MFU % will differ; the *reasoning* about the 40% gap is the real deliverable.
- Timing (tokens/sec, hence MFU) is non-deterministic run-to-run; the README gives a `~`
  value and the notebook shows the actual run. Everything else is seeded and reproducible.
- Task 4's "before the loss" is framed as the norm being *more sensitive* than the loss
  (4.4× vs +0.015) rather than a strict temporal lead with a big visible loss jump — that's
  the honest story under an adaptive optimiser; worth confirming the instructor accepts it.


---
title: "Axiom — Learning OS"
source: "https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms10mq7v3k8p5w2xn6de/lesson"
author:
published:
created: 2026-10-24
description: "A premium learning operating system for ambitious technical education."
tags:
  - "clippings"
---
[Axiom](https://axiom.theschoolofai.in/dashboard)

[Settings](https://axiom.theschoolofai.in/settings)

## Session 10: The Training Loop

## 1\. What this session is

Session 9 left us holding a single number. The model reads a batch, produces logits, compares them against the truth, and hands back one scalar called the loss.

<svg id="mermaid-31" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 620.3125px;" viewBox="0 0 620.3125 91" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-31_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-31_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-31_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-31_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-31_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-31_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-31_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-31_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-31_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-31_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-31_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-31_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 115.2 45.5 L 126.7 45.5 Q 138.2 45.5 138.2 45.5 L 138.2 45.5 Q 138.2 45.5 149.7 45.5 L 161.2 45.5" id="mermaid-31-L_A_B_0" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTE1LjE4NzUsInkiOjQ1LjV9LHsieCI6MTQwLjE4NzUsInkiOjQ1LjV9LHsieCI6MTY1LjE4NzUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-31_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 268.9 45.5 L 280.4 45.5 Q 291.9 45.5 291.9 45.5 L 291.9 45.5 Q 291.9 45.5 303.4 45.5 L 314.9 45.5" id="mermaid-31-L_B_C_0" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MjY4Ljg5MDYyNSwieSI6NDUuNX0seyJ4IjoyOTMuODkwNjI1LCJ5Ijo0NS41fSx7IngiOjMxOC44OTA2MjUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-31_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 417.0 45.5 L 428.5 45.5 Q 440.0 45.5 440.0 45.5 L 440.0 45.5 Q 440.0 45.5 451.5 45.5 L 463.0 45.5" id="mermaid-31-L_C_D_0" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NDE3LjAzMTI1LCJ5Ijo0NS41fSx7IngiOjQ0Mi4wMzEyNSwieSI6NDUuNX0seyJ4Ijo0NjcuMDMxMjUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-31_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_A_B_0" transform="translate(0, 0)"></g></g><g><g data-id="L_B_C_0" transform="translate(0, 0)"></g></g><g><g data-id="L_C_D_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-31-flowchart-A-0" data-look="classic" transform="translate(61.59375, 45.5)"><rect style="" x="-53.59375" y="-26.25" width="107.1875" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-23.59375, -11.25)"><rect></rect><foreignObject width="47.1875" height="22.5"><p>tokens</p></foreignObject></g></g><g id="mermaid-31-flowchart-B-1" data-look="classic" transform="translate(217.0390625, 45.5)"><rect style="" x="-51.8515625" y="-26.25" width="103.703125" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-21.8515625, -11.25)"><rect></rect><foreignObject width="43.703125" height="22.5"><p>model</p></foreignObject></g></g><g id="mermaid-31-flowchart-C-2" data-look="classic" transform="translate(367.9609375, 45.5)"><rect style="" x="-49.0703125" y="-26.25" width="98.140625" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-19.0703125, -11.25)"><rect></rect><foreignObject width="38.140625" height="22.5"><p>logits</p></foreignObject></g></g><g id="mermaid-31-flowchart-D-3" data-look="classic" transform="translate(539.671875, 45.5)"><rect style="" x="-72.640625" y="-37.5" width="145.28125" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-42.640625, -22.5)"><rect></rect><foreignObject width="85.28125" height="45"><p>loss<br>one number</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-31-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

One number, for a model with **536.9M weights in its output head alone** and several billion more behind it. Which raises the question we spend today on:

> **How does a single number reach back and move billions of weights, and while it runs for weeks, how do we know it is working?**

Two ideas have been quietly doing work in this course without ever being properly introduced, and today they finally get the introduction they deserve.

The first is **backpropagation**, the method by which that one loss value travels back through every layer and tells each weight how much it contributed to the mistake.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_1_one_neuron_1.html" width="100%" height="1100" title="widget_1_one_neuron_1"></iframe>

The second is the **floating point number**, which we have written as `bf16` since Session 7 as though its meaning were self-evident. It is not, and the newer formats only become readable once the original is clear.

<svg id="mermaid-32" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 769.171875px;" viewBox="0 0 769.171875 113.5" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-32_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-32_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-32_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-32_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-32_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-32_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-32_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-32_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-32_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-32_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-32_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-32_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 189.8 56.8 L 201.3 56.8 Q 212.8 56.8 212.8 56.8 L 212.8 56.8 Q 212.8 56.8 224.3 56.8 L 235.8 56.8" id="mermaid-32-L_S9_S10_0" style=";" data-edge="true" data-et="edge" data-id="L_S9_S10_0" data-points="W3sieCI6MTg5Ljc2NTYyNSwieSI6NTYuNzV9LHsieCI6MjE0Ljc2NTYyNSwieSI6NTYuNzV9LHsieCI6MjM5Ljc2NTYyNSwieSI6NTYuNzV9XQ==" data-look="classic" marker-end="url(#mermaid-32_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 451.2 56.8 L 462.7 56.8 Q 474.2 56.8 474.2 56.8 L 474.2 56.8 Q 474.2 56.8 485.7 56.8 L 497.2 56.8" id="mermaid-32-L_S10_S11_0" style=";" data-edge="true" data-et="edge" data-id="L_S10_S11_0" data-points="W3sieCI6NDUxLjE3MTg3NSwieSI6NTYuNzV9LHsieCI6NDc2LjE3MTg3NSwieSI6NTYuNzV9LHsieCI6NTAxLjE3MTg3NSwieSI6NTYuNzV9XQ==" data-look="classic" marker-end="url(#mermaid-32_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_S9_S10_0" transform="translate(0, 0)"></g></g><g><g data-id="L_S10_S11_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-32-flowchart-S9-0" data-look="classic" transform="translate(98.8828125, 56.75)"><rect style="" x="-90.8828125" y="-37.5" width="181.765625" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-60.8828125, -22.5)"><rect></rect><foreignObject width="121.765625" height="45"><p>Session 9<br>prediction to loss</p></foreignObject></g></g><g id="mermaid-32-flowchart-S10-1" data-look="classic" transform="translate(345.46875, 56.75)"><rect style="" x="-105.703125" y="-37.5" width="211.40625" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-75.703125, -22.5)"><rect></rect><foreignObject width="151.40625" height="45"><p>Session 10<br>loss to weight update</p></foreignObject></g></g><g id="mermaid-32-flowchart-S11-2" data-look="classic" transform="translate(631.171875, 56.75)"><rect style="" x="-130" y="-48.75" width="260" height="97.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-100, -33.75)"><rect></rect><foreignObject width="200" height="67.5"><p>Session 11<br>which optimiser, which schedule</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-32-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

Every section today answers one of two questions.

> **Two questions:** What does one training step actually do? And what does it cost V5 to take that step, over and over, for weeks?

## 2\. The words we will need

Before we follow a loss backwards, it is worth agreeing on the words for the pieces involved, because they are used loosely almost everywhere else and the confusion is avoidable.

A **gradient** is one number attached to one weight, saying how much the loss would change if that single weight moved.

A **step** is one complete pass of the loop: read some data, compute the loss, compute every gradient, then move every weight.

A **batch** is the group of samples read in a single step.

A **micro-batch** is what a batch gets broken into when the whole thing will not fit in memory at once, which for a model of our size is nearly always.

> **Carry this forward:** a gradient belongs to one weight, and a step moves all of them at once.

## 3\. Where a gradient comes from

We can begin without any calculus at all.

Imagine a model with a single weight sitting at 3.0, and a loss measuring how wrong its answer currently is. Raise that weight a very small amount and compute the loss again, and the size of the shift turns out to be the most useful number in all of training.

| Weight | Loss |
| --- | --- |
| 3.000 | 16.000 |
| 3.001 | 16.064 |

The loss rose by $0.064$ when the weight rose by $0.001$, so:

$$
\frac{0.064}{0.001} = 64
$$

The loss climbs about **64 units for every unit we raise this weight**, which also tells us that to bring it down we should move the weight the other way. A large value means this weight matters a great deal right now, and a value near zero means the optimiser can leave it alone.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_4_training_loop_1.html" width="100%" height="1240" title="widget_4_training_loop_1"></iframe>

> A training loop visualised one step at a time. A small network attempts to predict a value, the loss is displayed as a coloured bar, and as you click "step" the loss decreases while the weights inside the network change colour to show which of them moved and in which direction. Goal: you should leave with a working sense of the cycle that runs through every training step, which is a forward pass that produces a prediction, a comparison against the correct answer, and a backward pass that updates every weight in the network.

> **Carry this forward:** a gradient is the answer to a nudge. Which way, and how urgently.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_3_nudge.html" width="1280" height="726" title="s10_widget_3_nudge"></iframe>

> **The nudge, done live.** Move the weight, then change how big the nudge is, and watch the slope settle on the same answer from either side.

## 4\. Following the loss backwards

A real model is a long chain of steps, and a weight near the front never touches the loss directly. It has to work through everything downstream of it.

<svg id="mermaid-33" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 646.703125px;" viewBox="0 0 646.703125 68.5" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-33_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-33_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-33_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-33_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-33_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-33_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-33_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-33_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-33_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-33_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-33_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-33_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 76.2 34.3 L 87.7 34.3 Q 99.2 34.3 99.2 34.3 L 99.2 34.3 Q 99.2 34.3 110.7 34.3 L 122.2 34.3" id="mermaid-33-L_X_H_0" style=";" data-edge="true" data-et="edge" data-id="L_X_H_0" data-points="W3sieCI6NzYuMjAzMTI1LCJ5IjozNC4yNX0seyJ4IjoxMDEuMjAzMTI1LCJ5IjozNC4yNX0seyJ4IjoxMjYuMjAzMTI1LCJ5IjozNC4yNX1d" data-look="classic" marker-end="url(#mermaid-33_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 252.8 34.3 L 264.3 34.3 Q 275.8 34.3 275.8 34.3 L 275.8 34.3 Q 275.8 34.3 287.3 34.3 L 298.8 34.3" id="mermaid-33-L_H_Y_0" style=";" data-edge="true" data-et="edge" data-id="L_H_Y_0" data-points="W3sieCI6MjUyLjc2NTYyNSwieSI6MzQuMjV9LHsieCI6Mjc3Ljc2NTYyNSwieSI6MzQuMjV9LHsieCI6MzAyLjc2NTYyNSwieSI6MzQuMjV9XQ==" data-look="classic" marker-end="url(#mermaid-33_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 432.6 34.3 L 444.1 34.3 Q 455.6 34.3 455.6 34.3 L 455.6 34.3 Q 455.6 34.3 467.1 34.3 L 478.6 34.3" id="mermaid-33-L_Y_L_0" style=";" data-edge="true" data-et="edge" data-id="L_Y_L_0" data-points="W3sieCI6NDMyLjYwOTM3NSwieSI6MzQuMjV9LHsieCI6NDU3LjYwOTM3NSwieSI6MzQuMjV9LHsieCI6NDgyLjYwOTM3NSwieSI6MzQuMjV9XQ==" data-look="classic" marker-end="url(#mermaid-33_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_X_H_0" transform="translate(0, 0)"></g></g><g><g data-id="L_H_Y_0" transform="translate(0, 0)"></g></g><g><g data-id="L_Y_L_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-33-flowchart-X-0" data-look="classic" transform="translate(42.1015625, 34.25)"><rect style="" x="-34.1015625" y="-26.25" width="68.203125" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-4.1015625, -11.25)"><rect></rect><foreignObject width="8.203125" height="22.5"><p>x</p></foreignObject></g></g><g id="mermaid-33-flowchart-H-1" data-look="classic" transform="translate(189.484375, 34.25)"><rect style="" x="-63.28125" y="-26.25" width="126.5625" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-33.28125, -11.25)"><rect></rect><foreignObject width="66.5625" height="22.5"><p>h = w1 · x</p></foreignObject></g></g><g id="mermaid-33-flowchart-Y-2" data-look="classic" transform="translate(367.6875, 34.25)"><rect style="" x="-64.921875" y="-26.25" width="129.84375" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-34.921875, -11.25)"><rect></rect><foreignObject width="69.84375" height="22.5"><p>y = w2 · h</p></foreignObject></g></g><g id="mermaid-33-flowchart-L-3" data-look="classic" transform="translate(560.65625, 34.25)"><rect style="" x="-78.046875" y="-26.25" width="156.09375" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-48.046875, -11.25)"><rect></rect><foreignObject width="96.09375" height="22.5"><p>loss = (y − t)²</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-33-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

Here $t$ is the **target**, meaning the answer we wanted. If we feed in 2 and hope for 20, then $t$ is 20 and everything the model does is measured against it.

Putting real numbers through:

| Quantity | Calculation | Value |
| --- | --- | --- |
| $h$ | $w_1 \times x = 3 \times 2$ | 6 |
| $y$ | $w_2 \times h = 4 \times 6$ | 24 |
| loss | $(24 - 20)^2$ | 16 |

If we nudge $w_1$, how much does the loss move? Since it reaches the loss only through $h$ and then $y$, we walk back one link at a time, multiplying as we go.

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial y}  &= 2(y - t) = 8 \\
\frac{\partial \mathcal{L}}{\partial w_2} &= 8 \times h = 48 \\
\frac{\partial \mathcal{L}}{\partial h}  &= 8 \times w_2 = 32 \\
\frac{\partial \mathcal{L}}{\partial w_1} &= 32 \times x = 64
\end{aligned}
$$

That is the whole of backpropagation, and it sounds grander than it is: **one number comes back from the loss, and each link multiplies it by one local factor.** Nudging $w_1$ and measuring, as we did above, gives **64.0000**, agreeing to eight decimals.

> **Carry this forward:** backpropagation is the chain rule, applied one link at a time, from the loss towards the inputs.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_4_chain.html" width="1280" height="638" title="s10_widget_4_chain"></iframe>

> **One link at a time.** Step backwards through the chain and watch each multiplication as it happens, ending on the 64 we found by nudging.

## 5\. What autograd is doing for us

Four links by hand is pleasant enough, but several hundred layers is not, so the computer keeps notes for us. Every time it multiplies or adds during the forward pass it records what it did and what went in, and that record is the **computation graph**.

Walking it backwards while multiplying by each local factor is precisely what `loss.backward()` does. The machinery keeping those notes is **autograd**, and it is not doing anything cleverer than the section above, simply doing it a few billion times without losing its place.

> **Carry this forward:** autograd is bookkeeping, not magic. The mathematics is the chain rule you just did by hand.

## 6\. One step, start to finish

Everything above assembles into five lines, and every training run in the world is these five lines repeated until the data runs out.

```
logits = model(batch)          # forward
loss   = loss_fn(logits, y)    # one number
loss.backward()                # fill in every gradient
optimizer.step()               # move every weight
optimizer.zero_grad()          # wipe before the next batch
```

That last line surprises people. **Gradients add up rather than replace**, so each call to `backward()` piles the new gradient onto whatever was already there, and forgetting to wipe means batch two trains on top of batch one.

The loss will still fall, the model will simply be learning from a stale blend of everything it has seen, and nothing will raise an error. That looks like a defect until the next section, which is why it was built this way on purpose.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_1_step.html" width="1280" height="698" title="s10_widget_1_step"></iframe>

> **The whole step on one screen.** Press Step and watch the loss return as gradients, the gradients become an update, and the weights move. Change the target and the gradients change sign. The two bars beneath each weight are what the optimiser keeps between steps, and Session 11 explains why.

> **Carry this forward:** a step is forward, loss, backward, update, wipe. The wipe is not optional.

## 7\. Borrowing a bigger batch

There is a tension at the centre of every run. We want a large batch, because averaging over more samples gives a steadier gradient, and the accelerator wants a small one, because a large batch will not fit.

**Gradient accumulation** settles it using the behaviour that looked like a bug a moment ago. Run several small batches in turn, let their gradients pile up, and take one step once they have all contributed.

<svg id="mermaid-34" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 772.96875px;" viewBox="0 0 772.96875 91" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-34_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-34_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-34_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-34_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-34_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-34_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-34_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-34_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-34_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-34_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-34_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-34_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 164.9 45.5 L 176.4 45.5 Q 187.9 45.5 187.9 45.5 L 187.9 45.5 Q 187.9 45.5 199.4 45.5 L 210.9 45.5" id="mermaid-34-L_M1_M2_0" style=";" data-edge="true" data-et="edge" data-id="L_M1_M2_0" data-points="W3sieCI6MTY0LjkyMTg3NSwieSI6NDUuNX0seyJ4IjoxODkuOTIxODc1LCJ5Ijo0NS41fSx7IngiOjIxNC45MjE4NzUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-34_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 374.9 45.5 L 386.4 45.5 Q 397.9 45.5 397.9 45.5 L 397.9 45.5 Q 397.9 45.5 409.4 45.5 L 420.9 45.5" id="mermaid-34-L_M2_M3_0" style=";" data-edge="true" data-et="edge" data-id="L_M2_M3_0" data-points="W3sieCI6Mzc0Ljg5MDYyNSwieSI6NDUuNX0seyJ4IjozOTkuODkwNjI1LCJ5Ijo0NS41fSx7IngiOjQyNC44OTA2MjUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-34_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 585.0 45.5 L 596.5 45.5 Q 608.0 45.5 608.0 45.5 L 608.0 45.5 Q 608.0 45.5 619.5 45.5 L 631.0 45.5" id="mermaid-34-L_M3_S_0" style=";" data-edge="true" data-et="edge" data-id="L_M3_S_0" data-points="W3sieCI6NTg0Ljk2ODc1LCJ5Ijo0NS41fSx7IngiOjYwOS45Njg3NSwieSI6NDUuNX0seyJ4Ijo2MzQuOTY4NzUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-34_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_M1_M2_0" transform="translate(0, 0)"></g></g><g><g data-id="L_M2_M3_0" transform="translate(0, 0)"></g></g><g><g data-id="L_M3_S_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-34-flowchart-M1-0" data-look="classic" transform="translate(86.4609375, 45.5)"><rect style="" x="-78.4609375" y="-26.25" width="156.921875" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-48.4609375, -11.25)"><rect></rect><foreignObject width="96.921875" height="22.5"><p>micro-batch 1</p></foreignObject></g></g><g id="mermaid-34-flowchart-M2-1" data-look="classic" transform="translate(294.90625, 45.5)"><rect style="" x="-79.984375" y="-26.25" width="159.96875" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-49.984375, -11.25)"><rect></rect><foreignObject width="99.96875" height="22.5"><p>micro-batch 2</p></foreignObject></g></g><g id="mermaid-34-flowchart-M3-2" data-look="classic" transform="translate(504.9296875, 45.5)"><rect style="" x="-80.0390625" y="-26.25" width="160.078125" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-50.0390625, -11.25)"><rect></rect><foreignObject width="100.078125" height="22.5"><p>micro-batch 3</p></foreignObject></g></g><g id="mermaid-34-flowchart-S-3" data-look="classic" transform="translate(699.96875, 45.5)"><rect style="" x="-65" y="-37.5" width="130" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-35, -22.5)"><rect></rect><foreignObject width="70" height="45"><p>step once<br>then wipe</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-34-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

| Quantity | Meaning | Example |
| --- | --- | --- |
| micro-batch | what actually fits | 8 sequences |
| accumulation steps | how many we gather | 4 |
| **global batch** | what the optimiser sees | **32 sequences** |

For scale, V4 trained with a micro-batch of **7 per GPU and a global batch of 56** across one eight-GPU node, which is a good deal smaller than most people imagine.

> **Carry this forward:** the batch the optimiser sees is a choice, and it is independent of what fits in memory.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_5_accumulate.html" width="1280" height="645" title="s10_widget_5_accumulate"></iframe>

> **Gradients piling up.** Add micro-batches one at a time and watch the buffer fill, then compare the result against the same examples run as a single large batch.

## 8\. A mistake worth studying

A bug lived inside every major training framework until 2024, and it is worth walking through because the lesson generalises far past this one case.

Take three micro-batches holding different numbers of real tokens, as sequences naturally differ in length.

| Micro-batch | Valid tokens | Average loss |
| --- | --- | --- |
| 1 | 4 | 2.0 |
| 2 | 4 | 2.0 |
| 3 | 2 | 5.0 |

The correct combination adds up all the loss and divides by all the tokens, so every token carries equal weight:

$$
\frac{4(2.0) + 4(2.0) + 2(5.0)}{4 + 4 + 2} = \frac{26.0}{10} = 2.6000
$$

What the frameworks did instead was average the three averages:

$$
\frac{2.0 + 2.0 + 5.0}{3} = 3.0000
$$

The short micro-batch, holding half as many tokens, was given exactly the same vote as the long ones, and the result comes out **15.4% wrong**.

What makes it worth a section is how it hid. The error vanishes whenever micro-batches happen to hold equal token counts, which in casual testing they usually do, so the loss curves looked reasonable while being wrong.

> **Carry this forward:** a number that looks plausible is not evidence. This is the theme of the rest of the session.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_6_wrong_average.html" width="1280" height="646" title="s10_widget_6_wrong_average"></iframe>

> **The bug, and how it hid.** Both ways of combining the losses, side by side, at 15.4% apart. Now make the token counts equal and watch the error drop to zero.

## 9\. How a computer holds a number

A computer has a fixed number of bits and must store values as large as 50,000 and as small as 0.00000001 in the same box. Plain counting cannot stretch that far, so the bits are divided between three jobs.

| Piece | Its job | Bits |
| --- | --- | --- |
| **sign** | positive or negative | always 1 |
| **exponent** | how big the number is | a few |
| **mantissa** | the digits themselves | whatever remains |

It is the arrangement we use writing $6.02 \times 10^{23}$, where the power of ten says how large, and the 6.02 says which.

$$
\text{fp32} = 1 \text{ sign} + 8 \text{ exponent} + 23 \text{ mantissa} = 32 \text{ bits}
$$

Every format we will meet is simply a different way of dividing the bits between those last two jobs, and the trade is always the same one.

> **Carry this forward: exponent bits buy range, mantissa bits buy detail.** More of one always means less of the other.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_2_floats.html" width="1280" height="749" title="s10_widget_2_floats"></iframe>

> **Every step of the construction, shown.** The exponent bits added from their place values, the bias taken off, the mantissa read as a binary fraction, then the three pieces multiplied together. Click any bit to flip it, or build your own format with the two sliders.

## 10\. Why bf16 replaced fp16

Three formats matter for training, and one column of this table explains why the industry settled where it did.

| Format | Bits | Exponent | Mantissa | Smallest it holds | Decimal digits |
| --- | --- | --- | --- | --- | --- |
| fp32 | 32 | **8** | 23 | $1.18 \times 10^{-38}$ | 7.2 |
| fp16 | 16 | **5** | 10 | $6.1 \times 10^{-5}$ | 3.3 |
| bf16 | 16 | **8** | 7 | $1.18 \times 10^{-38}$ | 2.4 |

Look at the exponent column. **bf16 keeps all eight exponent bits that fp32 had**, so it reaches just as far, paying for that reach with detail, from 7.2 decimal digits down to 2.4.

fp16 made the opposite bargain, and that is what breaks it. Late in a run the gradients become genuinely tiny, and fp16 cannot hold anything below about $5.96 \times 10^{-8}$.

| Gradient | In fp16 | In bf16 |
| --- | --- | --- |
| $10^{-4}$ | fine | fine |
| $10^{-6}$ | losing digits | fine |
| $10^{-8}$ | **becomes exactly zero** | fine |

A gradient of zero means that weight does not move, so the model quietly stops learning exactly where the signal was faintest, which is usually where something was left to learn.

The rescue is **loss scaling**, as blunt as it sounds. Multiply the loss by about 1024 before the backward pass so the gradients clear the floor, then divide back down before the optimiser sees them.

$$
10^{-8} \times 1024 = 1.02 \times 10^{-5} \quad\text{survives, then divide back}
$$

It works, and it is one more setting to tune and eventually get wrong. bf16's floor sits at $9.18 \times 10^{-41}$, which nothing in training will ever reach, so the whole apparatus becomes unnecessary.

> **Carry this forward:** bf16 is less accurate than fp16 and won anyway, because range mattered more than digits.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_7_underflow.html" width="1280" height="693" title="s10_widget_7_underflow"></iframe>

> **A gradient shrinking.** Slide it down and watch fp16 give up while bf16 keeps going. Then switch loss scaling on and watch the same gradient survive.

## 11\. The newer formats

Once the trade is clear, the 2026 formats read easily.

| Format | Bits | Exponent | Mantissa | Decimal digits |
| --- | --- | --- | --- | --- |
| fp8 E4M3 | 8 | 4 | 3 | 1.2 |
| fp8 E5M2 | 8 | 5 | 2 | 0.9 |
| fp4 E2M1 | 4 | 2 | 1 | 0.6 |

Six tenths of a decimal digit ought to worry you, and on its own it should. **fp4 works only because its numbers are never alone.** Sixteen of them share a single exponent, stored once for the whole block.

$$
\frac{16 \times 4 \text{ bits} + 8 \text{ bits of shared scale}}{16} = 4.5 \text{ bits per value}
$$

This is the same idea as `int8`, which the widget shows in its last row. An `int8` tensor has no exponent anywhere in it, only whole numbers and one scale shared by everything.

As of 2026, **fp8 is a production recipe** using `E4M3` throughout, and **NVFP4 runs about 1.73 times faster than fp8**, though it needs five things working together to converge, including leaving attention in higher precision because softmax amplifies whatever noise you hand it.

> **Carry this forward:** we do not shrink everything equally. We shrink where the error does not accumulate.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_8_block_scale.html" width="1280" height="764" title="s10_widget_8_block_scale"></iframe>

> **Sixteen numbers sharing one exponent.** Change the block size and watch the cost per value fall, then put an outlier in and watch what sharing costs you.

## 12\. Guarding against one bad batch

A single unusual batch can end a run that has gone well for weeks. The gradients come back enormous, the weights take a stride far outside anything sensible, and the model lands somewhere it will not recover from.

The guard is a cap. We measure the combined size of all the gradients, and when it exceeds a threshold we shrink every one by the same factor, leaving the direction untouched and changing only the length.

$$
\text{norm} = 8.4, \quad \text{cap} = 1.0, \quad \text{so scale everything by } \frac{1.0}{8.4} = 0.119
$$

That combined figure is called the **grad norm**, and it turns out to be worth rather more than the clipping it enables.

> **Carry this forward:** the grad norm moves before the loss does. A run about to fail often shows it there thousands of steps early, which makes it the most useful trace on any dashboard.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_9_clip.html" width="1280" height="608" title="s10_widget_9_clip"></iframe>

> **The norm and the loss, on one axis.** The spike arrives in the norm at step 24 and only reaches the loss at step 26. Set the cap and watch it hold.

## 13\. What a step costs us

Running a finished model requires only its weights. Training one requires far more, and this arithmetic sets up the next three sessions.

| What must be held | Bytes per weight |
| --- | --- |
| the weight itself, in bf16 | 2 |
| its gradient, in bf16 | 2 |
| a full-precision copy, in fp32 | 4 |
| the optimiser's two running numbers | 8 |
| **total** | **16** |

Sixteen bytes for every single weight, and that is before a single activation is stored.

| Model | Training state alone |
| --- | --- |
| 2B | 29.8 GiB |
| 9B | 134.1 GiB |
| 20B | 298.0 GiB |
| 120B | 1,788 GiB |

An 80 GB accelerator therefore holds about a **5.4B** model and has nothing left over for the activations it also needs.

**Activation checkpointing** buys some back by discarding intermediates during the forward pass and recomputing them during the backward pass, costing about 30% more compute for a large share of the memory.

> **Carry this forward:** 16 bytes per weight is why Sessions 12 and 13 exist. One card is not enough, and this table is the reason.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_10_memory.html" width="1280" height="546" title="s10_widget_10_memory"></iframe>

> **Five things held for every weight.** Slide the model size up and find the point where one 80 GB card runs out, which happens sooner than most people expect.

## 14\. Watching a run that lasts weeks

Four traces, each catching something the others cannot.

| Trace | What it catches |
| --- | --- |
| loss | whether the model is learning at all |
| **grad norm** | trouble, before the loss shows it |
| tokens per second | whether something quietly got slower |
| **MFU** | whether the machine is being wasted |

The last needs explaining, because it is the one most often skipped.

**MFU**, or Model FLOPs Utilisation, asks what fraction of the machine we are genuinely using. A forward and backward pass costs roughly $6N$ operations per token for a model with $N$ weights:

$$
\text{MFU} = \frac{6 \times N \times \text{tokens per second}}{\text{what the machine can do per second}}
$$

Taking a 9B model producing 12,000 tokens per second on eight H100s:

| Quantity | Value |
| --- | --- |
| what we achieved | 648 TFLOP/s |
| what we are paying for | 7,912 TFLOP/s |
| **MFU** | **8.2%** |

A healthy run sits between **35% and 50%**, and the reason to watch it is uncomfortable: at 8% the loss curve looks exactly as it does at 45%. The model learns perfectly well, we simply pay four times as much and wait four times as long, and nothing in the loss will ever mention it.

> **Carry this forward:** the loss tells you whether it is learning. MFU tells you whether you are being robbed.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_11_mfu.html" width="1280" height="775" title="s10_widget_11_mfu"></iframe>

> **What the machine is actually doing.** The same run at 8% and at 45%, with the days it would take shown beside each. The loss curve is identical in both.

## 15\. The failures that stay quiet

V4 recorded eight failures during its run, and what they share matters more than any one of them.

| Failure | What happened |
| --- | --- |
| Silent checkpoint misload | the checkpoint did not match the model, and loaded without complaint |
| Flush divergence at scale | a technique that worked below 1B diverged at 120B |
| Clone-family collapse | copied experts died before they learned to differ |
| Role-blind expansion | auxiliary heads inflated when expanded like experts |
| Curriculum accounting | the data guarantee was never actually measured |
| Spectral upcycling | variety was created, function was not preserved |
| Mixture-shift instability | routing destabilised as the expert count changed |

The write-up puts it plainly: several of these **pass every cheap check** while staying silent. Not one raised an exception, and several produced loss curves that looked normal.

Two details from the run are worth carrying. **Dropout was set to zero** in every reversible model, because a reversible backward pass needs a repeatable forward pass and one dropout mask would have broken it silently. And the **routers trained more slowly** than the rest of the model, deliberately.

> **Carry this forward:** the dangerous failures do not crash. They produce a plausible number and let you keep going.

## 16\. What V5 has to decide

Three things this session settles for us.

Gradients accumulate, and the loss is normalised **by token rather than by micro-batch**. Clipping is on from step one, and the grad norm is logged from step one. Every memory decision in the next three sessions begins from the **16 bytes per weight** above.

Four things remain open.

| Question | What would settle it |
| --- | --- |
| **What precision do we train in?** bf16 is safe, fp8 is proven, NVFP4 is faster and needs Blackwell. | A short run in each on our real architecture, comparing loss and throughput. |
| **What clip threshold?** | The grad norm distribution over the first thousand steps. Choose it from data rather than from habit. |
| **What MFU will we accept** before stopping to fix the loop? | A number agreed before the run begins, so nobody is negotiating it at three in the morning. |
| **Activation checkpointing everywhere, or only in some layers?** | A memory against throughput sweep on the cluster we actually rent. |

## 17\. The assignment

Take a small model and a real loop, and make it tell you the truth about itself.

**Print every tensor shape** in the step, and write one line saying what each dimension means.

**Verify one gradient by hand.** Nudge a weight, measure how the loss changed, and compare against what `backward()` reported. They should agree to several decimals, and if they do not, you have found something worth understanding.

**Break gradient accumulation on purpose.** Use the average of the averages with micro-batches of different lengths, and plot both curves together so you see the gap rather than take my word for it.

**Log the grad norm at every step**, then find one step where it moved before the loss did.

**Compute your own MFU**, report it honestly, and say what you believe is costing you the distance to 40%.

**Take the number 0.1** and write out by hand what it looks like in fp32, bf16 and fp8 E4M3, showing the bits. Then say which one you would train in, and why.

> Print things and check things. **Every serious training bug is silent**, and the loss curve is not going to be the one that tells you.

[Transcript](https://docs.google.com/document/d/1obvxJFP-pQT5A8DuCgq-tcviA_ZZupTHhwPvvdzrNVo/edit?usp=sharing)

## Video

**Studio**

![](https://www.youtube.com/watch?v=Gg60by9Es_I)

**GMeet**

![](https://www.youtube.com/watch?v=xgwRkE4uIOo)

## Session 10: The Training Loop

## 1\. What this session is

Session 9 left us holding a single number. The model reads a batch, produces logits, compares them against the truth, and hands back one scalar called the loss.

<svg id="mermaid-31" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 620.3125px;" viewBox="0 0 620.3125 91" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-31_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-31_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-31_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-31_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-31_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-31_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-31_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-31_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-31_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-31_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-31_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-31_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 115.2 45.5 L 126.7 45.5 Q 138.2 45.5 138.2 45.5 L 138.2 45.5 Q 138.2 45.5 149.7 45.5 L 161.2 45.5" id="mermaid-31-L_A_B_0" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTE1LjE4NzUsInkiOjQ1LjV9LHsieCI6MTQwLjE4NzUsInkiOjQ1LjV9LHsieCI6MTY1LjE4NzUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-31_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 268.9 45.5 L 280.4 45.5 Q 291.9 45.5 291.9 45.5 L 291.9 45.5 Q 291.9 45.5 303.4 45.5 L 314.9 45.5" id="mermaid-31-L_B_C_0" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MjY4Ljg5MDYyNSwieSI6NDUuNX0seyJ4IjoyOTMuODkwNjI1LCJ5Ijo0NS41fSx7IngiOjMxOC44OTA2MjUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-31_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 417.0 45.5 L 428.5 45.5 Q 440.0 45.5 440.0 45.5 L 440.0 45.5 Q 440.0 45.5 451.5 45.5 L 463.0 45.5" id="mermaid-31-L_C_D_0" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NDE3LjAzMTI1LCJ5Ijo0NS41fSx7IngiOjQ0Mi4wMzEyNSwieSI6NDUuNX0seyJ4Ijo0NjcuMDMxMjUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-31_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_A_B_0" transform="translate(0, 0)"></g></g><g><g data-id="L_B_C_0" transform="translate(0, 0)"></g></g><g><g data-id="L_C_D_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-31-flowchart-A-0" data-look="classic" transform="translate(61.59375, 45.5)"><rect style="" x="-53.59375" y="-26.25" width="107.1875" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-23.59375, -11.25)"><rect></rect><foreignObject width="47.1875" height="22.5"><p>tokens</p></foreignObject></g></g><g id="mermaid-31-flowchart-B-1" data-look="classic" transform="translate(217.0390625, 45.5)"><rect style="" x="-51.8515625" y="-26.25" width="103.703125" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-21.8515625, -11.25)"><rect></rect><foreignObject width="43.703125" height="22.5"><p>model</p></foreignObject></g></g><g id="mermaid-31-flowchart-C-2" data-look="classic" transform="translate(367.9609375, 45.5)"><rect style="" x="-49.0703125" y="-26.25" width="98.140625" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-19.0703125, -11.25)"><rect></rect><foreignObject width="38.140625" height="22.5"><p>logits</p></foreignObject></g></g><g id="mermaid-31-flowchart-D-3" data-look="classic" transform="translate(539.671875, 45.5)"><rect style="" x="-72.640625" y="-37.5" width="145.28125" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-42.640625, -22.5)"><rect></rect><foreignObject width="85.28125" height="45"><p>loss<br>one number</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-31-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

One number, for a model with **536.9M weights in its output head alone** and several billion more behind it. Which raises the question we spend today on:

> **How does a single number reach back and move billions of weights, and while it runs for weeks, how do we know it is working?**

Two ideas have been quietly doing work in this course without ever being properly introduced, and today they finally get the introduction they deserve.

The first is **backpropagation**, the method by which that one loss value travels back through every layer and tells each weight how much it contributed to the mistake.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_1_one_neuron_1.html" width="100%" height="1100" title="widget_1_one_neuron_1"></iframe>

The second is the **floating point number**, which we have written as `bf16` since Session 7 as though its meaning were self-evident. It is not, and the newer formats only become readable once the original is clear.

<svg id="mermaid-32" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 769.171875px;" viewBox="0 0 769.171875 113.5" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-32_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-32_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-32_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-32_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-32_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-32_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-32_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-32_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-32_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-32_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-32_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-32_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 189.8 56.8 L 201.3 56.8 Q 212.8 56.8 212.8 56.8 L 212.8 56.8 Q 212.8 56.8 224.3 56.8 L 235.8 56.8" id="mermaid-32-L_S9_S10_0" style=";" data-edge="true" data-et="edge" data-id="L_S9_S10_0" data-points="W3sieCI6MTg5Ljc2NTYyNSwieSI6NTYuNzV9LHsieCI6MjE0Ljc2NTYyNSwieSI6NTYuNzV9LHsieCI6MjM5Ljc2NTYyNSwieSI6NTYuNzV9XQ==" data-look="classic" marker-end="url(#mermaid-32_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 451.2 56.8 L 462.7 56.8 Q 474.2 56.8 474.2 56.8 L 474.2 56.8 Q 474.2 56.8 485.7 56.8 L 497.2 56.8" id="mermaid-32-L_S10_S11_0" style=";" data-edge="true" data-et="edge" data-id="L_S10_S11_0" data-points="W3sieCI6NDUxLjE3MTg3NSwieSI6NTYuNzV9LHsieCI6NDc2LjE3MTg3NSwieSI6NTYuNzV9LHsieCI6NTAxLjE3MTg3NSwieSI6NTYuNzV9XQ==" data-look="classic" marker-end="url(#mermaid-32_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_S9_S10_0" transform="translate(0, 0)"></g></g><g><g data-id="L_S10_S11_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-32-flowchart-S9-0" data-look="classic" transform="translate(98.8828125, 56.75)"><rect style="" x="-90.8828125" y="-37.5" width="181.765625" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-60.8828125, -22.5)"><rect></rect><foreignObject width="121.765625" height="45"><p>Session 9<br>prediction to loss</p></foreignObject></g></g><g id="mermaid-32-flowchart-S10-1" data-look="classic" transform="translate(345.46875, 56.75)"><rect style="" x="-105.703125" y="-37.5" width="211.40625" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-75.703125, -22.5)"><rect></rect><foreignObject width="151.40625" height="45"><p>Session 10<br>loss to weight update</p></foreignObject></g></g><g id="mermaid-32-flowchart-S11-2" data-look="classic" transform="translate(631.171875, 56.75)"><rect style="" x="-130" y="-48.75" width="260" height="97.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-100, -33.75)"><rect></rect><foreignObject width="200" height="67.5"><p>Session 11<br>which optimiser, which schedule</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-32-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

Every section today answers one of two questions.

> **Two questions:** What does one training step actually do? And what does it cost V5 to take that step, over and over, for weeks?

## 2\. The words we will need

Before we follow a loss backwards, it is worth agreeing on the words for the pieces involved, because they are used loosely almost everywhere else and the confusion is avoidable.

A **gradient** is one number attached to one weight, saying how much the loss would change if that single weight moved.

A **step** is one complete pass of the loop: read some data, compute the loss, compute every gradient, then move every weight.

A **batch** is the group of samples read in a single step.

A **micro-batch** is what a batch gets broken into when the whole thing will not fit in memory at once, which for a model of our size is nearly always.

> **Carry this forward:** a gradient belongs to one weight, and a step moves all of them at once.

## 3\. Where a gradient comes from

We can begin without any calculus at all.

Imagine a model with a single weight sitting at 3.0, and a loss measuring how wrong its answer currently is. Raise that weight a very small amount and compute the loss again, and the size of the shift turns out to be the most useful number in all of training.

| Weight | Loss |
| --- | --- |
| 3.000 | 16.000 |
| 3.001 | 16.064 |

The loss rose by $0.064$ when the weight rose by $0.001$, so:

$$
\frac{0.064}{0.001} = 64
$$

The loss climbs about **64 units for every unit we raise this weight**, which also tells us that to bring it down we should move the weight the other way. A large value means this weight matters a great deal right now, and a value near zero means the optimiser can leave it alone.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_4_training_loop_1.html" width="100%" height="1240" title="widget_4_training_loop_1"></iframe>

> A training loop visualised one step at a time. A small network attempts to predict a value, the loss is displayed as a coloured bar, and as you click "step" the loss decreases while the weights inside the network change colour to show which of them moved and in which direction. Goal: you should leave with a working sense of the cycle that runs through every training step, which is a forward pass that produces a prediction, a comparison against the correct answer, and a backward pass that updates every weight in the network.

> **Carry this forward:** a gradient is the answer to a nudge. Which way, and how urgently.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_3_nudge.html" width="1280" height="726" title="s10_widget_3_nudge"></iframe>

> **The nudge, done live.** Move the weight, then change how big the nudge is, and watch the slope settle on the same answer from either side.

## 4\. Following the loss backwards

A real model is a long chain of steps, and a weight near the front never touches the loss directly. It has to work through everything downstream of it.

<svg id="mermaid-33" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 646.703125px;" viewBox="0 0 646.703125 68.5" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-33_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-33_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-33_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-33_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-33_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-33_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-33_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-33_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-33_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-33_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-33_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-33_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 76.2 34.3 L 87.7 34.3 Q 99.2 34.3 99.2 34.3 L 99.2 34.3 Q 99.2 34.3 110.7 34.3 L 122.2 34.3" id="mermaid-33-L_X_H_0" style=";" data-edge="true" data-et="edge" data-id="L_X_H_0" data-points="W3sieCI6NzYuMjAzMTI1LCJ5IjozNC4yNX0seyJ4IjoxMDEuMjAzMTI1LCJ5IjozNC4yNX0seyJ4IjoxMjYuMjAzMTI1LCJ5IjozNC4yNX1d" data-look="classic" marker-end="url(#mermaid-33_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 252.8 34.3 L 264.3 34.3 Q 275.8 34.3 275.8 34.3 L 275.8 34.3 Q 275.8 34.3 287.3 34.3 L 298.8 34.3" id="mermaid-33-L_H_Y_0" style=";" data-edge="true" data-et="edge" data-id="L_H_Y_0" data-points="W3sieCI6MjUyLjc2NTYyNSwieSI6MzQuMjV9LHsieCI6Mjc3Ljc2NTYyNSwieSI6MzQuMjV9LHsieCI6MzAyLjc2NTYyNSwieSI6MzQuMjV9XQ==" data-look="classic" marker-end="url(#mermaid-33_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 432.6 34.3 L 444.1 34.3 Q 455.6 34.3 455.6 34.3 L 455.6 34.3 Q 455.6 34.3 467.1 34.3 L 478.6 34.3" id="mermaid-33-L_Y_L_0" style=";" data-edge="true" data-et="edge" data-id="L_Y_L_0" data-points="W3sieCI6NDMyLjYwOTM3NSwieSI6MzQuMjV9LHsieCI6NDU3LjYwOTM3NSwieSI6MzQuMjV9LHsieCI6NDgyLjYwOTM3NSwieSI6MzQuMjV9XQ==" data-look="classic" marker-end="url(#mermaid-33_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_X_H_0" transform="translate(0, 0)"></g></g><g><g data-id="L_H_Y_0" transform="translate(0, 0)"></g></g><g><g data-id="L_Y_L_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-33-flowchart-X-0" data-look="classic" transform="translate(42.1015625, 34.25)"><rect style="" x="-34.1015625" y="-26.25" width="68.203125" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-4.1015625, -11.25)"><rect></rect><foreignObject width="8.203125" height="22.5"><p>x</p></foreignObject></g></g><g id="mermaid-33-flowchart-H-1" data-look="classic" transform="translate(189.484375, 34.25)"><rect style="" x="-63.28125" y="-26.25" width="126.5625" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-33.28125, -11.25)"><rect></rect><foreignObject width="66.5625" height="22.5"><p>h = w1 · x</p></foreignObject></g></g><g id="mermaid-33-flowchart-Y-2" data-look="classic" transform="translate(367.6875, 34.25)"><rect style="" x="-64.921875" y="-26.25" width="129.84375" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-34.921875, -11.25)"><rect></rect><foreignObject width="69.84375" height="22.5"><p>y = w2 · h</p></foreignObject></g></g><g id="mermaid-33-flowchart-L-3" data-look="classic" transform="translate(560.65625, 34.25)"><rect style="" x="-78.046875" y="-26.25" width="156.09375" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-48.046875, -11.25)"><rect></rect><foreignObject width="96.09375" height="22.5"><p>loss = (y − t)²</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-33-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

Here $t$ is the **target**, meaning the answer we wanted. If we feed in 2 and hope for 20, then $t$ is 20 and everything the model does is measured against it.

Putting real numbers through:

| Quantity | Calculation | Value |
| --- | --- | --- |
| $h$ | $w_1 \times x = 3 \times 2$ | 6 |
| $y$ | $w_2 \times h = 4 \times 6$ | 24 |
| loss | $(24 - 20)^2$ | 16 |

If we nudge $w_1$, how much does the loss move? Since it reaches the loss only through $h$ and then $y$, we walk back one link at a time, multiplying as we go.

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial y}  &= 2(y - t) = 8 \\
\frac{\partial \mathcal{L}}{\partial w_2} &= 8 \times h = 48 \\
\frac{\partial \mathcal{L}}{\partial h}  &= 8 \times w_2 = 32 \\
\frac{\partial \mathcal{L}}{\partial w_1} &= 32 \times x = 64
\end{aligned}
$$

That is the whole of backpropagation, and it sounds grander than it is: **one number comes back from the loss, and each link multiplies it by one local factor.** Nudging $w_1$ and measuring, as we did above, gives **64.0000**, agreeing to eight decimals.

> **Carry this forward:** backpropagation is the chain rule, applied one link at a time, from the loss towards the inputs.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_4_chain.html" width="1280" height="638" title="s10_widget_4_chain"></iframe>

> **One link at a time.** Step backwards through the chain and watch each multiplication as it happens, ending on the 64 we found by nudging.

## 5\. What autograd is doing for us

Four links by hand is pleasant enough, but several hundred layers is not, so the computer keeps notes for us. Every time it multiplies or adds during the forward pass it records what it did and what went in, and that record is the **computation graph**.

Walking it backwards while multiplying by each local factor is precisely what `loss.backward()` does. The machinery keeping those notes is **autograd**, and it is not doing anything cleverer than the section above, simply doing it a few billion times without losing its place.

> **Carry this forward:** autograd is bookkeeping, not magic. The mathematics is the chain rule you just did by hand.

## 6\. One step, start to finish

Everything above assembles into five lines, and every training run in the world is these five lines repeated until the data runs out.

```
logits = model(batch)          # forward
loss   = loss_fn(logits, y)    # one number
loss.backward()                # fill in every gradient
optimizer.step()               # move every weight
optimizer.zero_grad()          # wipe before the next batch
```

That last line surprises people. **Gradients add up rather than replace**, so each call to `backward()` piles the new gradient onto whatever was already there, and forgetting to wipe means batch two trains on top of batch one.

The loss will still fall, the model will simply be learning from a stale blend of everything it has seen, and nothing will raise an error. That looks like a defect until the next section, which is why it was built this way on purpose.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_1_step.html" width="1280" height="698" title="s10_widget_1_step"></iframe>

> **The whole step on one screen.** Press Step and watch the loss return as gradients, the gradients become an update, and the weights move. Change the target and the gradients change sign. The two bars beneath each weight are what the optimiser keeps between steps, and Session 11 explains why.

> **Carry this forward:** a step is forward, loss, backward, update, wipe. The wipe is not optional.

## 7\. Borrowing a bigger batch

There is a tension at the centre of every run. We want a large batch, because averaging over more samples gives a steadier gradient, and the accelerator wants a small one, because a large batch will not fit.

**Gradient accumulation** settles it using the behaviour that looked like a bug a moment ago. Run several small batches in turn, let their gradients pile up, and take one step once they have all contributed.

<svg id="mermaid-34" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 772.96875px;" viewBox="0 0 772.96875 91" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-34_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-34_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-34_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-34_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-34_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-34_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-34_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-34_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-34_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-34_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-34_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-34_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 164.9 45.5 L 176.4 45.5 Q 187.9 45.5 187.9 45.5 L 187.9 45.5 Q 187.9 45.5 199.4 45.5 L 210.9 45.5" id="mermaid-34-L_M1_M2_0" style=";" data-edge="true" data-et="edge" data-id="L_M1_M2_0" data-points="W3sieCI6MTY0LjkyMTg3NSwieSI6NDUuNX0seyJ4IjoxODkuOTIxODc1LCJ5Ijo0NS41fSx7IngiOjIxNC45MjE4NzUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-34_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 374.9 45.5 L 386.4 45.5 Q 397.9 45.5 397.9 45.5 L 397.9 45.5 Q 397.9 45.5 409.4 45.5 L 420.9 45.5" id="mermaid-34-L_M2_M3_0" style=";" data-edge="true" data-et="edge" data-id="L_M2_M3_0" data-points="W3sieCI6Mzc0Ljg5MDYyNSwieSI6NDUuNX0seyJ4IjozOTkuODkwNjI1LCJ5Ijo0NS41fSx7IngiOjQyNC44OTA2MjUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-34_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 585.0 45.5 L 596.5 45.5 Q 608.0 45.5 608.0 45.5 L 608.0 45.5 Q 608.0 45.5 619.5 45.5 L 631.0 45.5" id="mermaid-34-L_M3_S_0" style=";" data-edge="true" data-et="edge" data-id="L_M3_S_0" data-points="W3sieCI6NTg0Ljk2ODc1LCJ5Ijo0NS41fSx7IngiOjYwOS45Njg3NSwieSI6NDUuNX0seyJ4Ijo2MzQuOTY4NzUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-34_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_M1_M2_0" transform="translate(0, 0)"></g></g><g><g data-id="L_M2_M3_0" transform="translate(0, 0)"></g></g><g><g data-id="L_M3_S_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-34-flowchart-M1-0" data-look="classic" transform="translate(86.4609375, 45.5)"><rect style="" x="-78.4609375" y="-26.25" width="156.921875" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-48.4609375, -11.25)"><rect></rect><foreignObject width="96.921875" height="22.5"><p>micro-batch 1</p></foreignObject></g></g><g id="mermaid-34-flowchart-M2-1" data-look="classic" transform="translate(294.90625, 45.5)"><rect style="" x="-79.984375" y="-26.25" width="159.96875" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-49.984375, -11.25)"><rect></rect><foreignObject width="99.96875" height="22.5"><p>micro-batch 2</p></foreignObject></g></g><g id="mermaid-34-flowchart-M3-2" data-look="classic" transform="translate(504.9296875, 45.5)"><rect style="" x="-80.0390625" y="-26.25" width="160.078125" height="52.5" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-50.0390625, -11.25)"><rect></rect><foreignObject width="100.078125" height="22.5"><p>micro-batch 3</p></foreignObject></g></g><g id="mermaid-34-flowchart-S-3" data-look="classic" transform="translate(699.96875, 45.5)"><rect style="" x="-65" y="-37.5" width="130" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-35, -22.5)"><rect></rect><foreignObject width="70" height="45"><p>step once<br>then wipe</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-34-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

| Quantity | Meaning | Example |
| --- | --- | --- |
| micro-batch | what actually fits | 8 sequences |
| accumulation steps | how many we gather | 4 |
| **global batch** | what the optimiser sees | **32 sequences** |

For scale, V4 trained with a micro-batch of **7 per GPU and a global batch of 56** across one eight-GPU node, which is a good deal smaller than most people imagine.

> **Carry this forward:** the batch the optimiser sees is a choice, and it is independent of what fits in memory.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_5_accumulate.html" width="1280" height="645" title="s10_widget_5_accumulate"></iframe>

> **Gradients piling up.** Add micro-batches one at a time and watch the buffer fill, then compare the result against the same examples run as a single large batch.

## 8\. A mistake worth studying

A bug lived inside every major training framework until 2024, and it is worth walking through because the lesson generalises far past this one case.

Take three micro-batches holding different numbers of real tokens, as sequences naturally differ in length.

| Micro-batch | Valid tokens | Average loss |
| --- | --- | --- |
| 1 | 4 | 2.0 |
| 2 | 4 | 2.0 |
| 3 | 2 | 5.0 |

The correct combination adds up all the loss and divides by all the tokens, so every token carries equal weight:

$$
\frac{4(2.0) + 4(2.0) + 2(5.0)}{4 + 4 + 2} = \frac{26.0}{10} = 2.6000
$$

What the frameworks did instead was average the three averages:

$$
\frac{2.0 + 2.0 + 5.0}{3} = 3.0000
$$

The short micro-batch, holding half as many tokens, was given exactly the same vote as the long ones, and the result comes out **15.4% wrong**.

What makes it worth a section is how it hid. The error vanishes whenever micro-batches happen to hold equal token counts, which in casual testing they usually do, so the loss curves looked reasonable while being wrong.

> **Carry this forward:** a number that looks plausible is not evidence. This is the theme of the rest of the session.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_6_wrong_average.html" width="1280" height="646" title="s10_widget_6_wrong_average"></iframe>

> **The bug, and how it hid.** Both ways of combining the losses, side by side, at 15.4% apart. Now make the token counts equal and watch the error drop to zero.

## 9\. How a computer holds a number

A computer has a fixed number of bits and must store values as large as 50,000 and as small as 0.00000001 in the same box. Plain counting cannot stretch that far, so the bits are divided between three jobs.

| Piece | Its job | Bits |
| --- | --- | --- |
| **sign** | positive or negative | always 1 |
| **exponent** | how big the number is | a few |
| **mantissa** | the digits themselves | whatever remains |

It is the arrangement we use writing $6.02 \times 10^{23}$, where the power of ten says how large, and the 6.02 says which.

$$
\text{fp32} = 1 \text{ sign} + 8 \text{ exponent} + 23 \text{ mantissa} = 32 \text{ bits}
$$

Every format we will meet is simply a different way of dividing the bits between those last two jobs, and the trade is always the same one.

> **Carry this forward: exponent bits buy range, mantissa bits buy detail.** More of one always means less of the other.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_2_floats.html" width="1280" height="749" title="s10_widget_2_floats"></iframe>

> **Every step of the construction, shown.** The exponent bits added from their place values, the bias taken off, the mantissa read as a binary fraction, then the three pieces multiplied together. Click any bit to flip it, or build your own format with the two sliders.

## 10\. Why bf16 replaced fp16

Three formats matter for training, and one column of this table explains why the industry settled where it did.

| Format | Bits | Exponent | Mantissa | Smallest it holds | Decimal digits |
| --- | --- | --- | --- | --- | --- |
| fp32 | 32 | **8** | 23 | $1.18 \times 10^{-38}$ | 7.2 |
| fp16 | 16 | **5** | 10 | $6.1 \times 10^{-5}$ | 3.3 |
| bf16 | 16 | **8** | 7 | $1.18 \times 10^{-38}$ | 2.4 |

Look at the exponent column. **bf16 keeps all eight exponent bits that fp32 had**, so it reaches just as far, paying for that reach with detail, from 7.2 decimal digits down to 2.4.

fp16 made the opposite bargain, and that is what breaks it. Late in a run the gradients become genuinely tiny, and fp16 cannot hold anything below about $5.96 \times 10^{-8}$.

| Gradient | In fp16 | In bf16 |
| --- | --- | --- |
| $10^{-4}$ | fine | fine |
| $10^{-6}$ | losing digits | fine |
| $10^{-8}$ | **becomes exactly zero** | fine |

A gradient of zero means that weight does not move, so the model quietly stops learning exactly where the signal was faintest, which is usually where something was left to learn.

The rescue is **loss scaling**, as blunt as it sounds. Multiply the loss by about 1024 before the backward pass so the gradients clear the floor, then divide back down before the optimiser sees them.

$$
10^{-8} \times 1024 = 1.02 \times 10^{-5} \quad\text{survives, then divide back}
$$

It works, and it is one more setting to tune and eventually get wrong. bf16's floor sits at $9.18 \times 10^{-41}$, which nothing in training will ever reach, so the whole apparatus becomes unnecessary.

> **Carry this forward:** bf16 is less accurate than fp16 and won anyway, because range mattered more than digits.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_7_underflow.html" width="1280" height="693" title="s10_widget_7_underflow"></iframe>

> **A gradient shrinking.** Slide it down and watch fp16 give up while bf16 keeps going. Then switch loss scaling on and watch the same gradient survive.

## 11\. The newer formats

Once the trade is clear, the 2026 formats read easily.

| Format | Bits | Exponent | Mantissa | Decimal digits |
| --- | --- | --- | --- | --- |
| fp8 E4M3 | 8 | 4 | 3 | 1.2 |
| fp8 E5M2 | 8 | 5 | 2 | 0.9 |
| fp4 E2M1 | 4 | 2 | 1 | 0.6 |

Six tenths of a decimal digit ought to worry you, and on its own it should. **fp4 works only because its numbers are never alone.** Sixteen of them share a single exponent, stored once for the whole block.

$$
\frac{16 \times 4 \text{ bits} + 8 \text{ bits of shared scale}}{16} = 4.5 \text{ bits per value}
$$

This is the same idea as `int8`, which the widget shows in its last row. An `int8` tensor has no exponent anywhere in it, only whole numbers and one scale shared by everything.

As of 2026, **fp8 is a production recipe** using `E4M3` throughout, and **NVFP4 runs about 1.73 times faster than fp8**, though it needs five things working together to converge, including leaving attention in higher precision because softmax amplifies whatever noise you hand it.

> **Carry this forward:** we do not shrink everything equally. We shrink where the error does not accumulate.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_8_block_scale.html" width="1280" height="764" title="s10_widget_8_block_scale"></iframe>

> **Sixteen numbers sharing one exponent.** Change the block size and watch the cost per value fall, then put an outlier in and watch what sharing costs you.

## 12\. Guarding against one bad batch

A single unusual batch can end a run that has gone well for weeks. The gradients come back enormous, the weights take a stride far outside anything sensible, and the model lands somewhere it will not recover from.

The guard is a cap. We measure the combined size of all the gradients, and when it exceeds a threshold we shrink every one by the same factor, leaving the direction untouched and changing only the length.

$$
\text{norm} = 8.4, \quad \text{cap} = 1.0, \quad \text{so scale everything by } \frac{1.0}{8.4} = 0.119
$$

That combined figure is called the **grad norm**, and it turns out to be worth rather more than the clipping it enables.

> **Carry this forward:** the grad norm moves before the loss does. A run about to fail often shows it there thousands of steps early, which makes it the most useful trace on any dashboard.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_9_clip.html" width="1280" height="608" title="s10_widget_9_clip"></iframe>

> **The norm and the loss, on one axis.** The spike arrives in the norm at step 24 and only reaches the loss at step 26. Set the cap and watch it hold.

## 13\. What a step costs us

Running a finished model requires only its weights. Training one requires far more, and this arithmetic sets up the next three sessions.

| What must be held | Bytes per weight |
| --- | --- |
| the weight itself, in bf16 | 2 |
| its gradient, in bf16 | 2 |
| a full-precision copy, in fp32 | 4 |
| the optimiser's two running numbers | 8 |
| **total** | **16** |

Sixteen bytes for every single weight, and that is before a single activation is stored.

| Model | Training state alone |
| --- | --- |
| 2B | 29.8 GiB |
| 9B | 134.1 GiB |
| 20B | 298.0 GiB |
| 120B | 1,788 GiB |

An 80 GB accelerator therefore holds about a **5.4B** model and has nothing left over for the activations it also needs.

**Activation checkpointing** buys some back by discarding intermediates during the forward pass and recomputing them during the backward pass, costing about 30% more compute for a large share of the memory.

> **Carry this forward:** 16 bytes per weight is why Sessions 12 and 13 exist. One card is not enough, and this table is the reason.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_10_memory.html" width="1280" height="546" title="s10_widget_10_memory"></iframe>

> **Five things held for every weight.** Slide the model size up and find the point where one 80 GB card runs out, which happens sooner than most people expect.

## 14\. Watching a run that lasts weeks

Four traces, each catching something the others cannot.

| Trace | What it catches |
| --- | --- |
| loss | whether the model is learning at all |
| **grad norm** | trouble, before the loss shows it |
| tokens per second | whether something quietly got slower |
| **MFU** | whether the machine is being wasted |

The last needs explaining, because it is the one most often skipped.

**MFU**, or Model FLOPs Utilisation, asks what fraction of the machine we are genuinely using. A forward and backward pass costs roughly $6N$ operations per token for a model with $N$ weights:

$$
\text{MFU} = \frac{6 \times N \times \text{tokens per second}}{\text{what the machine can do per second}}
$$

Taking a 9B model producing 12,000 tokens per second on eight H100s:

| Quantity | Value |
| --- | --- |
| what we achieved | 648 TFLOP/s |
| what we are paying for | 7,912 TFLOP/s |
| **MFU** | **8.2%** |

A healthy run sits between **35% and 50%**, and the reason to watch it is uncomfortable: at 8% the loss curve looks exactly as it does at 45%. The model learns perfectly well, we simply pay four times as much and wait four times as long, and nothing in the loss will ever mention it.

> **Carry this forward:** the loss tells you whether it is learning. MFU tells you whether you are being robbed.

<iframe src="https://axiom.theschoolofai.in/widgets/s10_widget_11_mfu.html" width="1280" height="775" title="s10_widget_11_mfu"></iframe>

> **What the machine is actually doing.** The same run at 8% and at 45%, with the days it would take shown beside each. The loss curve is identical in both.

## 15\. The failures that stay quiet

V4 recorded eight failures during its run, and what they share matters more than any one of them.

| Failure | What happened |
| --- | --- |
| Silent checkpoint misload | the checkpoint did not match the model, and loaded without complaint |
| Flush divergence at scale | a technique that worked below 1B diverged at 120B |
| Clone-family collapse | copied experts died before they learned to differ |
| Role-blind expansion | auxiliary heads inflated when expanded like experts |
| Curriculum accounting | the data guarantee was never actually measured |
| Spectral upcycling | variety was created, function was not preserved |
| Mixture-shift instability | routing destabilised as the expert count changed |

The write-up puts it plainly: several of these **pass every cheap check** while staying silent. Not one raised an exception, and several produced loss curves that looked normal.

Two details from the run are worth carrying. **Dropout was set to zero** in every reversible model, because a reversible backward pass needs a repeatable forward pass and one dropout mask would have broken it silently. And the **routers trained more slowly** than the rest of the model, deliberately.

> **Carry this forward:** the dangerous failures do not crash. They produce a plausible number and let you keep going.

## 16\. What V5 has to decide

Three things this session settles for us.

Gradients accumulate, and the loss is normalised **by token rather than by micro-batch**. Clipping is on from step one, and the grad norm is logged from step one. Every memory decision in the next three sessions begins from the **16 bytes per weight** above.

Four things remain open.

| Question | What would settle it |
| --- | --- |
| **What precision do we train in?** bf16 is safe, fp8 is proven, NVFP4 is faster and needs Blackwell. | A short run in each on our real architecture, comparing loss and throughput. |
| **What clip threshold?** | The grad norm distribution over the first thousand steps. Choose it from data rather than from habit. |
| **What MFU will we accept** before stopping to fix the loop? | A number agreed before the run begins, so nobody is negotiating it at three in the morning. |
| **Activation checkpointing everywhere, or only in some layers?** | A memory against throughput sweep on the cluster we actually rent. |

## 17\. The assignment

Take a small model and a real loop, and make it tell you the truth about itself.

**Print every tensor shape** in the step, and write one line saying what each dimension means.

**Verify one gradient by hand.** Nudge a weight, measure how the loss changed, and compare against what `backward()` reported. They should agree to several decimals, and if they do not, you have found something worth understanding.

**Break gradient accumulation on purpose.** Use the average of the averages with micro-batches of different lengths, and plot both curves together so you see the gap rather than take my word for it.

**Log the grad norm at every step**, then find one step where it moved before the loss did.

**Compute your own MFU**, report it honestly, and say what you believe is costing you the distance to 40%.

**Take the number 0.1** and write out by hand what it looks like in fp32, bf16 and fp8 E4M3, showing the bits. Then say which one you would train in, and why.

> Print things and check things. **Every serious training bug is silent**, and the loss curve is not going to be the one that tells you.

[Transcript](https://docs.google.com/document/d/1obvxJFP-pQT5A8DuCgq-tcviA_ZZupTHhwPvvdzrNVo/edit?usp=sharing)

## Video

**Studio**

![](https://www.youtube.com/watch?v=Gg60by9Es_I)

**GMeet**

![](https://www.youtube.com/watch?v=xgwRkE4uIOo)
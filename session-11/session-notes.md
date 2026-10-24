# Session 11 — Notes

> Working notes: understanding of the task, design decisions, open questions, progress.

## Understanding the task

Five tasks, each a claim about Adam or its LR schedule that's easy to state and easy to
get subtly wrong. The binding instruction is the warning: **tune both sides before
accepting a comparison** — the reason most "optimizer X beats Y" claims don't replicate.
So each task has to be a *fair* measurement, not a flattering one.

1. Reproduce Adam by hand (m, v, m̂, v̂, step) for one weight / five gradients, check vs PyTorch.
2. Disable bias correction; plot 20 steps both ways; say when the difference stops mattering.
3. Log `‖update‖/‖weight‖` per layer; find the step where warmup stops changing it.
4. Train the same model cosine vs WSD for 300 steps, stop both at 200; report both losses + which to keep.
5. Sweep LR at widths 256/512/1024; plot loss-vs-LR; mark 3 minima; state the 4096 value + confidence.

Deliverable: a detailed README + support code (1000 pts).

## Approach / design

Same pattern as sessions 9–10: one runnable `.py` as source of truth → `build_notebook.py`
splits it → executed `.ipynb` with outputs/plots → README with the numbers.

- Tasks 1–2 use a **single scalar weight** (independent of any model) so the Adam math is naked.
- Tasks 3–5 use a small **width-parameterised MLP**. Task choice matters a lot:
  - a task that converges to ~0 kills the gradients and *degenerates* tasks 3 & 4, so 3 & 4 use
    a **hard nonlinear classification** task (random teacher + 20% label noise → a loss floor);
  - task 4 uses a deliberately **small-capacity (width-64) proxy** so the high-LR phase can't
    memorize the noise and the value of decaying stays visible;
  - task 5 needs a **sharp optimum** to resolve a subtle factor-~2 LR shift, so it uses a
    **smooth regression** surface with a finer (quarter-decade) LR grid.

## Progress log

- Built `adam_lab.py`; all five tasks run clean on CPU (deterministic).
- Task 1 reproduces the lecture's exact table (final w = **0.995031**), agrees with PyTorch to ~1e-8.
- Task 2: 1.00η vs 3.162η at step 1; m-correction gone by ~44, full gap within 1% by ~3925.
- First pass of tasks 3 & 4 washed out (easy regression task converged to ~1e-4) → switched to the
  hard classification task; fixed task-3 detector (ratio *peaks* at warmup end, doesn't plateau).
- Task 5 drift washed out on the noisy classifier + coarse grid → switched to smooth regression +
  quarter-decade grid → clean monotonic minima 1e-2 / 3.16e-3 / 1e-3 (0.32 per doubling).
- Notebook built + executed (14 cells, 4 plots embedded, 0 errors). README + docs written.

## Open questions

- The observed per-doubling LR ratio (0.32) is steeper than the ideal 0.50 on this toy surface —
  plausibly depth/activation-dependent; muP would be the principled next step to verify transfer.
- Task 4's "decay beats flat" ordering only emerges once the model is past the undertrained regime;
  a longer stable phase before the stop would sharpen it (noted honestly in the README).


---
title: "Axiom — Learning OS"
source: "https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms11kq8w4l9r6y3zp7ef/lesson"
author:
published:
created: 2026-10-24
description: "A premium learning operating system for ambitious technical education."
tags:
  - "clippings"
---
[Axiom](https://axiom.theschoolofai.in/dashboard)

[Settings](https://axiom.theschoolofai.in/settings)

## Session 11: Optimizers and Learning-Rate Schedules

## 1\. Introduction

Session 10 defined the training step as five lines of code, one of which was left unspecified. `loss.backward()` computes a gradient for every weight, and `optimizer.step()` then uses those gradients to move the weights, but how was never explained.

This session covers that. It begins from the observation that a gradient is incomplete information.

> **A gradient specifies the direction in which a weight should move. It does not specify the distance.**

The gradient computed in Session 10 was 64. That value does not prescribe a movement of 64, or of 0.64, or of any particular size. It gives a direction and a relative urgency, and the distance is a separate decision. Every method in this session is a different rule for making that decision.

| Method | How it sets the distance | Section |
| --- | --- | --- |
| learning rate | one fixed value for every weight | 3 |
| momentum | from an average of recent gradients | 4 |
| second moment | separately per weight, from its own gradient history | 5 |
| Adam | both of the above combined | 6 |
| weight decay | adds a constant pull towards zero | 7 |
| warmup | small at first, while the estimates are unreliable | 9 |
| schedule | decreasing over the course of the run | 10 |
| batch scaling | from the number of samples averaged | 11 |
| muP | from the width of the model | 12 |
| Muon | from the shape of the weight matrix | 13 |

<svg id="mermaid-37" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 680.53125px;" viewBox="0 0 680.53125 91" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-37_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-37_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-37_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-37_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-37_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-37_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-37_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-37_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-37_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-37_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-37_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-37_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 177.3 45.5 L 188.8 45.5 Q 200.3 45.5 200.3 45.5 L 200.3 45.5 Q 200.3 45.5 211.8 45.5 L 223.3 45.5" id="mermaid-37-L_S10_S11_0" style=";" data-edge="true" data-et="edge" data-id="L_S10_S11_0" data-points="W3sieCI6MTc3LjI2NTYyNSwieSI6NDUuNX0seyJ4IjoyMDIuMjY1NjI1LCJ5Ijo0NS41fSx7IngiOjIyNy4yNjU2MjUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-37_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 428.4 45.5 L 439.9 45.5 Q 451.4 45.5 451.4 45.5 L 451.4 45.5 Q 451.4 45.5 462.9 45.5 L 474.4 45.5" id="mermaid-37-L_S11_S12_0" style=";" data-edge="true" data-et="edge" data-id="L_S11_S12_0" data-points="W3sieCI6NDI4LjM3NSwieSI6NDUuNX0seyJ4Ijo0NTMuMzc1LCJ5Ijo0NS41fSx7IngiOjQ3OC4zNzUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-37_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_S10_S11_0" transform="translate(0, 0)"></g></g><g><g data-id="L_S11_S12_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-37-flowchart-S10-0" data-look="classic" transform="translate(92.6328125, 45.5)"><rect style="" x="-84.6328125" y="-37.5" width="169.265625" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-54.6328125, -22.5)"><rect></rect><foreignObject width="109.265625" height="45"><p>Session 10<br>loss to gradient</p></foreignObject></g></g><g id="mermaid-37-flowchart-S11-1" data-look="classic" transform="translate(327.8203125, 45.5)"><rect style="" x="-100.5546875" y="-37.5" width="201.109375" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-70.5546875, -22.5)"><rect></rect><foreignObject width="141.109375" height="45"><p>Session 11<br>gradient to distance</p></foreignObject></g></g><g id="mermaid-37-flowchart-S12-2" data-look="classic" transform="translate(575.453125, 45.5)"><rect style="" x="-97.078125" y="-37.5" width="194.15625" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-67.078125, -22.5)"><rect></rect><foreignObject width="134.15625" height="45"><p>Session 12<br>across many GPUs</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-37-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

The methods appear in the order in which they were developed, because each one corrects a specific failure of the one before it.

## 2\. Terminology

Five terms recur throughout the session, each given here with a representative value.

The **learning rate**, written $\eta$, is the factor that converts a gradient into a distance. For a model of our size it usually lies near $0.0003$.

A **hyperparameter** is a setting we choose before training rather than learn during it. The learning rate is one, and the batch size from Session 10 is another.

**Optimizer state** is any quantity the optimizer stores about a weight between steps. Plain gradient descent stores none. The two values in the Session 10 widget are optimizer state, and Section 6 identifies them.

An **epoch** is one complete pass over the training data. Our run makes roughly one pass, so steps rather than epochs are the useful unit.

A **schedule** is a rule that varies the learning rate over the course of training, and it is the subject of Section 10.

> **Carry this forward:** the learning rate is the most consequential single number in the run, and the rest of this session exists to constrain how it is set.

## 3\. Gradient Descent

The simplest rule that uses a gradient moves each weight against it by a fixed fraction.

$$
w \leftarrow w - \eta \cdot g
$$

This is **gradient descent**. Take one weight starting at 0, and a loss $\mathcal{L} = (w - 5)^2$ that is smallest when the weight reaches 5. Its gradient is $2(w - 5)$, the 2 coming from differentiating the square.

Call the distance still to travel $d = w - 5$. Substituting that gradient into the update rule and then subtracting 5 from both sides gives the distance after one step:

$$
\begin{aligned}
w_{\text{new}} &= w - \eta \cdot 2(w - 5) \\
w_{\text{new}} - 5 &= (w - 5) - 2\eta(w - 5) \\
d_{\text{new}} &= d \cdot (1 - 2\eta)
\end{aligned}
$$

The $(w - 5)$ factors out and leaves a single number multiplying it, so **every step shrinks or grows the remaining distance by the same factor $(1 - 2\eta)$**, and the whole behaviour of the run is decided by that one number.

That factor carries two separate pieces of information. Its **sign** says which side of the minimum the weight lands on: positive and it stayed on the side it started, negative and it crossed over, which is an overshoot. Its **magnitude** says whether the distance shrank or grew: below 1 and the weight is closer than before, above 1 and it is further away. Convergence therefore needs $|1 - 2\eta| < 1$, which for this loss means $\eta$ between 0 and 1.

| $\eta$ | multiplier | $w$ over five steps |
| --- | --- | --- |
| 0.01 | +0.98 | 0.10, 0.20, 0.29, 0.39, 0.48 |
| 0.10 | +0.80 | 1.00, 1.80, 2.44, 2.95, 3.36 |
| 0.90 | −0.80 | 9.00, 1.80, 7.56, 2.95, 6.64 |
| 1.10 | **−1.20** | 11.0, −2.2, 13.6, −5.4, **17.4** |

Four different behaviours from one number. At $0.01$ the weight crawls and will need several hundred steps. At $0.10$ it converges cleanly. At $0.90$ it overshoots the target every step but the overshoot shrinks, so it still converges. At $1.10$ the overshoot grows instead, and the weight is further from the target after five steps than when it started.

The same three lines work on any loss of this shape. If the gradient is $c \cdot d$ rather than $2d$, the multiplier comes out as $(1 - \eta c)$, where $c$ measures how sharply the loss bends around its minimum. We call $c$ the **curvature**, and it was 2 in the example above. Convergence requires $|1 - \eta c| < 1$, so **the safe range for $\eta$ is set by the curvature.**

A real loss does not have one curvature. It bends sharply along some directions and gently along others, and a single $\eta$ has to serve all of them. Consider

$$
\mathcal{L} = \tfrac{1}{2}\left(20u^2 + v^2\right)
$$

which has curvature 20 along $u$ and curvature 1 along $v$. Starting from $(1, 1)$, each step multiplies $u$ by $(1 - 20\eta)$ and $v$ by $(1 - \eta)$, read exactly as above.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_1b_surface.html" width="1280" height="403" title="s11_widget_1b_surface"></iframe>

| $\eta$ | $u$ multiplier | $v$ multiplier | after five steps |
| --- | --- | --- | --- |
| 0.01 | +0.80 | 0.99 | $u$ is converging, $v$ has barely moved at 0.951 |
| 0.09 | −0.80 | 0.91 | $u$ overshoots every step, $v$ is still slow at 0.624 |
| 0.11 | **−1.20** | 0.89 | $u$ has diverged to −2.49 and is growing |

The value that suits $u$ leaves $v$ almost stationary, and the value that moves $v$ makes $u$ diverge. **This is the failure the next three sections correct.**

> **Carry this forward:** one learning rate shared by every weight is the first assumption to break, because the weights do not share a curvature.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_1_descent.html" width="1280" height="678" title="s11_widget_1_descent"></iframe>

> **Gradient descent in two dimensions.** Raise the learning rate and the steep direction starts to oscillate and then diverges, while the shallow direction is still crawling.

## 4\. Momentum

A weight whose gradient has kept the same sign over many steps can safely be moved further than one whose gradient alternates. Using that fact requires a record of recent gradients whose size does not grow with the number of steps.

An **exponential moving average** keeps a single running value and mixes each new observation into it by a fixed fraction:

$$
m \leftarrow \beta_1 \cdot m + (1 - \beta_1) \cdot g
$$

The fraction $\beta_1$ is a hyperparameter and is almost always 0.9. Each new gradient then contributes one tenth while the existing average keeps nine tenths. A gradient's share is multiplied by $\beta_1$ on every later step, so it has faded to almost nothing after about $1/(1 - \beta_1) = 10$ steps, and that is the sense in which $m$ remembers the last ten gradients while storing one number.

The two directions of Section 3 produce gradients of two characteristic shapes. Along the steep direction the sign flips on every step, because the weight lands past the minimum each time. Along the shallow direction the sign never changes, because the weight approaches the minimum from one side and never reaches it.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_2b_directions.html" width="1280" height="389" title="s11_widget_2b_directions"></iframe>

Take the cleanest version of each shape, one alternating between $+1$ and $-1$ and one holding at a constant $+0.2$, five times smaller.

| step | steep direction | shallow direction |
| --- | --- | --- |
| 1 | 0.100 | 0.020 |
| 2 | −0.010 | 0.038 |
| 3 | 0.091 | 0.054 |
| 4 | −0.018 | 0.069 |
| 5 | **0.084** | **0.082** |

The alternating gradients cancel each other while the constant one accumulates, so after five steps the two averages are nearly equal **despite a fivefold difference in the gradients themselves**. Momentum is the use of $m$ in place of $g$ in the update rule.

> **Carry this forward:** an exponential moving average cancels what alternates and accumulates what is consistent.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_2_momentum.html" width="1280" height="687" title="s11_widget_2_momentum"></iframe>

> **Momentum.** Enter gradients by hand and the average follows them, then apply it to the surface from Section 3 and move $\beta_1$ from 0 to 0.99.

## 5\. Per-Parameter Learning Rates

Momentum still leaves every weight sharing one learning rate, and the weights of a language model differ enormously in how often they receive a gradient at all. Session 7 gives the clearest case. The embedding row for a common word receives a large gradient in almost every batch, while the row for a rare word may receive none across a thousand batches. **One learning rate cannot suit both.**

The weights of a model are also called its **parameters**, and the fix is to give each parameter a learning rate of its own, computed from its own gradient history. We keep a second exponential moving average, this time over the squared gradient, which estimates how large that parameter's gradients typically are:

$$
v \leftarrow \beta_2 \cdot v + (1 - \beta_2) \cdot g^2
$$

The subscript distinguishes it from $\beta_1$ above, and $\beta_2$ is almost always 0.999, so by the same $1/(1 - \beta_2)$ count this average spans about a thousand gradients rather than ten. If a parameter's gradient holds steady at $g$, the average settles on $g^2$, so $\sqrt{v}$ settles on $|g|$, which is where the two values in the table below come from. The step is then divided by $\sqrt{v}$, and writing the update out shows what that does:

$$
w \leftarrow w - \frac{\eta}{\sqrt{v}} \cdot g
$$

The quantity $\eta / \sqrt{v}$ is a learning rate belonging to that parameter alone, and it is what gives this section its name. For two parameters, one receiving a gradient of $1.0$ on every step and one receiving $0.01$, after two hundred steps:

| quantity | Parameter A | Parameter B |
| --- | --- | --- |
| gradient $g$ | 1.0000 | 0.0100 |
| $\sqrt{v}$ | 1.0000 | 0.0100 |
| **its own learning rate, $\eta / \sqrt{v}$** | **1.00 $\eta$** | **100.00 $\eta$** |
| resulting step, $\eta g / \sqrt{v}$ | 1.0000 $\eta$ | 1.0000 $\eta$ |

Parameter B has been given a learning rate a hundred times larger than Parameter A's, and the two then take exactly the same step. **The division removes the magnitude of the gradient and keeps only its sign and its consistency.** A parameter that has been inactive for a thousand steps therefore takes a full-sized step when a gradient finally arrives, which is what a rare word's embedding requires.

> **Carry this forward:** $\eta / \sqrt{v}$ is a separate learning rate for every parameter, and it gives them all the same step size whatever their gradients.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_3_per_parameter.html" width="1280" height="689" title="s11_widget_3_per_parameter"></iframe>

> **Per-parameter learning rates.** Two parameters with gradients a hundred times apart, and the learning rate each one is given, before and after the division.

## 6\. Adam

**Adam combines the two averages.** The first sets the direction of the step, the second sets its scale, and a correction removes a bias they share.

Both averages start at zero, so their early values are pulled towards zero and read too low. Feed a constant gradient $g$ into the first average and after $t$ steps it has reached $m_t = g\,(1 - \beta_1^t)$, short of $g$ by exactly the factor $(1 - \beta_1^t)$. Dividing by that factor restores it, and the same argument applies to $v$. This is **bias correction**, and it is exact rather than approximate.

It matters most at the first step, because $\beta_2 = 0.999$ is much closer to 1 than $\beta_1 = 0.9$ and the two averages are therefore pulled down by very different amounts. At $t = 1$ we have $m_1 = 0.1\,g$ and $v_1 = 0.001\,g^2$, so without the correction

$$
\frac{m_1}{\sqrt{v_1}} = \frac{0.1\,g}{0.0316\,|g|} = 3.16
$$

while with it the two corrections cancel and the ratio is exactly $g / |g| = 1$.

| at step 1, with $g = 0.5$ | step taken |
| --- | --- |
| without bias correction | **3.16 $\eta$** |
| with bias correction | **1.00 $\eta$** |

The complete rule is

$$
w \leftarrow w - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

where $\epsilon$ is a very small constant, usually $10^{-8}$, added only so that a weight whose gradients have all been zero is not divided by zero. Computed by hand for five steps at $\eta = 0.001$:

| $t$ | $g$ | $m$ | $v$ | $\hat{m}$ | $\hat{v}$ | step | $w$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.50 | 0.0500 | 0.000250 | 0.5000 | 0.2500 | −0.001000 | 0.999000 |
| 2 | 0.40 | 0.0850 | 0.000410 | 0.4474 | 0.2050 | −0.000988 | 0.998012 |
| 3 | 0.60 | 0.1365 | 0.000769 | 0.5037 | 0.2567 | −0.000994 | 0.997018 |
| 4 | 0.45 | 0.1678 | 0.000971 | 0.4881 | 0.2431 | −0.000990 | 0.996028 |
| 5 | 0.55 | 0.2061 | 0.001273 | 0.5032 | 0.2550 | −0.000996 | 0.995031 |

The step column follows from the two averages. Once both are corrected, $\hat{m}$ is close to the recent average gradient and $\sqrt{\hat{v}}$ is close to its typical size, so their ratio sits near $\pm 1$ and the step sits near $\pm\eta$. The gradients range from 0.40 to 0.60, while **every step falls within half a percent of $0.001$**, which is the learning rate itself. This is the property that made Adam the default: the gradient sets the direction and the learning rate sets the distance, and the two decisions stop interfering with each other.

$m$ and $v$ are the two values stored in the Session 10 widget.

> **Carry this forward:** Adam moves every weight by approximately $\eta$, whatever the magnitude of its gradient.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_4_adam.html" width="1280" height="654" title="s11_widget_4_adam"></iframe>

> **Adam, one weight, five steps.** Every intermediate value as it is computed, with bias correction switchable so the first step moves between 1.00 and 3.16 times $\eta$.

## 7\. Weight Decay and AdamW

Weights tend to grow over a long run, and large weights make a model brittle. The established remedy is **L2 regularization**, which adds a penalty for magnitude directly to the loss:

$$
\mathcal{L}_{\text{total}} = \mathcal{L} + \tfrac{1}{2}\lambda w^2
$$

The derivative of that penalty is $\lambda w$, which joins the gradient. Under plain gradient descent it shrinks every weight by the same fraction on each step. Under Adam it does not, because that added $\lambda w$ is divided by $\sqrt{\hat{v}}$ along with everything else.

For two weights, both sitting at $0.5$, with $\lambda = 0.1$ and $\eta = 0.001$, the decoupled amount is $\eta\lambda w = 0.001 \times 0.1 \times 0.5 = 5.0 \times 10^{-5}$, and the L2 route is that same amount divided by $\sqrt{\hat{v}}$:

| parameter | $\sqrt{\hat{v}}$ | shrinkage per step |
| --- | --- | --- |
| A | 1.00 | $5.0 \times 10^{-5}$ |
| B | 0.01 | $5.0 \times 10^{-3}$ |

Parameter B is decayed **a hundred times more strongly** than Parameter A, for a reason unconnected to how large it is. How much regularization a weight receives has become a function of its gradient history.

**AdamW** applies the shrinkage separately, after the Adam step, where nothing divides it:

$$
w \leftarrow w - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} - \eta \lambda w
$$

Both weights above then shrink by $5.0 \times 10^{-5}$.

A further consequence was established in 2025. The decoupled term makes the final weights an exponential moving average of the updates applied along the way, with a timescale of $1/(\eta\lambda)$ steps. At $\eta = 0.0003$ and $\lambda = 0.1$ that is $1/(0.0003 \times 0.1) =$ **33,333 steps**, so the model we finish with is an average over roughly the last thirty thousand steps of training. It follows that $\eta$ and $\lambda$ are not two independent settings. **Their product is the setting.**

We exclude normalization scales and biases from decay, because shrinking them changes what the layer computes rather than how large it is.

> **Carry this forward:** decoupled decay shrinks every weight equally, and $\eta\lambda$ determines how far back the finished model averages.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_5_decay.html" width="1280" height="625" title="s11_widget_5_decay"></iframe>

> **L2 against decoupled decay.** The same weight down both routes, identical while its gradients are large and a hundred times apart when they are small.

## 8\. Optimizer Memory Cost

Session 10 counted sixteen bytes for every weight in training. Two of the five entries in that count belong to the optimizer, and Section 6 identified them.

| What is stored | Bytes |
| --- | --- |
| the weight, in bf16 | 2 |
| its gradient, in bf16 | 2 |
| a full-precision copy, in fp32 | 4 |
| **$m$, the gradient average** | **4** |
| **$v$, the squared-gradient average** | **4** |
| total | **16** |

**Half the training memory is optimizer state**, so the choice of rule changes the bill directly. A 9B model at 16 bytes a weight needs $9 \times 10^9 \times 16 = 144$ billion bytes, and dividing by $2^{30}$ turns that into gibibytes:

| Optimizer | Bytes per weight | A 9B model |
| --- | --- | --- |
| gradient descent | 8 | 67.1 GiB |
| with momentum | 12 | 100.6 GiB |
| **AdamW** | **16** | **134.1 GiB** |
| 8-bit AdamW | 10 | 83.8 GiB |

The 8-bit variant stores $m$ and $v$ in one byte each with a shared scale, using the block-scaling method from Session 10, and saves 50 GiB on a 9B model.

> **Carry this forward:** the optimizer is half of the memory required to hold a model in training.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_10_memory.html" width="1280" height="537" title="s11_widget_10_memory"></iframe>

> **Optimizer state memory.** Vary the model size and the choice of rule, and each one crosses the capacity of an 80 GB card at a different point.

## 9\. Learning Rate Warmup

The warmup in this section is a different thing from the mixture warmup band of Sessions 5 and 6, which blended a change of data distribution. **Here we are ramping the learning rate itself.**

Section 6 established that Adam moves each weight by approximately $\eta$. That approximation depends on how consistent the gradients have been:

| gradient behaviour | resulting step |
| --- | --- |
| same sign on every step | **1.000 $\eta$** |
| noisy, averaging to zero | **0.281 $\eta$** |

The first row is the exact case from Section 6: a gradient of unchanging sign makes $\hat{m}$ and $\sqrt{\hat{v}}$ equal, so the ratio is 1. The second is measured rather than derived, by running the same rule on gradients drawn independently with mean zero, where $\hat{m}$ largely cancels while $\sqrt{\hat{v}}$ does not.

Through most of a run the gradients disagree with each other and every weight takes a fraction of $\eta$. **At the start of a run they agree**, because the model is randomly initialized and almost every weight is wrong in the same direction, so every weight takes close to the full step, repeatedly.

Weights are initialized at a scale of $1/\sqrt{\text{fan-in}}$, so that a layer's output has about the same size as its input however wide it is. At $d_{\text{model}} = 4{,}096$ that is $1/\sqrt{4096} = 0.0156$. A full step of $\eta = 0.0003$ therefore moves a weight by

$$
\frac{0.0003}{0.0156} = 0.0192
$$

of its own size, on the first step, in a direction chosen by an untrained model. A healthy run sits near $0.001$, so the starting value is **nineteen times too large**. Warmup corrects this by raising $\eta$ from near zero over the first few thousand steps, during which the gradients become less correlated.

That ratio, the size of the update divided by the size of the weight, logged per layer, is the quantity to monitor. **It should stay near $10^{-3}$ for the whole run.**

> **Carry this forward:** warmup is needed because early gradients are correlated, and correlated gradients produce Adam's largest possible step.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_6_warmup.html" width="1280" height="729" title="s11_widget_6_warmup"></iframe>

> **Learning rate warmup.** The update-to-weight ratio over 10,000 steps. The largest ratio the run ever sees falls from 19.2e-3 without warmup to 2.83e-3 with it.

## 10\. Learning Rate Schedules

Warmup fixes the first part of the learning rate's trajectory, and the rest of it must also be specified. Large steps early cover distance quickly and small steps late let the model settle, so every schedule rises and then falls. They differ in the shape of the fall.

**Cosine decay** brings the learning rate from its peak to near zero along a cosine curve. It has been the standard choice for several years and it performs well, but it carries a structural constraint. The curve is defined in terms of the total number of steps, so **the length of the run must be fixed before the first step is taken.** A run stopped early has not completed its decay, and the model it leaves behind is worse than one trained to that shorter length deliberately.

**WSD**, for warmup, stable and decay, holds the peak rate flat for an unspecified duration and decays only over the final few percent of the run. Because the flat phase has no predetermined end, we can save the weights at any point, which is called a **checkpoint**, and decay separately from there. One run therefore yields finished models at many budgets and can continue afterwards.

<svg id="mermaid-38" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 648.140625px;" viewBox="0 0 648.140625 216" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-38_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-38_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-38_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-38_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-38_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-38_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-38_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-38_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-38_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-38_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-38_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-38_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 162.8 108.0 L 174.3 108.0 Q 185.8 108.0 185.8 108.0 L 185.8 108.0 Q 185.8 108.0 197.3 108.0 L 208.8 108.0" id="mermaid-38-L_W_S_0" style=";" data-edge="true" data-et="edge" data-id="L_W_S_0" data-points="W3sieCI6MTYyLjgxMjUsInkiOjEwOH0seyJ4IjoxODcuODEyNSwieSI6MTA4fSx7IngiOjIxMi44MTI1LCJ5IjoxMDh9XQ==" data-look="classic" marker-end="url(#mermaid-38_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 359.8 70.5 L 400.6 70.5 Q 412.6 70.5 412.6 58.5 L 412.6 57.5 Q 412.6 45.5 424.6 45.5 L 465.3 45.5" id="mermaid-38-L_S_D_0" style=";" data-edge="true" data-et="edge" data-id="L_S_D_0" data-points="W3sieCI6MzU5LjgxMjUsInkiOjcwLjV9LHsieCI6NDAyLjgxMjUsInkiOjQ1LjV9LHsieCI6NDY5LjMxMjUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-38_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 359.8 145.5 L 379.8 145.5 Q 391.8 145.5 391.8 157.5 L 391.8 158.5 Q 391.8 170.5 403.8 170.5 L 423.8 170.5" id="mermaid-38-L_S_C_0" style=";" data-edge="true" data-et="edge" data-id="L_S_C_0" data-points="W3sieCI6MzU5LjgxMjUsInkiOjE0NS41fSx7IngiOjQwMi44MTI1LCJ5IjoxNzAuNX0seyJ4Ijo0MjcuODEyNSwieSI6MTcwLjV9XQ==" data-look="classic" marker-end="url(#mermaid-38_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_W_S_0" transform="translate(0, 0)"></g></g><g><g data-id="L_S_D_0" transform="translate(0, 0)"></g></g><g><g data-id="L_S_C_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-38-flowchart-W-0" data-look="classic" transform="translate(85.40625, 108)"><rect style="" x="-77.40625" y="-37.5" width="154.8125" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-47.40625, -22.5)"><rect></rect><foreignObject width="94.8125" height="45"><p>warmup<br>~2% of steps</p></foreignObject></g></g><g id="mermaid-38-flowchart-S-1" data-look="classic" transform="translate(295.3125, 108)"><rect style="" x="-82.5" y="-37.5" width="165" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-52.5, -22.5)"><rect></rect><foreignObject width="105" height="45"><p>stable<br>flat, any length</p></foreignObject></g></g><g id="mermaid-38-flowchart-D-2" data-look="classic" transform="translate(533.9765625, 45.5)"><rect style="" x="-64.6640625" y="-37.5" width="129.328125" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-34.6640625, -22.5)"><rect></rect><foreignObject width="69.328125" height="45"><p>decay<br>last ~10%</p></foreignObject></g></g><g id="mermaid-38-flowchart-C-4" data-look="classic" transform="translate(533.9765625, 170.5)"><rect style="" x="-106.1640625" y="-37.5" width="212.328125" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-76.1640625, -22.5)"><rect></rect><foreignObject width="152.328125" height="45"><p>checkpoint here<br>and decay separately</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-38-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

A result published in 2026 goes further. Holding the learning rate **constant for the whole run** while keeping an exponential moving average of the weights, which is the mechanism of Section 4 applied to $w$ rather than to $g$, produces averaged weights that match cosine at every point along the run. The decay phase and weight averaging turn out to serve the same purpose, which is suppressing noise over the final stretch. This has been measured at 150M and 300M parameters and not at our scale, so it stays unconfirmed for a run of our size.

> **Carry this forward:** cosine needs the length of the run in advance and WSD does not, which matters more than any difference in final loss.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_7_schedules.html" width="1280" height="639" title="s11_widget_7_schedules"></iframe>

> **Schedule comparison.** Cosine, WSD and constant with weight averaging on one axis. Move the stopping point and each shape yields a different quality of model.

## 11\. Batch Size and Learning Rate Scaling

Session 10 established that the global batch size is chosen independently of what fits in memory. That choice has a consequence, because it changes the appropriate learning rate.

A gradient computed from a batch is an estimate, and averaging more samples improves it. The average of $N$ independent samples carries $1/\sqrt{N}$ of the noise of a single one, so against a batch of 8 the noise is $\sqrt{8/N}$. At V4's batch of 56 that is $\sqrt{8/56} = 0.378$:

| global batch | noise, relative to 8 |
| --- | --- |
| 8 | 1.000 |
| 32 | 0.500 |
| **56**, which V4 used | **0.378** |
| 128 | 0.250 |
| 512 | 0.125 |

A more accurate gradient can be trusted over a longer step, so a larger batch supports a larger learning rate. **For gradient descent the relationship is linear**, and four times the batch permits four times the rate. **For Adam it is the square root**, and four times the batch permits twice the rate, because Adam has already divided out the magnitude of the gradient and only its consistency remains.

Two separate mechanisms raise the global batch, and the learning rate distinguishes neither of them. Session 10 built **gradient accumulation**, which runs several micro-batches one after another and steps once at the end. **Data parallelism** runs those micro-batches at the same time on several GPUs and averages their gradients before the step. To the optimizer these are the same change.

$$
\text{global batch} = \text{micro-batch} \times \text{GPUs} \times \text{accumulation steps}
$$

V4 ran a micro-batch of 7 on each of 8 GPUs with no accumulation, which is where its global batch of 56 comes from.

| Change | Global batch | Learning rate under Adam |
| --- | --- | --- |
| accumulation steps from 1 to 4 | x4 | **x2** |
| GPUs from 8 to 32 | x4 | **x2** |
| both together | x16 | **x4** |

The trap is in how the gradients are combined. Data parallelism averages across GPUs, so nothing further is required. Accumulation adds, so the sum has to be divided before the step, and Session 10 showed that dividing by the number of micro-batches instead of by the number of tokens is itself a bug. Getting this wrong multiplies the learning rate by the accumulation count without anyone having chosen to.

Both relationships have a limit. Past a threshold called the **critical batch size** the gradient is already accurate enough that more samples buy almost nothing, and the extra compute is spent for no return. Where that threshold sits is a property of the model and the data, and it has to be measured.

> **Carry this forward:** under Adam, four times the batch permits twice the learning rate, up to the critical batch size. More GPUs and more accumulation steps are the same change, because only the global batch enters the rule.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_8_batch.html" width="1280" height="763" title="s11_widget_8_batch"></iframe>

> **Batch size and learning rate.** The linear rule against the square-root rule, and the point at which doubling the batch produces no further improvement.

## 12\. Hyperparameter Transfer (muP)

Finding the learning rate by sweeping it on a small model and applying the result to a large one would be inexpensive, but it is not valid, because **the best learning rate depends on the width of the model.** Under the standard parameterization it varies roughly in inverse proportion to width:

| width | approximate best $\eta$ |
| --- | --- |
| 256 | $3.0 \times 10^{-3}$ |
| 512 | $1.5 \times 10^{-3}$ |
| 1,024 | $7.5 \times 10^{-4}$ |
| 2,048 | $3.8 \times 10^{-4}$ |
| **4,096** | $1.9 \times 10^{-4}$ |

Carrying the value from width 256 across to width 4,096 would overstate it by a factor of **sixteen**.

**muP** changes how the initialization scale and the per-layer learning rates depend on width, so that this dependence cancels. Under it the curve of loss against learning rate has its minimum at the same value at every width, and a sweep at width 256 therefore determines the value for width 4,096. The same transfer has been demonstrated for weight decay, at model sizes of several billion parameters.

For this run the difference is one day of small sweeps against one estimate.

> **Carry this forward:** muP does not improve the model. It makes the small model's measurement valid for the large one.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_9_transfer.html" width="1280" height="566" title="s11_widget_9_transfer"></iframe>

> **Learning rate transfer.** Loss against learning rate at four widths, with the minima drifting leftwards under the standard parameterization and aligning under muP.

## 13\. Matrix Optimizers: Muon, MuonClip, Hyperball

Adam treats a weight matrix as a collection of independent numbers and scales each one separately. A matrix is not a collection of independent numbers. It maps an input vector to an output vector, and it amplifies some input directions more than others. Those amplification factors are the matrix's **singular values**, and when a few of them are much larger than the rest, nearly all of the matrix's effect is confined to a few directions while the others contribute very little.

The momentum matrix has exactly this property. **Muon** replaces it with the nearest matrix whose singular values are all equal to 1, so every direction is updated by the same amount. It stores only $m$, which puts it at 12 bytes per weight against AdamW's 16.

The reported results:

| Finding | 2026 |
| --- | --- |
| Speedup over a **well tuned** AdamW | 1.4x at 0.1B, falling to **1.1x at 1.2B** |
| Tokens to reach the same loss at very large batches | **10 to 15% fewer** |
| Batch size at which AdamW destabilises and Muon holds | around 100M tokens |
| Kimi K2, one trillion parameters | **15.5T tokens, zero loss spikes** |

The first row governs the decision. **Most published two-fold speedups were measured against an AdamW baseline that had not been tuned to the same standard.** When both sides are tuned equally the advantage is real but modest, and it decreases as the model grows.

Muon introduces a failure of its own. Making every direction equally strong lets the query and key matrices grow without limit, and the attention scores of Session 8 then rise past 1,000, at which point the softmax saturates and the run fails. **MuonClip** rescales the query and key matrices directly after each update, so the correction lands on the weights rather than after the softmax. This is the mechanism that carried Kimi K2 through 15.5 trillion tokens without a loss spike.

**Hyperball**, published in June 2026, replaces weight decay's indirect control of matrix magnitude with a direct projection onto a fixed magnitude after every step. It reports 20 to 30% over AdamW at 1.2B, increasing with the length of the run, against approximately 10% for Muon with conventional decay.

One constraint holds across every recipe that has been shown to work. **Muon is applied only to two-dimensional weight matrices.** Embeddings, normalization scales and the output head stay on AdamW, because they do not amplify directions and the argument does not apply to them.

> **Carry this forward:** a reported speedup is a statement about someone else's baseline until both sides have been tuned to the same standard.

## 14\. V5 Decisions

This session settles four decisions. We train with **AdamW** and decoupled decay, excluding normalization scales and biases from it. We **warm up** over the first few thousand steps, and we log the **update-to-weight ratio per layer** from step one, alongside the grad norm that Session 10 put on the dashboard. We treat $\eta\lambda$ as one setting rather than two. The schedule is **WSD**, because the run has to be able to stop, branch and continue.

Four decisions stay open.

| Question | What would settle it |
| --- | --- |
| **AdamW or Muon** for the two-dimensional matrices? | A matched short run at our real width with both sides tuned to the same standard, taking the 1.1x figure as the prior. |
| **Whether to spend a day on a muP sweep** at widths 256 to 2,048. | The sweep costs about one day and fixes every hyperparameter that follows, which makes it the highest-value day available to us. |
| **One learning rate, or a separate one for the output head?** | The head is 536.9M weights and dense. Log its update-to-weight ratio against the body's for a thousand steps and compare. |
| **Whether to decay the Kronecker factors from Session 7.** | No published result exists. This is an ablation the cohort can run. |

## 15\. Assignment

**Reproduce Adam by hand.** Take one weight and five gradients, compute $m$, $v$, $\hat{m}$, $\hat{v}$ and the resulting step yourself, then check each against PyTorch. They should agree to several decimal places.

**Disable bias correction** and plot the first twenty steps both ways. Report the number of steps after which the difference stops mattering.

**Log the update-to-weight ratio** for every layer, and identify the step at which warmup stops changing it.

**Train the same model twice for 300 steps**, once under cosine and once under WSD, and stop both at step 200. Report both losses and state which model you would keep.

**Sweep the learning rate at widths 256, 512 and 1,024**, plot loss against learning rate, and mark the three minima. State the value you would use at width 4,096 and how confident you are in it.

> Tune both sides before accepting a comparison. **Almost every optimizer claim that failed to replicate was a well tuned method measured against a badly tuned one.**

[Transcript](https://docs.google.com/document/d/16nNdlSP5XRprvgbFSo50RZmi_1bcTGoRWki5jNWsJyc/edit?usp=sharing)

## Video

**Studio**

![](https://www.youtube.com/watch?v=e-J51Ju-Ho0)

**GMeet**

![](https://www.youtube.com/watch?v=EUJMyNbiwTE)

## Session 11: Optimizers and Learning-Rate Schedules

## 1\. Introduction

Session 10 defined the training step as five lines of code, one of which was left unspecified. `loss.backward()` computes a gradient for every weight, and `optimizer.step()` then uses those gradients to move the weights, but how was never explained.

This session covers that. It begins from the observation that a gradient is incomplete information.

> **A gradient specifies the direction in which a weight should move. It does not specify the distance.**

The gradient computed in Session 10 was 64. That value does not prescribe a movement of 64, or of 0.64, or of any particular size. It gives a direction and a relative urgency, and the distance is a separate decision. Every method in this session is a different rule for making that decision.

| Method | How it sets the distance | Section |
| --- | --- | --- |
| learning rate | one fixed value for every weight | 3 |
| momentum | from an average of recent gradients | 4 |
| second moment | separately per weight, from its own gradient history | 5 |
| Adam | both of the above combined | 6 |
| weight decay | adds a constant pull towards zero | 7 |
| warmup | small at first, while the estimates are unreliable | 9 |
| schedule | decreasing over the course of the run | 10 |
| batch scaling | from the number of samples averaged | 11 |
| muP | from the width of the model | 12 |
| Muon | from the shape of the weight matrix | 13 |

<svg id="mermaid-37" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 680.53125px;" viewBox="0 0 680.53125 91" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-37_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-37_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-37_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-37_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-37_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-37_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-37_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-37_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-37_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-37_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-37_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-37_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 177.3 45.5 L 188.8 45.5 Q 200.3 45.5 200.3 45.5 L 200.3 45.5 Q 200.3 45.5 211.8 45.5 L 223.3 45.5" id="mermaid-37-L_S10_S11_0" style=";" data-edge="true" data-et="edge" data-id="L_S10_S11_0" data-points="W3sieCI6MTc3LjI2NTYyNSwieSI6NDUuNX0seyJ4IjoyMDIuMjY1NjI1LCJ5Ijo0NS41fSx7IngiOjIyNy4yNjU2MjUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-37_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 428.4 45.5 L 439.9 45.5 Q 451.4 45.5 451.4 45.5 L 451.4 45.5 Q 451.4 45.5 462.9 45.5 L 474.4 45.5" id="mermaid-37-L_S11_S12_0" style=";" data-edge="true" data-et="edge" data-id="L_S11_S12_0" data-points="W3sieCI6NDI4LjM3NSwieSI6NDUuNX0seyJ4Ijo0NTMuMzc1LCJ5Ijo0NS41fSx7IngiOjQ3OC4zNzUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-37_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_S10_S11_0" transform="translate(0, 0)"></g></g><g><g data-id="L_S11_S12_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-37-flowchart-S10-0" data-look="classic" transform="translate(92.6328125, 45.5)"><rect style="" x="-84.6328125" y="-37.5" width="169.265625" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-54.6328125, -22.5)"><rect></rect><foreignObject width="109.265625" height="45"><p>Session 10<br>loss to gradient</p></foreignObject></g></g><g id="mermaid-37-flowchart-S11-1" data-look="classic" transform="translate(327.8203125, 45.5)"><rect style="" x="-100.5546875" y="-37.5" width="201.109375" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-70.5546875, -22.5)"><rect></rect><foreignObject width="141.109375" height="45"><p>Session 11<br>gradient to distance</p></foreignObject></g></g><g id="mermaid-37-flowchart-S12-2" data-look="classic" transform="translate(575.453125, 45.5)"><rect style="" x="-97.078125" y="-37.5" width="194.15625" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-67.078125, -22.5)"><rect></rect><foreignObject width="134.15625" height="45"><p>Session 12<br>across many GPUs</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-37-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

The methods appear in the order in which they were developed, because each one corrects a specific failure of the one before it.

## 2\. Terminology

Five terms recur throughout the session, each given here with a representative value.

The **learning rate**, written $\eta$, is the factor that converts a gradient into a distance. For a model of our size it usually lies near $0.0003$.

A **hyperparameter** is a setting we choose before training rather than learn during it. The learning rate is one, and the batch size from Session 10 is another.

**Optimizer state** is any quantity the optimizer stores about a weight between steps. Plain gradient descent stores none. The two values in the Session 10 widget are optimizer state, and Section 6 identifies them.

An **epoch** is one complete pass over the training data. Our run makes roughly one pass, so steps rather than epochs are the useful unit.

A **schedule** is a rule that varies the learning rate over the course of training, and it is the subject of Section 10.

> **Carry this forward:** the learning rate is the most consequential single number in the run, and the rest of this session exists to constrain how it is set.

## 3\. Gradient Descent

The simplest rule that uses a gradient moves each weight against it by a fixed fraction.

$$
w \leftarrow w - \eta \cdot g
$$

This is **gradient descent**. Take one weight starting at 0, and a loss $\mathcal{L} = (w - 5)^2$ that is smallest when the weight reaches 5. Its gradient is $2(w - 5)$, the 2 coming from differentiating the square.

Call the distance still to travel $d = w - 5$. Substituting that gradient into the update rule and then subtracting 5 from both sides gives the distance after one step:

$$
\begin{aligned}
w_{\text{new}} &= w - \eta \cdot 2(w - 5) \\
w_{\text{new}} - 5 &= (w - 5) - 2\eta(w - 5) \\
d_{\text{new}} &= d \cdot (1 - 2\eta)
\end{aligned}
$$

The $(w - 5)$ factors out and leaves a single number multiplying it, so **every step shrinks or grows the remaining distance by the same factor $(1 - 2\eta)$**, and the whole behaviour of the run is decided by that one number.

That factor carries two separate pieces of information. Its **sign** says which side of the minimum the weight lands on: positive and it stayed on the side it started, negative and it crossed over, which is an overshoot. Its **magnitude** says whether the distance shrank or grew: below 1 and the weight is closer than before, above 1 and it is further away. Convergence therefore needs $|1 - 2\eta| < 1$, which for this loss means $\eta$ between 0 and 1.

| $\eta$ | multiplier | $w$ over five steps |
| --- | --- | --- |
| 0.01 | +0.98 | 0.10, 0.20, 0.29, 0.39, 0.48 |
| 0.10 | +0.80 | 1.00, 1.80, 2.44, 2.95, 3.36 |
| 0.90 | −0.80 | 9.00, 1.80, 7.56, 2.95, 6.64 |
| 1.10 | **−1.20** | 11.0, −2.2, 13.6, −5.4, **17.4** |

Four different behaviours from one number. At $0.01$ the weight crawls and will need several hundred steps. At $0.10$ it converges cleanly. At $0.90$ it overshoots the target every step but the overshoot shrinks, so it still converges. At $1.10$ the overshoot grows instead, and the weight is further from the target after five steps than when it started.

The same three lines work on any loss of this shape. If the gradient is $c \cdot d$ rather than $2d$, the multiplier comes out as $(1 - \eta c)$, where $c$ measures how sharply the loss bends around its minimum. We call $c$ the **curvature**, and it was 2 in the example above. Convergence requires $|1 - \eta c| < 1$, so **the safe range for $\eta$ is set by the curvature.**

A real loss does not have one curvature. It bends sharply along some directions and gently along others, and a single $\eta$ has to serve all of them. Consider

$$
\mathcal{L} = \tfrac{1}{2}\left(20u^2 + v^2\right)
$$

which has curvature 20 along $u$ and curvature 1 along $v$. Starting from $(1, 1)$, each step multiplies $u$ by $(1 - 20\eta)$ and $v$ by $(1 - \eta)$, read exactly as above.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_1b_surface.html" width="1280" height="403" title="s11_widget_1b_surface"></iframe>

| $\eta$ | $u$ multiplier | $v$ multiplier | after five steps |
| --- | --- | --- | --- |
| 0.01 | +0.80 | 0.99 | $u$ is converging, $v$ has barely moved at 0.951 |
| 0.09 | −0.80 | 0.91 | $u$ overshoots every step, $v$ is still slow at 0.624 |
| 0.11 | **−1.20** | 0.89 | $u$ has diverged to −2.49 and is growing |

The value that suits $u$ leaves $v$ almost stationary, and the value that moves $v$ makes $u$ diverge. **This is the failure the next three sections correct.**

> **Carry this forward:** one learning rate shared by every weight is the first assumption to break, because the weights do not share a curvature.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_1_descent.html" width="1280" height="678" title="s11_widget_1_descent"></iframe>

> **Gradient descent in two dimensions.** Raise the learning rate and the steep direction starts to oscillate and then diverges, while the shallow direction is still crawling.

## 4\. Momentum

A weight whose gradient has kept the same sign over many steps can safely be moved further than one whose gradient alternates. Using that fact requires a record of recent gradients whose size does not grow with the number of steps.

An **exponential moving average** keeps a single running value and mixes each new observation into it by a fixed fraction:

$$
m \leftarrow \beta_1 \cdot m + (1 - \beta_1) \cdot g
$$

The fraction $\beta_1$ is a hyperparameter and is almost always 0.9. Each new gradient then contributes one tenth while the existing average keeps nine tenths. A gradient's share is multiplied by $\beta_1$ on every later step, so it has faded to almost nothing after about $1/(1 - \beta_1) = 10$ steps, and that is the sense in which $m$ remembers the last ten gradients while storing one number.

The two directions of Section 3 produce gradients of two characteristic shapes. Along the steep direction the sign flips on every step, because the weight lands past the minimum each time. Along the shallow direction the sign never changes, because the weight approaches the minimum from one side and never reaches it.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_2b_directions.html" width="1280" height="389" title="s11_widget_2b_directions"></iframe>

Take the cleanest version of each shape, one alternating between $+1$ and $-1$ and one holding at a constant $+0.2$, five times smaller.

| step | steep direction | shallow direction |
| --- | --- | --- |
| 1 | 0.100 | 0.020 |
| 2 | −0.010 | 0.038 |
| 3 | 0.091 | 0.054 |
| 4 | −0.018 | 0.069 |
| 5 | **0.084** | **0.082** |

The alternating gradients cancel each other while the constant one accumulates, so after five steps the two averages are nearly equal **despite a fivefold difference in the gradients themselves**. Momentum is the use of $m$ in place of $g$ in the update rule.

> **Carry this forward:** an exponential moving average cancels what alternates and accumulates what is consistent.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_2_momentum.html" width="1280" height="687" title="s11_widget_2_momentum"></iframe>

> **Momentum.** Enter gradients by hand and the average follows them, then apply it to the surface from Section 3 and move $\beta_1$ from 0 to 0.99.

## 5\. Per-Parameter Learning Rates

Momentum still leaves every weight sharing one learning rate, and the weights of a language model differ enormously in how often they receive a gradient at all. Session 7 gives the clearest case. The embedding row for a common word receives a large gradient in almost every batch, while the row for a rare word may receive none across a thousand batches. **One learning rate cannot suit both.**

The weights of a model are also called its **parameters**, and the fix is to give each parameter a learning rate of its own, computed from its own gradient history. We keep a second exponential moving average, this time over the squared gradient, which estimates how large that parameter's gradients typically are:

$$
v \leftarrow \beta_2 \cdot v + (1 - \beta_2) \cdot g^2
$$

The subscript distinguishes it from $\beta_1$ above, and $\beta_2$ is almost always 0.999, so by the same $1/(1 - \beta_2)$ count this average spans about a thousand gradients rather than ten. If a parameter's gradient holds steady at $g$, the average settles on $g^2$, so $\sqrt{v}$ settles on $|g|$, which is where the two values in the table below come from. The step is then divided by $\sqrt{v}$, and writing the update out shows what that does:

$$
w \leftarrow w - \frac{\eta}{\sqrt{v}} \cdot g
$$

The quantity $\eta / \sqrt{v}$ is a learning rate belonging to that parameter alone, and it is what gives this section its name. For two parameters, one receiving a gradient of $1.0$ on every step and one receiving $0.01$, after two hundred steps:

| quantity | Parameter A | Parameter B |
| --- | --- | --- |
| gradient $g$ | 1.0000 | 0.0100 |
| $\sqrt{v}$ | 1.0000 | 0.0100 |
| **its own learning rate, $\eta / \sqrt{v}$** | **1.00 $\eta$** | **100.00 $\eta$** |
| resulting step, $\eta g / \sqrt{v}$ | 1.0000 $\eta$ | 1.0000 $\eta$ |

Parameter B has been given a learning rate a hundred times larger than Parameter A's, and the two then take exactly the same step. **The division removes the magnitude of the gradient and keeps only its sign and its consistency.** A parameter that has been inactive for a thousand steps therefore takes a full-sized step when a gradient finally arrives, which is what a rare word's embedding requires.

> **Carry this forward:** $\eta / \sqrt{v}$ is a separate learning rate for every parameter, and it gives them all the same step size whatever their gradients.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_3_per_parameter.html" width="1280" height="689" title="s11_widget_3_per_parameter"></iframe>

> **Per-parameter learning rates.** Two parameters with gradients a hundred times apart, and the learning rate each one is given, before and after the division.

## 6\. Adam

**Adam combines the two averages.** The first sets the direction of the step, the second sets its scale, and a correction removes a bias they share.

Both averages start at zero, so their early values are pulled towards zero and read too low. Feed a constant gradient $g$ into the first average and after $t$ steps it has reached $m_t = g\,(1 - \beta_1^t)$, short of $g$ by exactly the factor $(1 - \beta_1^t)$. Dividing by that factor restores it, and the same argument applies to $v$. This is **bias correction**, and it is exact rather than approximate.

It matters most at the first step, because $\beta_2 = 0.999$ is much closer to 1 than $\beta_1 = 0.9$ and the two averages are therefore pulled down by very different amounts. At $t = 1$ we have $m_1 = 0.1\,g$ and $v_1 = 0.001\,g^2$, so without the correction

$$
\frac{m_1}{\sqrt{v_1}} = \frac{0.1\,g}{0.0316\,|g|} = 3.16
$$

while with it the two corrections cancel and the ratio is exactly $g / |g| = 1$.

| at step 1, with $g = 0.5$ | step taken |
| --- | --- |
| without bias correction | **3.16 $\eta$** |
| with bias correction | **1.00 $\eta$** |

The complete rule is

$$
w \leftarrow w - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

where $\epsilon$ is a very small constant, usually $10^{-8}$, added only so that a weight whose gradients have all been zero is not divided by zero. Computed by hand for five steps at $\eta = 0.001$:

| $t$ | $g$ | $m$ | $v$ | $\hat{m}$ | $\hat{v}$ | step | $w$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.50 | 0.0500 | 0.000250 | 0.5000 | 0.2500 | −0.001000 | 0.999000 |
| 2 | 0.40 | 0.0850 | 0.000410 | 0.4474 | 0.2050 | −0.000988 | 0.998012 |
| 3 | 0.60 | 0.1365 | 0.000769 | 0.5037 | 0.2567 | −0.000994 | 0.997018 |
| 4 | 0.45 | 0.1678 | 0.000971 | 0.4881 | 0.2431 | −0.000990 | 0.996028 |
| 5 | 0.55 | 0.2061 | 0.001273 | 0.5032 | 0.2550 | −0.000996 | 0.995031 |

The step column follows from the two averages. Once both are corrected, $\hat{m}$ is close to the recent average gradient and $\sqrt{\hat{v}}$ is close to its typical size, so their ratio sits near $\pm 1$ and the step sits near $\pm\eta$. The gradients range from 0.40 to 0.60, while **every step falls within half a percent of $0.001$**, which is the learning rate itself. This is the property that made Adam the default: the gradient sets the direction and the learning rate sets the distance, and the two decisions stop interfering with each other.

$m$ and $v$ are the two values stored in the Session 10 widget.

> **Carry this forward:** Adam moves every weight by approximately $\eta$, whatever the magnitude of its gradient.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_4_adam.html" width="1280" height="654" title="s11_widget_4_adam"></iframe>

> **Adam, one weight, five steps.** Every intermediate value as it is computed, with bias correction switchable so the first step moves between 1.00 and 3.16 times $\eta$.

## 7\. Weight Decay and AdamW

Weights tend to grow over a long run, and large weights make a model brittle. The established remedy is **L2 regularization**, which adds a penalty for magnitude directly to the loss:

$$
\mathcal{L}_{\text{total}} = \mathcal{L} + \tfrac{1}{2}\lambda w^2
$$

The derivative of that penalty is $\lambda w$, which joins the gradient. Under plain gradient descent it shrinks every weight by the same fraction on each step. Under Adam it does not, because that added $\lambda w$ is divided by $\sqrt{\hat{v}}$ along with everything else.

For two weights, both sitting at $0.5$, with $\lambda = 0.1$ and $\eta = 0.001$, the decoupled amount is $\eta\lambda w = 0.001 \times 0.1 \times 0.5 = 5.0 \times 10^{-5}$, and the L2 route is that same amount divided by $\sqrt{\hat{v}}$:

| parameter | $\sqrt{\hat{v}}$ | shrinkage per step |
| --- | --- | --- |
| A | 1.00 | $5.0 \times 10^{-5}$ |
| B | 0.01 | $5.0 \times 10^{-3}$ |

Parameter B is decayed **a hundred times more strongly** than Parameter A, for a reason unconnected to how large it is. How much regularization a weight receives has become a function of its gradient history.

**AdamW** applies the shrinkage separately, after the Adam step, where nothing divides it:

$$
w \leftarrow w - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} - \eta \lambda w
$$

Both weights above then shrink by $5.0 \times 10^{-5}$.

A further consequence was established in 2025. The decoupled term makes the final weights an exponential moving average of the updates applied along the way, with a timescale of $1/(\eta\lambda)$ steps. At $\eta = 0.0003$ and $\lambda = 0.1$ that is $1/(0.0003 \times 0.1) =$ **33,333 steps**, so the model we finish with is an average over roughly the last thirty thousand steps of training. It follows that $\eta$ and $\lambda$ are not two independent settings. **Their product is the setting.**

We exclude normalization scales and biases from decay, because shrinking them changes what the layer computes rather than how large it is.

> **Carry this forward:** decoupled decay shrinks every weight equally, and $\eta\lambda$ determines how far back the finished model averages.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_5_decay.html" width="1280" height="625" title="s11_widget_5_decay"></iframe>

> **L2 against decoupled decay.** The same weight down both routes, identical while its gradients are large and a hundred times apart when they are small.

## 8\. Optimizer Memory Cost

Session 10 counted sixteen bytes for every weight in training. Two of the five entries in that count belong to the optimizer, and Section 6 identified them.

| What is stored | Bytes |
| --- | --- |
| the weight, in bf16 | 2 |
| its gradient, in bf16 | 2 |
| a full-precision copy, in fp32 | 4 |
| **$m$, the gradient average** | **4** |
| **$v$, the squared-gradient average** | **4** |
| total | **16** |

**Half the training memory is optimizer state**, so the choice of rule changes the bill directly. A 9B model at 16 bytes a weight needs $9 \times 10^9 \times 16 = 144$ billion bytes, and dividing by $2^{30}$ turns that into gibibytes:

| Optimizer | Bytes per weight | A 9B model |
| --- | --- | --- |
| gradient descent | 8 | 67.1 GiB |
| with momentum | 12 | 100.6 GiB |
| **AdamW** | **16** | **134.1 GiB** |
| 8-bit AdamW | 10 | 83.8 GiB |

The 8-bit variant stores $m$ and $v$ in one byte each with a shared scale, using the block-scaling method from Session 10, and saves 50 GiB on a 9B model.

> **Carry this forward:** the optimizer is half of the memory required to hold a model in training.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_10_memory.html" width="1280" height="537" title="s11_widget_10_memory"></iframe>

> **Optimizer state memory.** Vary the model size and the choice of rule, and each one crosses the capacity of an 80 GB card at a different point.

## 9\. Learning Rate Warmup

The warmup in this section is a different thing from the mixture warmup band of Sessions 5 and 6, which blended a change of data distribution. **Here we are ramping the learning rate itself.**

Section 6 established that Adam moves each weight by approximately $\eta$. That approximation depends on how consistent the gradients have been:

| gradient behaviour | resulting step |
| --- | --- |
| same sign on every step | **1.000 $\eta$** |
| noisy, averaging to zero | **0.281 $\eta$** |

The first row is the exact case from Section 6: a gradient of unchanging sign makes $\hat{m}$ and $\sqrt{\hat{v}}$ equal, so the ratio is 1. The second is measured rather than derived, by running the same rule on gradients drawn independently with mean zero, where $\hat{m}$ largely cancels while $\sqrt{\hat{v}}$ does not.

Through most of a run the gradients disagree with each other and every weight takes a fraction of $\eta$. **At the start of a run they agree**, because the model is randomly initialized and almost every weight is wrong in the same direction, so every weight takes close to the full step, repeatedly.

Weights are initialized at a scale of $1/\sqrt{\text{fan-in}}$, so that a layer's output has about the same size as its input however wide it is. At $d_{\text{model}} = 4{,}096$ that is $1/\sqrt{4096} = 0.0156$. A full step of $\eta = 0.0003$ therefore moves a weight by

$$
\frac{0.0003}{0.0156} = 0.0192
$$

of its own size, on the first step, in a direction chosen by an untrained model. A healthy run sits near $0.001$, so the starting value is **nineteen times too large**. Warmup corrects this by raising $\eta$ from near zero over the first few thousand steps, during which the gradients become less correlated.

That ratio, the size of the update divided by the size of the weight, logged per layer, is the quantity to monitor. **It should stay near $10^{-3}$ for the whole run.**

> **Carry this forward:** warmup is needed because early gradients are correlated, and correlated gradients produce Adam's largest possible step.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_6_warmup.html" width="1280" height="729" title="s11_widget_6_warmup"></iframe>

> **Learning rate warmup.** The update-to-weight ratio over 10,000 steps. The largest ratio the run ever sees falls from 19.2e-3 without warmup to 2.83e-3 with it.

## 10\. Learning Rate Schedules

Warmup fixes the first part of the learning rate's trajectory, and the rest of it must also be specified. Large steps early cover distance quickly and small steps late let the model settle, so every schedule rises and then falls. They differ in the shape of the fall.

**Cosine decay** brings the learning rate from its peak to near zero along a cosine curve. It has been the standard choice for several years and it performs well, but it carries a structural constraint. The curve is defined in terms of the total number of steps, so **the length of the run must be fixed before the first step is taken.** A run stopped early has not completed its decay, and the model it leaves behind is worse than one trained to that shorter length deliberately.

**WSD**, for warmup, stable and decay, holds the peak rate flat for an unspecified duration and decays only over the final few percent of the run. Because the flat phase has no predetermined end, we can save the weights at any point, which is called a **checkpoint**, and decay separately from there. One run therefore yields finished models at many budgets and can continue afterwards.

<svg id="mermaid-38" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width: 648.140625px;" viewBox="0 0 648.140625 216" role="graphics-document document" aria-roledescription="flowchart-v2"><g><marker id="mermaid-38_flowchart-v2-pointEnd" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-38_flowchart-v2-pointStart" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-38_flowchart-v2-pointEnd-margin" viewBox="0 0 11.5 14" refX="11.5" refY="7" markerUnits="userSpaceOnUse" markerWidth="10.5" markerHeight="14" orient="auto"><path d="M 0 0 L 11.5 7 L 0 14 z" style="stroke-width: 0; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-38_flowchart-v2-pointStart-margin" viewBox="0 0 11.5 14" refX="1" refY="7" markerUnits="userSpaceOnUse" markerWidth="11.5" markerHeight="14" orient="auto"><polygon points="0,7 11.5,14 11.5,0" style="stroke-width: 0; stroke-dasharray: 1, 0;"></polygon></marker><marker id="mermaid-38_flowchart-v2-circleEnd" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-38_flowchart-v2-circleStart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-38_flowchart-v2-circleEnd-margin" viewBox="0 0 10 10" refY="5" refX="12.25" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-38_flowchart-v2-circleStart-margin" viewBox="0 0 10 10" refX="-2" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto"><circle cx="5" cy="5" r="5" style="stroke-width: 0; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-38_flowchart-v2-crossEnd" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-38_flowchart-v2-crossStart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-38_flowchart-v2-crossEnd-margin" viewBox="0 0 15 15" refX="17.7" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5;"></path></marker><marker id="mermaid-38_flowchart-v2-crossStart-margin" viewBox="0 0 15 15" refX="-3.5" refY="7.5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1,1 L 14,14 M 1,14 L 14,1" style="stroke-width: 2.5; stroke-dasharray: 1, 0;"></path></marker><g><g></g><g><path d="M 162.8 108.0 L 174.3 108.0 Q 185.8 108.0 185.8 108.0 L 185.8 108.0 Q 185.8 108.0 197.3 108.0 L 208.8 108.0" id="mermaid-38-L_W_S_0" style=";" data-edge="true" data-et="edge" data-id="L_W_S_0" data-points="W3sieCI6MTYyLjgxMjUsInkiOjEwOH0seyJ4IjoxODcuODEyNSwieSI6MTA4fSx7IngiOjIxMi44MTI1LCJ5IjoxMDh9XQ==" data-look="classic" marker-end="url(#mermaid-38_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 359.8 70.5 L 400.6 70.5 Q 412.6 70.5 412.6 58.5 L 412.6 57.5 Q 412.6 45.5 424.6 45.5 L 465.3 45.5" id="mermaid-38-L_S_D_0" style=";" data-edge="true" data-et="edge" data-id="L_S_D_0" data-points="W3sieCI6MzU5LjgxMjUsInkiOjcwLjV9LHsieCI6NDAyLjgxMjUsInkiOjQ1LjV9LHsieCI6NDY5LjMxMjUsInkiOjQ1LjV9XQ==" data-look="classic" marker-end="url(#mermaid-38_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path><path d="M 359.8 145.5 L 379.8 145.5 Q 391.8 145.5 391.8 157.5 L 391.8 158.5 Q 391.8 170.5 403.8 170.5 L 423.8 170.5" id="mermaid-38-L_S_C_0" style=";" data-edge="true" data-et="edge" data-id="L_S_C_0" data-points="W3sieCI6MzU5LjgxMjUsInkiOjE0NS41fSx7IngiOjQwMi44MTI1LCJ5IjoxNzAuNX0seyJ4Ijo0MjcuODEyNSwieSI6MTcwLjV9XQ==" data-look="classic" marker-end="url(#mermaid-38_flowchart-v2-pointEnd)" fill="none" stroke="currentColor"></path></g><g><g><g data-id="L_W_S_0" transform="translate(0, 0)"></g></g><g><g data-id="L_S_D_0" transform="translate(0, 0)"></g></g><g><g data-id="L_S_C_0" transform="translate(0, 0)"></g></g></g><g><g id="mermaid-38-flowchart-W-0" data-look="classic" transform="translate(85.40625, 108)"><rect style="" x="-77.40625" y="-37.5" width="154.8125" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-47.40625, -22.5)"><rect></rect><foreignObject width="94.8125" height="45"><p>warmup<br>~2% of steps</p></foreignObject></g></g><g id="mermaid-38-flowchart-S-1" data-look="classic" transform="translate(295.3125, 108)"><rect style="" x="-82.5" y="-37.5" width="165" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-52.5, -22.5)"><rect></rect><foreignObject width="105" height="45"><p>stable<br>flat, any length</p></foreignObject></g></g><g id="mermaid-38-flowchart-D-2" data-look="classic" transform="translate(533.9765625, 45.5)"><rect style="" x="-64.6640625" y="-37.5" width="129.328125" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-34.6640625, -22.5)"><rect></rect><foreignObject width="69.328125" height="45"><p>decay<br>last ~10%</p></foreignObject></g></g><g id="mermaid-38-flowchart-C-4" data-look="classic" transform="translate(533.9765625, 170.5)"><rect style="" x="-106.1640625" y="-37.5" width="212.328125" height="75" fill="none" stroke="currentColor"></rect><g style="" transform="translate(-76.1640625, -22.5)"><rect></rect><foreignObject width="152.328125" height="45"><p>checkpoint here<br>and decay separately</p></foreignObject></g></g></g></g></g><defs></defs><defs></defs><linearGradient id="mermaid-38-gradient" gradientUnits="objectBoundingBox" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#60A5FA" stop-opacity="1"></stop><stop offset="100%" stop-color="hsl(240, 0%, 0.9803921569%)" stop-opacity="1"></stop></linearGradient></svg>

A result published in 2026 goes further. Holding the learning rate **constant for the whole run** while keeping an exponential moving average of the weights, which is the mechanism of Section 4 applied to $w$ rather than to $g$, produces averaged weights that match cosine at every point along the run. The decay phase and weight averaging turn out to serve the same purpose, which is suppressing noise over the final stretch. This has been measured at 150M and 300M parameters and not at our scale, so it stays unconfirmed for a run of our size.

> **Carry this forward:** cosine needs the length of the run in advance and WSD does not, which matters more than any difference in final loss.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_7_schedules.html" width="1280" height="639" title="s11_widget_7_schedules"></iframe>

> **Schedule comparison.** Cosine, WSD and constant with weight averaging on one axis. Move the stopping point and each shape yields a different quality of model.

## 11\. Batch Size and Learning Rate Scaling

Session 10 established that the global batch size is chosen independently of what fits in memory. That choice has a consequence, because it changes the appropriate learning rate.

A gradient computed from a batch is an estimate, and averaging more samples improves it. The average of $N$ independent samples carries $1/\sqrt{N}$ of the noise of a single one, so against a batch of 8 the noise is $\sqrt{8/N}$. At V4's batch of 56 that is $\sqrt{8/56} = 0.378$:

| global batch | noise, relative to 8 |
| --- | --- |
| 8 | 1.000 |
| 32 | 0.500 |
| **56**, which V4 used | **0.378** |
| 128 | 0.250 |
| 512 | 0.125 |

A more accurate gradient can be trusted over a longer step, so a larger batch supports a larger learning rate. **For gradient descent the relationship is linear**, and four times the batch permits four times the rate. **For Adam it is the square root**, and four times the batch permits twice the rate, because Adam has already divided out the magnitude of the gradient and only its consistency remains.

Two separate mechanisms raise the global batch, and the learning rate distinguishes neither of them. Session 10 built **gradient accumulation**, which runs several micro-batches one after another and steps once at the end. **Data parallelism** runs those micro-batches at the same time on several GPUs and averages their gradients before the step. To the optimizer these are the same change.

$$
\text{global batch} = \text{micro-batch} \times \text{GPUs} \times \text{accumulation steps}
$$

V4 ran a micro-batch of 7 on each of 8 GPUs with no accumulation, which is where its global batch of 56 comes from.

| Change | Global batch | Learning rate under Adam |
| --- | --- | --- |
| accumulation steps from 1 to 4 | x4 | **x2** |
| GPUs from 8 to 32 | x4 | **x2** |
| both together | x16 | **x4** |

The trap is in how the gradients are combined. Data parallelism averages across GPUs, so nothing further is required. Accumulation adds, so the sum has to be divided before the step, and Session 10 showed that dividing by the number of micro-batches instead of by the number of tokens is itself a bug. Getting this wrong multiplies the learning rate by the accumulation count without anyone having chosen to.

Both relationships have a limit. Past a threshold called the **critical batch size** the gradient is already accurate enough that more samples buy almost nothing, and the extra compute is spent for no return. Where that threshold sits is a property of the model and the data, and it has to be measured.

> **Carry this forward:** under Adam, four times the batch permits twice the learning rate, up to the critical batch size. More GPUs and more accumulation steps are the same change, because only the global batch enters the rule.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_8_batch.html" width="1280" height="763" title="s11_widget_8_batch"></iframe>

> **Batch size and learning rate.** The linear rule against the square-root rule, and the point at which doubling the batch produces no further improvement.

## 12\. Hyperparameter Transfer (muP)

Finding the learning rate by sweeping it on a small model and applying the result to a large one would be inexpensive, but it is not valid, because **the best learning rate depends on the width of the model.** Under the standard parameterization it varies roughly in inverse proportion to width:

| width | approximate best $\eta$ |
| --- | --- |
| 256 | $3.0 \times 10^{-3}$ |
| 512 | $1.5 \times 10^{-3}$ |
| 1,024 | $7.5 \times 10^{-4}$ |
| 2,048 | $3.8 \times 10^{-4}$ |
| **4,096** | $1.9 \times 10^{-4}$ |

Carrying the value from width 256 across to width 4,096 would overstate it by a factor of **sixteen**.

**muP** changes how the initialization scale and the per-layer learning rates depend on width, so that this dependence cancels. Under it the curve of loss against learning rate has its minimum at the same value at every width, and a sweep at width 256 therefore determines the value for width 4,096. The same transfer has been demonstrated for weight decay, at model sizes of several billion parameters.

For this run the difference is one day of small sweeps against one estimate.

> **Carry this forward:** muP does not improve the model. It makes the small model's measurement valid for the large one.

<iframe src="https://axiom.theschoolofai.in/widgets/s11_widget_9_transfer.html" width="1280" height="566" title="s11_widget_9_transfer"></iframe>

> **Learning rate transfer.** Loss against learning rate at four widths, with the minima drifting leftwards under the standard parameterization and aligning under muP.

## 13\. Matrix Optimizers: Muon, MuonClip, Hyperball

Adam treats a weight matrix as a collection of independent numbers and scales each one separately. A matrix is not a collection of independent numbers. It maps an input vector to an output vector, and it amplifies some input directions more than others. Those amplification factors are the matrix's **singular values**, and when a few of them are much larger than the rest, nearly all of the matrix's effect is confined to a few directions while the others contribute very little.

The momentum matrix has exactly this property. **Muon** replaces it with the nearest matrix whose singular values are all equal to 1, so every direction is updated by the same amount. It stores only $m$, which puts it at 12 bytes per weight against AdamW's 16.

The reported results:

| Finding | 2026 |
| --- | --- |
| Speedup over a **well tuned** AdamW | 1.4x at 0.1B, falling to **1.1x at 1.2B** |
| Tokens to reach the same loss at very large batches | **10 to 15% fewer** |
| Batch size at which AdamW destabilises and Muon holds | around 100M tokens |
| Kimi K2, one trillion parameters | **15.5T tokens, zero loss spikes** |

The first row governs the decision. **Most published two-fold speedups were measured against an AdamW baseline that had not been tuned to the same standard.** When both sides are tuned equally the advantage is real but modest, and it decreases as the model grows.

Muon introduces a failure of its own. Making every direction equally strong lets the query and key matrices grow without limit, and the attention scores of Session 8 then rise past 1,000, at which point the softmax saturates and the run fails. **MuonClip** rescales the query and key matrices directly after each update, so the correction lands on the weights rather than after the softmax. This is the mechanism that carried Kimi K2 through 15.5 trillion tokens without a loss spike.

**Hyperball**, published in June 2026, replaces weight decay's indirect control of matrix magnitude with a direct projection onto a fixed magnitude after every step. It reports 20 to 30% over AdamW at 1.2B, increasing with the length of the run, against approximately 10% for Muon with conventional decay.

One constraint holds across every recipe that has been shown to work. **Muon is applied only to two-dimensional weight matrices.** Embeddings, normalization scales and the output head stay on AdamW, because they do not amplify directions and the argument does not apply to them.

> **Carry this forward:** a reported speedup is a statement about someone else's baseline until both sides have been tuned to the same standard.

## 14\. V5 Decisions

This session settles four decisions. We train with **AdamW** and decoupled decay, excluding normalization scales and biases from it. We **warm up** over the first few thousand steps, and we log the **update-to-weight ratio per layer** from step one, alongside the grad norm that Session 10 put on the dashboard. We treat $\eta\lambda$ as one setting rather than two. The schedule is **WSD**, because the run has to be able to stop, branch and continue.

Four decisions stay open.

| Question | What would settle it |
| --- | --- |
| **AdamW or Muon** for the two-dimensional matrices? | A matched short run at our real width with both sides tuned to the same standard, taking the 1.1x figure as the prior. |
| **Whether to spend a day on a muP sweep** at widths 256 to 2,048. | The sweep costs about one day and fixes every hyperparameter that follows, which makes it the highest-value day available to us. |
| **One learning rate, or a separate one for the output head?** | The head is 536.9M weights and dense. Log its update-to-weight ratio against the body's for a thousand steps and compare. |
| **Whether to decay the Kronecker factors from Session 7.** | No published result exists. This is an ablation the cohort can run. |

## 15\. Assignment

**Reproduce Adam by hand.** Take one weight and five gradients, compute $m$, $v$, $\hat{m}$, $\hat{v}$ and the resulting step yourself, then check each against PyTorch. They should agree to several decimal places.

**Disable bias correction** and plot the first twenty steps both ways. Report the number of steps after which the difference stops mattering.

**Log the update-to-weight ratio** for every layer, and identify the step at which warmup stops changing it.

**Train the same model twice for 300 steps**, once under cosine and once under WSD, and stop both at step 200. Report both losses and state which model you would keep.

**Sweep the learning rate at widths 256, 512 and 1,024**, plot loss against learning rate, and mark the three minima. State the value you would use at width 4,096 and how confident you are in it.

> Tune both sides before accepting a comparison. **Almost every optimizer claim that failed to replicate was a well tuned method measured against a badly tuned one.**

[Transcript](https://docs.google.com/document/d/16nNdlSP5XRprvgbFSo50RZmi_1bcTGoRWki5jNWsJyc/edit?usp=sharing)

## Video

**Studio**

![](https://www.youtube.com/watch?v=e-J51Ju-Ho0)

**GMeet**

![](https://www.youtube.com/watch?v=EUJMyNbiwTE)
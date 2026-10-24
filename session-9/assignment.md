# Session 9 — Assignment

> Paste the assignment text from the Axiom course portal here
> (the "The Assignment", "Submission", and "Evaluation" sections).

## The Assignment

_TODO: paste spec_

## Submission

_TODO: paste submission requirements_

## Evaluation

_TODO: paste rubric_


---
title: "Axiom — Learning OS"
source: "https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/assignments/cmt3xb6yo0j9e09s1cp2vbydr"
author:
published:
created: 2026-10-24
description: "A premium learning operating system for ambitious technical education."
tags:
  - "clippings"
---
## Session 9 - Assignment QnA

## 24\. The assignment

One notebook, one loss harness, and one thing you have to get right by reading rather than by guessing.

## Part 1: the harness

```
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)
```

Take that and make it correct and observable. You must:

1. **Print every tensor shape** and say in one line what each dimension is.
2. **Verify the shift** by printing the actual token strings, inputs beside targets. Not the ids. The strings. You will not catch an off-by-one in a wall of integers.
3. **Mask padding** and confirm the count of contributing tokens changes.
4. **Pack two documents** into one sequence and mask the boundary. Show the loss before and after masking it, and explain the difference.
5. **Compute perplexity**, and show that an untrained model sits near your vocabulary size. If it does not, find the bug before you go further.
6. **Compare tied against untied** head parameter counts on your configuration.
7. **Measure peak memory** for ordinary cross-entropy against a chunked version you write yourself. Report both numbers and the ratio.

## Part 2: one extra head

Add a second output head predicting token `t+2`. Report both losses separately and their sum, and say what happens to the second head's loss over training compared with the first. Explain what you see.

## What to submit

A Google Colab notebook **moved to GitHub** that runs top to bottom, and a short write-up with the seven numbers from Part 1, the two losses from Part 2

> One warning, the same one as last time. A target shift in the incorrect direction can produce a beautiful loss curve. **Print the strings.** Many serious training bugs live in the few lines between the model output and the scalar, and they do not always raise an exception.

What are you submitting? **GitHub README.md** link.

### Your submission

Not submitted

#### GitHub README.md link

Github repo should have the ipynb file or training logs as well to backyo your readme

1000 pts

0/1 answered · 1 will be left blank
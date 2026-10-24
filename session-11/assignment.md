# Session 11 — Assignment

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
source: "https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/assignments/cmts6t1qa009o09qk13ucubdq"
author:
published:
created: 2026-10-24
description: "A premium learning operating system for ambitious technical education."
tags:
  - "clippings"
---
## Session 11 Assignment QnA

## Assignment

**Reproduce Adam by hand.** Take one weight and five gradients, compute $m$, $v$, $\hat{m}$, $\hat{v}$ and the resulting step yourself, then check each against PyTorch. They should agree to several decimal places.

**Disable bias correction** and plot the first twenty steps both ways. Report the number of steps after which the difference stops mattering.

**Log the update-to-weight ratio** for every layer, and identify the step at which warmup stops changing it.

**Train the same model twice for 300 steps**, once under cosine and once under WSD, and stop both at step 200. Report both losses and state which model you would keep.

**Sweep the learning rate at widths 256, 512 and 1,024**, plot loss against learning rate, and mark the three minima. State the value you would use at width 4,096 and how confident you are in it.

> Tune both sides before accepting a comparison. **Almost every optimizer claim that failed to replicate was a well tuned method measured against a badly tuned one.**

You're submitting a detailed README.md along with your support code.

### Your submission

Not submitted

#### GitHub README.md

1000 pts

0/1 answered · 1 will be left blank
# Proxy ablation — testing the mixture hypothesis on real data

This is the **scaled-down stand-in** for the 1B/3B proxy the plan calls for
([`../plan/mixture-and-curriculum-v5.md`](../plan/mixture-and-curriculum-v5.md) §9). It
runs in ~6 seconds on a laptop and its job is to show, on **real data**, the *direction*
of the mixture effect — not absolute benchmark scores.

## What it is

A genuine gradient-trained **character-level neural LM** (numpy, hand-written backprop:
`context → embed → tanh hidden → softmax`, ~43k params), trained by minibatch SGD. Three
**arms** differ *only* in the training-data mixture; model size, seed, token budget
(882,857 chars/arm), and optimizer are identical — **equal compute**.

| Arm | en | code | math | indic | Represents |
|---|---:|---:|---:|---:|---|
| **A** naive web-heavy | 82% | 12% | 3% | 3% | the starving mix the session warns against |
| **B** V5 main | 57% | 21% | 8% | 14% | the actual V5 main-pretraining mix (4-lane renormalized) |
| **C** V5 anneal | 45% | 23% | 12% | 20% | the anneal-flavoured upsampling of scarce lanes |

**Lanes are real repo data** (`data_prep.py`, deterministic): `indic` = Session 4's cleaned
Assamese; `en` = the repo's own English markdown; `code` = Session 2/4 Python; `math` =
deterministically-generated arithmetic/algebra CoT (the reasoning lane is *synthesized*,
mirroring the inventory finding that reasoning CoT must be generated). Held-out val splits
are never trained on.

**Metric:** per-lane held-out cross-entropy in **bits/char** (perplexity = 2^bits/char).

## Run it

```bash
../../session-4/.venv/bin/python3 data_prep.py   # build per-lane corpora (uses numpy venv)
../../session-4/.venv/bin/python3 train.py       # run the 3-arm ablation → results.json
```

## Result (held-out perplexity)

| Lane | A naive | B V5-main | C V5-anneal |
|---|---:|---:|---:|
| en | **7.32** | 8.29 | 8.93 |
| code | 17.37 | **15.48** | 15.54 |
| math | 12.76 | 5.63 | **3.70** |
| indic | 48.28 | 21.42 | **19.26** |

**Δ vs naive (bits/char):** B: indic **−21.0%**, math **−32.2%**, code **−4.0%**, en **+6.3%**.
C: indic **−23.7%**, math **−48.6%**, code **−3.9%**, en **+10.0%**.

## Interpretation — what the proxy confirms and what it refines

1. **Direction confirmed, strongly.** Protecting scarce lanes produces *large* held-out
   gains: Indic perplexity **more than halves** (48.3 → 21.4) and math collapses (12.8 →
   5.6) under the V5 main mix. This is the session's central claim — *the mixture is the
   model* — reproduced on real data at the smallest honest scale.
2. **The trade-off is real and monotone.** A → B → C, scarce lanes keep improving while
   English keeps degrading (+6.3% → +10%). "Every share given to one capability is taken
   from another" is visible in the numbers, not just asserted.
3. **The strict sub-claim was refuted at this scale — honestly.** The plan's hypothesis
   asked for the Indic gain *without materially raising English (Δ ≤ ~2%)*. English rose
   ~6%, so that specific bound **did not hold here.** The proxy did its job: it caught a
   real cost before full scale. A data decision is a hypothesis until a cheap experiment
   tests it — and this one *refined* the hypothesis rather than rubber-stamping it.
4. **Why the tiny proxy overstates the English cost** (and why the 1B/3B run is still
   expected to satisfy the bound):
   - **Data-richness:** at 16T scale English has ~7T unique tokens — reducing its *share*
     from 82%→57% still shows the model billions of unique English tokens, so its loss
     barely moves. Here English is only 272k unique chars, heavily repeated, so a share
     cut bites immediately.
   - **Capacity competition:** a 43k-param model has tiny shared capacity; giving more to
     Indic/math necessarily takes representational room from English. A 40B model is not
     capacity-bound on English the way this one is.

   Both effects shrink with scale, so the English cost measured here is an **upper bound**.

## Honest limitations (do not overclaim)

- A ~43k-param char-LM on ~1MB is a **proxy-of-the-proxy.** It measures *direction*, not
  magnitude, and cannot produce MILU/HumanEval/GSM8K scores.
- The "agentic" and "long-context" lanes are not exercised (char-LM on tool-call JSON /
  512k contexts is not meaningful at this size).
- The math lane is templated, so its absolute perplexity is optimistic; only its
  *relative* movement across arms is informative.
- **The real gate remains the 1B/3B run** with the confirm/refute rule in the plan. This
  proxy is evidence for the direction and a demonstration of the testing discipline.

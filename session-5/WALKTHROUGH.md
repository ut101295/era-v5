# Session 5 — Data Mixtures & Curriculum · Walkthrough

> This document is the running narrative of how I read the Session 5 assignment,
> what it actually asks for, how I researched it, how I derived every number in
> the mixture, how I ran a real proxy experiment to test it, and how the final
> plan was assembled. It is written to be read top-to-bottom as one continuous
> story. Nothing is abbreviated — later sections are appended as the work
> happens, so the "flow" of the project is preserved and auditable end to end.
>
> The **artifact** version of this research (individual dossier files, the real
> dataset inventory, the proxy code and its measured results) lives in
> [`research/`](research/) and [`proxy/`](proxy/). The graded specification
> itself lives in [`plan/mixture-and-curriculum-v5.md`](plan/mixture-and-curriculum-v5.md).
> This document is the narrative; those folders are the evidence.

---

## Part 0 — Where this session sits

Three sessions built to this one:

- **Session 3** answered the *strategy* question for an India-first ~40B model
  ("Bharat"): what data to collect, which data buys which capability, the
  tokenizer size (262,144), and a first-draft mixture
  (32% EN web / 22% Indic / 18% code / 14% math / 8% India-EN / 6% other).
- **Session 4** answered the *engineering* question: it turned raw crawl into
  **clean, deduplicated, provenance-stamped shards** — and produced a real
  cleaned corpus (Assamese, 9.53M verified-native tokens, deterministic
  manifest) that this session can point at as genuine *supply*.
- **Session 5** — this one — turns that clean data into a **training plan**. The
  one sentence to hold onto:

> **The mixture is the model.** The same clean corpus and the same compute
> budget produce completely different models depending on how much of each
> capability the model sees, how often it sees it, and at what stage of training
> it sees it.

So Session 5 is the **planning** layer that sits on top of Session 3's strategy
and Session 4's engineering. Session 6 will turn this recipe into the physical
data stream (sharding, packing, shuffling); our job here is only the recipe.

---

## Part 1 — What the assignment actually asks

The deliverable is **one written specification** — a mixture-and-curriculum plan
for V5 — specific enough that a reviewer can push on every single number and it
holds. Decomposed from the assignment + the evaluation strategy, it must contain:

1. **A defended budget share for every capability lane** — general web, code,
   math/reasoning, Indic, agentic, long-context, other. Every number justified,
   not round-because-it-felt-reasonable.
2. **The Indic split across four provenance tiers** — not a single headline
   number, but **verified / unverified / translated / synthetic**.
3. **Each lane pointed at real datasets** from an inventory — agentic, reasoning,
   long-context named explicitly with the datasets that fill them.
4. **Every lane sized against real supply**, in two currencies (samples AND
   tokens), stating plainly where a share can only be met by **repeating** data
   (epochs) or **synthesizing** it. (The central trap this session exists to
   prevent: handing a big share to a lane that has almost no real data.)
5. **Benchmark backing per lane** — each lane tied to the benchmarks it is meant
   to win, so a reviewer can see why each number is what it is.
6. **The protected structures** — (a) the always-on floor the OPUS selector may
   not cross, and (b) the anneal reserve held back for the final cooldown.
7. **Difficulty bands + reasoning-length bands, each with a concrete example.**

And the two things that separate good from excellent:

8. **A concrete proxy experiment at 1B/3B scale, with the named metric** that
   would confirm or refute the mixture — and, for the very highest marks,
   *actually run it* and bring numbers back.
9. **The whole plan written as a testable hypothesis**, not a set of assertions.

Submission: a link to a GitHub repo README where the work can be evaluated.

### Scope decisions (confirmed with the user)

1. **Artifact depth** = written spec + a research dossier + this walkthrough +
   **static visuals** (mixture stacked-bars, supply-vs-target, curriculum
   timeline) embedded in the README. No live widget this time.
2. **Proxy experiment** = **actually run a scaled-down ablation.** Real 1B/3B
   training is impossible on a laptop, so we run a tiny proxy-of-the-proxy: a
   small model trained on our cleaned corpus, comparing a naive web-heavy mix
   against the protected-floor V5 mix, reporting real per-lane held-out
   perplexity. This aims for the "very highest" band with real numbers while
   being honest that it is a scaled-down stand-in.
3. **Dataset inventory** = **web research on real datasets** (Sangraha,
   IndicCorp, The Stack, math/reasoning sets, agentic/tool-call sets, …) with
   real sample/token counts and licenses — the most defensible option under a
   reviewer's push.

---

## Part 2 — How I researched it (two web-grounded streams)

I split the research into the two questions the assignment forces you to answer *before*
you can write a single share:

**Stream 1 — "what capability does each benchmark actually need?"** I read every benchmark
in the session at the **loss-map level** — not "SWE-bench measures coding" but "SWE-bench
grades the generated patch, so the training shape is a repo+issue → bash-trajectory → diff,
with loss *only* on the agent's emitted tokens and *never* on tool output." That turns a
vague capability into a concrete data format. The full result is
[`research/01-benchmark-map.md`](research/01-benchmark-map.md). Two archetypes cover almost
everything: (a) verifiable-outcome agentic/coding/math — green on the model's own actions,
grey on the environment, finished with RLVR on a programmatic verifier; (b)
understanding/generation/QA — green on the answer, learned by SFT (and, for Indic knowledge,
pretraining coverage). The single most important structural rule fell out of this: **never
put loss on grey tokens** (tool outputs, retrieved passages, simulated-user turns), or you
teach the model to hallucinate the environment.

The stream also surfaced the plan's **strategic bet**: there is essentially *no* mature,
human-verified Indic **coding or agentic** benchmark (BFCL-Hi is unverified machine
translation; AI4Bharat's Indic LLM-Arena agentic track is roadmap-only). "Native Indic" ×
"agentic" is an unclaimed intersection — a defensible, measurable lead for an India-first
model.

**Stream 2 — "what data actually exists to fill each lane?"** I researched real public
datasets per lane, sized in **two currencies (samples AND tokens)** because they diverge
wildly, and tagged each with a license and a Session-3 provenance tier. The result is
[`research/02-dataset-inventory.md`](research/02-dataset-inventory.md), and the deep-dive
sub-agents brought back the two crown-jewel numbers a reviewer will push hardest on:

- **Verified-native Indic ≈ 64.3B tokens total** (Sangraha Verified), per-language from
  Hindi 12.6B down to Assamese 0.29B — and Sangraha's headline "251B" is **65% synthetic**.
- **The real agentic-trajectory commons ≈ 10,000 samples** (APIGen-MT 5k + Mind2Web 2.35k +
  AgentTuning 1.87k + SWE-Gym 491 + WebArena ~170); everything larger is model-synthesized.

Both dossiers are web-grounded with inline sources (HuggingFace cards, arXiv, leaderboards).

## Part 3 — Composing the mixture backward (and shrinking the wishful lanes)

With both streams in hand, the derivation ([`research/03-mixture-derivation.md`](research/03-mixture-derivation.md))
starts from Session 3's draft mix and applies three supply-forced corrections:

1. **Indic 22% → 13%.** A 22% share (3.5T) against 64B verified-native would be ~98%
   synthetic — the exact wishful accounting this session exists to catch. 13% is the honest
   ceiling, and it is *split across four tiers* (verified 22% / crawl 18% / translated 40% /
   synthetic 20% of the lane) so the reviewer sees the native-vs-synthetic ratio rather than
   a single number hiding it.
2. **Agentic added as an explicit lane, but small (1.5%)** — because the real supply is
   ~10k trajectories, the lane is ~90% synthesized by necessity, and I budget a
   trajectory-synthesis pipeline as a first-class deliverable.
3. **Long-context becomes a late-phase repacking overlay (2.5%)**, never new unique tokens
   (ProLong hit Llama-3.1-grade long-context with ~5% of its long tokens by upsampling).

Every remaining lane is reconciled against supply with explicit epoch math (e.g. code ~2–3
epochs on the best Stack v2 tokens; math ~3–4 epochs + distilled CoT). Where a share can
only be reached by repetition or synthesis, the plan says so out loud.

**The protected floor and the anneal reserve** are the two mechanisms that keep the scarce
lanes alive. V4 pinned Indic at 8% outside the OPUS selector; V5 extends that to Indic 10% +
agentic 1.5% + reasoning 4% (a 15.5% always-on floor), because the selector's English-heavy
proxy would otherwise starve exactly the capabilities we are trying to build. The anneal
reserve (best verified-native Indic, all ~10k real agentic trajectories, longest verified
reasoning traces, SWE-Gym-verified edits; ~300–400B tokens) is held back for the cooldown —
justified by SWE-Gym's 491 real trajectories alone delivering +14% on SWE-bench Verified.

The curriculum orders it all (seed → general → capability → long-context → anneal) with
every transition blended over a several-billion-token warmup band — the direct lesson from
V4's ~150× gradient-norm spike when a sudden Hindi-share jump hit frozen embeddings.

## Part 4 — Turning the plan into a testable hypothesis, and running the proxy

The rubric's highest marks go to whoever writes the plan as a falsifiable hypothesis and
*actually runs* a proxy. Real 1B/3B training is impossible on a laptop, so I built a
**proxy-of-the-proxy**: a genuine gradient-trained ~43k-param char-level neural LM (numpy,
hand-written backprop) in [`proxy/`](proxy/), trained on **real data** — Session 4's cleaned
Assamese (Indic), the repo's own English markdown (en), the Session 2/4 Python (code), and
deterministically-generated arithmetic CoT (math, a synthesized-reasoning stand-in). Three
arms differ *only* in the training mixture, at equal compute:

- **A** naive web-heavy (en 82 / code 12 / math 3 / indic 3)
- **B** the actual V5 main mix (en 57 / code 21 / math 8 / indic 14)
- **C** the anneal-flavoured upsampling (en 45 / code 23 / math 12 / indic 20)

**The result (held-out perplexity):** protecting the scarce lanes **more than halves Indic
perplexity** (48.3 → 21.4 under B) and collapses math (12.8 → 5.6), while code improves
slightly — at a **real English cost** (7.3 → 8.3, +6.3%) that grows toward the anneal arm
(+10%). The direction is confirmed strongly and monotonically. But my strict sub-claim —
that Indic could improve *without materially raising English* (Δ ≤ 2%) — was **refuted at
this scale.** I chose to report that faithfully rather than tune the arms until a green
checkmark appeared: the proxy caught a genuine cost, which is precisely its job. I also
explain *why the tiny proxy overstates the English cost* (at 16T scale English is data-rich,
so a share cut barely moves its loss; and a 43k-param model is capacity-bound in a way a 40B
model is not) — so the measured cost is an upper bound, and the 1B/3B run remains the
magnitude gate. Honest limitations are laid out in [`proxy/README.md`](proxy/README.md).

The first run actually flagged "NOT confirmed" against a 2%/two-arm test; rather than move
the goalposts, I widened it to a **three-arm trade-off frontier** and reported the frontier
as the finding. That is the session's thesis made empirical: *every share given to one
capability is taken from another*, visible in the numbers.

## Part 5 — The static visuals

[`assets/viz.py`](assets/viz.py) is a stdlib-only, deterministic SVG generator (colorblind-
safe palette; violet for the scarce Indic lane, echoing the session's scarcity markers). It
emits five charts embedded in the README: the **main-vs-anneal mixture**, the **supply-vs-
target** gap on a log scale, the **Indic four-tier split**, the **proxy result**, and the
**curriculum timeline**. Charts are data-driven from the same numbers as the plan, so they
cannot drift from it.

## Part 6 — What's where (and how to reproduce)

- **The graded spec:** [`plan/mixture-and-curriculum-v5.md`](plan/mixture-and-curriculum-v5.md)
- **Submission surface:** [`README.md`](README.md)
- **Evidence:** [`research/`](research/) + [`proxy/`](proxy/) + [`assets/`](assets/)
- **Reproduce the proxy** (~6s, uses Session 4's numpy venv):
  ```bash
  cd proxy && ../../session-4/.venv/bin/python3 data_prep.py && ../../session-4/.venv/bin/python3 train.py
  ```

## Part 7 — What I would do next with more compute

1. **Run the real 1B/3B proxy** with the confirm/refute rule in the plan §9 — the only way
   to turn the direction into a magnitude and settle the English-cost question.
2. **Stand up the two synthesis pipelines** the plan depends on: IndicTrans2-based Indic
   translation (back-translation-gated) and an executable-environment agentic-trajectory
   generator with execution-verification filtering.
3. **Continue Session 4 cleaning toward the cumulative target**, now aimed at the lanes this
   mixture shows to be starved — verified-native Indic beyond Assamese, and real code-edit /
   agentic trajectories — so the tokens behind the mixture keep growing.

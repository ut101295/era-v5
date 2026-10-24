# Benchmark → Training-Data Format & Loss-Shape Dossier
### For the "Bharat" data-mixture plan (India-first ~40B, coding + agentic, controllable reasoning, native Indic)

> Research dossier for Session 5. Each benchmark is read at the "loss-map" level —
> what tokens are trained (green) vs context-only (grey), and which training stage
> uses it — so that "capabilities we want to win" becomes a concrete list of
> training-data formats the mixture must contain. Web-researched with current
> (2026) figures; sources at the end.

## 0. How to read this dossier: the "loss map" framing

Every benchmark implicitly specifies a *training-data shape*. The bridge from "capability we want to win" to "data the mixture must contain" is answering three questions per benchmark:

1. **What is graded?** The specific span the scorer inspects (a patch, a JSON call, a boxed number, a final DB state, a rubric-scored essay).
2. **What is the loss map?** Which tokens carry gradient ("green" = loss applied = the target the model must learn to *produce*) versus which are read-only "grey" context (prompt, tool responses, retrieved docs, user turns). **Loss must fall only on tokens the model is responsible for emitting at eval time** — putting loss on tool outputs or user turns teaches the model to hallucinate the environment.
3. **Which stage?** Pretraining (next-token on raw corpora), SFT (supervised imitation of ideal completions), Preference (DPO/reward-model on chosen vs. rejected pairs), or RLVR (RL with verifiable rewards — a scalar reward on the outcome, frequently **no token-level target at all**).

A recurring pattern for agentic/reasoning benchmarks: they are **outcome-verified**, so the natural training signal is a *reward*, not a *label*. You can bootstrap them with SFT on trajectories (green on the agent's actions, grey on everything the environment returned), then finish with RLVR on the same verifier.

---

## 1. Agentic & Tool Use

### 1.1 SWE-bench Verified

**(a) What it measures.** Can an agent resolve a real GitHub issue in a real Python repo? 500 human-validated instances (filtered with OpenAI) across 12 mature OSS repos. It jointly stresses repo navigation, multi-file reasoning, and producing a working code edit. Note it dual-evaluates *model + harness*; the official leaderboard now standardizes on `mini-SWE-agent` (a bash-only ReAct loop) for apples-to-apples model comparison.

**(b) Scoring.** Strict, execution-based. A patch is "resolved" only if (i) the **fail-to-pass** tests (that failed on the buggy code) now pass, *and* (ii) all **pass-to-pass** tests still pass (no regressions). Each instance runs in an isolated Docker container with post-issue git history stripped. Metric = **resolved rate** (pass@1, sometimes pass@3). Frontier models are ~80% on Verified in 2026; the harder SWE-bench Pro sits ~23%.

**(c) Sample task.**
- *Input (grey context):* Issue text — e.g. *"`django.utils.numberformat.format()` drops the sign for very small negative decimals rendered in scientific notation."* — plus the checked-out repo tree and a bash shell.
- *Expected output (green):* a unified diff that makes the new regression test pass.

**(d) Loss map / data shape.**
- **Data format:** agentic trajectories = `(issue + repo state) → [bash action → observation]* → final patch`. Or, simpler, **code-edit pairs**: (buggy file region + issue) → diff.
- **Green (loss on):** the agent's *emitted* tokens only — the bash commands it issues and the final patch/diff.
- **Grey (context, no loss):** issue text, repo file contents, and crucially **every tool/shell observation** returned between actions (test logs, `grep` output, tracebacks). Applying loss to these teaches the model to fabricate command output.
- **Stage:** **SFT** on gold trajectories/diffs to bootstrap the edit format, then **RLVR** where the reward *is the test harness* (fail-to-pass ∧ pass-to-pass → reward 1). There is no single "correct" patch, so RL beats pure imitation. Mine your own repos + synthetic bug injection to avoid contamination.

### 1.2 BFCL (Berkeley Function-Calling Leaderboard, v4)

**(a) What it measures.** LLM ability to select the right function and format a syntactically/semantically valid call from a natural-language query + a set of API schemas. v3 added multi-turn/stateful interaction; v4 added holistic agentic evaluation, relevance/irrelevance detection ("abstain when no function fits"), and multi-language function pools mined from top-starred GitHub repos.

**(b) Scoring.** Two tracks. **AST accuracy** parses the generated call into an abstract syntax tree and checks function name, required params, and value/type matching against ground truth — *no execution needed*. **Execution accuracy** actually runs the call. Also reports relevance/irrelevance and multi-turn success. AST is brittle: naturalistic paraphrases drop exact-match 13–19 points.

**(c) Sample task.**
- *Input (grey):* User: *"What's the weather in Boston in Fahrenheit?"* + schema `get_current_weather(location: str, unit: enum['celsius','fahrenheit'])`.
- *Expected output (green):* exactly
  ```json
  {"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}
  ```

**(d) Loss map / data shape.**
- **Data format:** `(user query + tool schemas [+ prior turns]) → structured tool call(s)`.
- **Green (loss on):** the generated function-call JSON (name + arguments) — and, for abstention items, the refusal.
- **Grey (context):** the user query, the schema/tool definitions, and **tool return values** in multi-turn.
- **Stage:** primarily **SFT** — exact-format imitation of correct calls. Add a small **preference/RLVR** pass rewarding AST-valid calls and correct abstention. Highest-leverage lane for Bharat's tool-use goal because the target is crisp.

### 1.3 τ-bench / τ²-bench (Tool-Agent-User)

**(a) What it measures.** Sustained, *reliable* tool use inside a policy-constrained conversation. The agent talks to an **LLM-simulated user**, must elicit missing info, call domain APIs that read/write a stateful DB, and obey a written policy. Domains: retail, airline (harder), and in τ²: telecom, banking, voice.

**(b) Scoring.** **Outcome-based**: compare the final DB state against an annotated goal state. Signature metric **pass^k** = solved on *all* k independent trials (reliability, not luck). Even strong function-callers historically scored <50%, with pass^8 ~25% in retail — consistency is the hard part.

**(c) Sample task.**
- *Input (grey):* Policy doc ("*refunds to original payment method only; exchange within 30 days*"), tool set (`get_order`, `process_refund`, …), DB, and a simulated user: *"I want to return the shoes I bought."*
- *Expected behavior (green = agent turns):* agent asks for order ID, verifies eligibility, calls `process_refund(...)`, confirms. **Success = DB shows the correct refund**, regardless of wording.

**(d) Loss map / data shape.**
- **Green (loss on):** *only the agent's turns* — its NL replies and its tool calls.
- **Grey (context):** policy, tool schemas, **simulated-user utterances**, and **all tool/DB responses**.
- **Stage:** **SFT** on successful policy-compliant trajectories → **RLVR** with the DB-state verifier as reward. pass^k rewards *consistency*, so variance-reducing RL matters. Directly relevant to Bharat's agentic goal; build an **Indic-language τ-style domain** (see §4.5).

### 1.4 Terminal-Bench (v2/v2.1)

**(a) What it measures.** Hard, realistic command-line tasks — compiling code, training a small model, server config, sysadmin, debugging — through a real shell in an isolated Docker sandbox. v2.x = 89 human-audited tasks; frontier < 65%.

**(b) Scoring.** **End-state verification**: after the agent stops, a pytest-style suite asserts the environment reached the correct final state. Resolved only if *all* tests pass. Ships a canary GUID to detect leakage.

**(c) Sample task.**
- *Input (grey):* *"Set up a venv, install pinned deps, make `python train.py` produce `model.pkl` with test accuracy ≥ 0.9."* + a bash shell.
- *Expected output (green):* the shell commands the agent emits; verifier checks `model.pkl` exists and hits the threshold.

**(d) Loss map / data shape.** Green: emitted shell commands. Grey: instruction + all stdout/stderr. **Stage:** SFT on gold command trajectories → **RLVR** with the end-state suite as reward. Nearly identical loss shape to SWE-bench over a broader OS surface.

### 1.5 GAIA / WebArena (secondary / transfer)

**GAIA**: 466 human-verified general-assistant questions needing multi-step browsing/tools; exact-match answer. **WebArena**: 812 long-horizon tasks on self-hosted real sites; programmatic end-state success. Same loss shape as §1.3–1.4 (green on agent actions + final answer; grey on page content/observations). Treat as secondary transfer targets — the loss machinery is shared, so wins transfer.

---

## 2. Coding

### 2.1 HumanEval
Function-level Python synthesis from signature + docstring (164 problems). **Saturated** (>95% frontier) — a floor, not a differentiator; use HumanEval+ for real signal. **pass@1**, execution-based. Data: (signature+docstring)→body. Green: code body. Grey: prompt. **Stage:** SFT; light RLVR on unit tests.

### 2.2 MBPP
~974 entry-level Python tasks from NL + 3 asserts. Saturated basic-competence floor. pass@1. Data: NL instruction (+asserts) → full function. Green: function. Grey: instruction/asserts. **Stage:** SFT; optional RLVR.

### 2.3 LiveCodeBench (v6+)
**Contamination-free** competitive coding — problems scraped continuously with release dates, evaluated only *after* a model's cutoff. v6 ≈ 1055 problems. Also tests self-repair and test-output prediction. **pass@1** on the post-cutoff window. Data: problem→full solution; *also* self-repair pairs `(buggy + failing test) → fix`. Green: solution/repair. Grey: statement, sample I/O, test feedback. **Stage:** SFT on solutions + **RLVR** with the test executor as reward. Use *recent/self-generated* problems to beat the temporal cutoff.

---

## 3. Reasoning & Math

*General note:* this whole lane is where **long chain-of-thought + RLVR** pays off most. Answers are cheaply and deterministically verifiable (exact number / boxed value / MCQ letter): SFT on CoT to teach format → **RLVR (GRPO/DAPO-style)** with a binary correctness reward on the *final answer*. Standard prompt: *"reason step by step, put your final answer in \boxed{}."*

### 3.1 GSM8K
Grade-school multi-step arithmetic (1,319 test; 2–8 steps). Reasoning floor. Exact-match on final number. Data: problem → CoT + answer. Green: CoT + final number. Grey: problem. **Stage:** SFT (CoT) → RLVR (exact-match). Process/step-level rewards can beat outcome-only for smaller models — relevant at 40B for sample-efficiency.

### 3.2 MATH
12.5k competition problems, 5 difficulty levels; MATH-500 common for eval. Exact match on boxed answer with math-aware normalization. Data: problem → long-CoT + boxed answer. Green: CoT + boxed. Grey: problem. **Stage:** SFT (long-CoT) → RLVR. Strong target for **controllable reasoning** (budget-controlled CoT length).

### 3.3 AIME
Olympiad-level; integer answers 0–999. AIME 2025 = prized contamination-free set. **Only 30 problems/year → high variance**; report avg@k. **Stage:** flagship **RLVR** target (verifiable, high headroom, extended CoT). Best "controllable reasoning" testbed (thinking-budget scaling).

### 3.4 GPQA (Diamond)
Graduate-level "Google-proof" science MCQs by PhD experts; Diamond = hardest ~198. MCQ accuracy. Data: question+options → CoT + chosen letter. Green: CoT + choice. Grey: question, options. **Stage:** SFT (expert CoT) → RLVR. Win comes from broad high-quality STEM pretraining + reasoning SFT, not benchmark-fitting.

---

## 4. Indic

*Context:* the Indic landscape is mature for **understanding** and **generation/QA** but has a **real, flaggable gap** for **agentic/coding/tool-use in Indian languages** (§4.5) — a strategic opening for an India-first model.

### 4.1 MILU
India-centric knowledge + reasoning, 8 domains / 41 subjects, **11 languages**, built natively from 40+ Indian competitive exams (~80k MCQs). Culturally specific (local history, arts, law). GPT-4o ~74%. MCQ accuracy. Green: answer/CoT. Grey: question+options. **Stage:** driven mostly by **pretraining** on high-quality *native* Indic corpora (a knowledge benchmark), + light SFT on Indic MCQ+CoT. Requires substantial native (not translated) text across all 11 scripts.

### 4.2 IndicMMLU-Pro
MMLU-Pro adapted to **9 Indic languages** via IndicTrans2 + expert proofreading. Translation-based. Scores ~9–45% → large headroom. MCQ accuracy. **Stage:** pretraining + SFT (cross-lingual reasoning transfer). Signals need for **cross-lingual reasoning data** (English CoT ↔ Indic problem).

### 4.3 IndicGenBench
*Generation* across **29 Indic languages / 13 scripts**: CrossSum-IN (summarization), Flores-IN (MT both directions), XQuAD-IN (in-language QA), XorQA-IN (cross-lingual QA). Metrics: chrF++/BLEU/ROUGE, F1/EM. Data: parallel/aligned pairs → generated target. Green: generated target text. Grey: source, passage, question. **Stage:** SFT on parallel MT/summ/QA; optional preference (chrF++/F1 rewards). Demands large clean **parallel Indic↔English** + native summarization data.

### 4.4 IndicQA / Indic-QA
Context-grounded QA in **11 Indian languages** (AI4Bharat's manually-curated extractive set, and the larger Indic-QA mixing native/translated/synthetic). Span F1/EM (extractive), ROUGE/chrF or LLM-judge (abstractive). Data: (passage + question) → answer. Green: answer only. Grey: passage + question (**RAG-shaped**: given context is always grey). **Stage:** SFT on grounded QA; strengthens Indic faithfulness. Transfers to Indic RAG/agentic products.

### 4.5 Indic coding / agentic — **GAP (flag this)**

**There is essentially no mature, human-verified coding *or* agentic/tool-use benchmark in Indian languages.** A genuine gap and strategic opportunity. What exists is partial:
- **BFCL-Hi** — a Hindi machine-*translation* of BFCL (~2,251 items), **not human-verified**; preliminary.
- **AI4Bharat Indic LLM-Arena** — agentic tasks are on the **Phase-3 roadmap**, not yet deployed.
- **IndicDB** — Indic **Text-to-SQL** over Indian government datasets (En + Hi, Mr, Bn, Ta, Te + Hinglish) — closest Indic coding-adjacent agentic benchmark, but narrow (semantic parsing).
- **IndQA (OpenAI, Nov 2025)** — newest Bharat-relevant *reasoning* benchmark: 2,278 culturally grounded, reasoning-heavy questions across **12 languages** (incl. **Hinglish**), 10 domains, 261 Indian experts, **adversarially filtered** (kept only questions frontier models failed), **rubric-graded**. GPT-5 leads at only ~34.9% — huge headroom. Green: generated answer. Grey: prompt (rubric = grader artifact, never a target). **Stage:** SFT on expert answers → **preference / RLVR-with-rubric-grader**.

**Recommendation for the gap:** Bharat should *build and train toward* (i) an **Indic BFCL** (native, human-verified function-calling with Indic + Hinglish queries), (ii) an **Indic τ-style** policy-constrained tool-agent-user benchmark, and (iii) **Indic-instruction code generation** (issues/docstrings in Indian languages → code). Training data follows the *same loss maps* as English (green on agent's calls/code, grey on tool responses/user turns) with Indic/Hinglish prompts. This converts "native Indic" + "agentic" from two lanes into one differentiated capability.

---

## 5. Summary table: benchmark → capability lane → data format → loss shape → stage

| Benchmark | Capability lane | Required training-data format | Loss shape (green = loss / grey = context) | Primary stage(s) |
|---|---|---|---|---|
| **SWE-bench Verified** | Agentic coding | Repo + issue → bash-action trajectory → unified-diff patch | Green: agent's commands + final patch. Grey: issue, repo files, all shell/test output | SFT (trajectories) → **RLVR** (test harness = reward) |
| **BFCL v4** | Tool use / function calling | (query + tool schemas [+ turns]) → structured JSON call(s) / abstention | Green: emitted call JSON (name+args) or refusal. Grey: query, schemas, tool returns | **SFT** (exact-format) → light preference/RLVR |
| **τ-bench / τ²** | Agentic tool-agent-user | Policy + tools + simulated-user dialogue → agent turns + tool calls | Green: agent turns + tool calls only. Grey: policy, user utterances, tool/DB responses | SFT (rollouts) → **RLVR** (final DB state = reward) |
| **Terminal-Bench** | Agentic terminal | Instruction → shell-command trajectory | Green: emitted shell commands. Grey: instruction, all stdout/stderr | SFT → **RLVR** (end-state tests = reward) |
| **GAIA / WebArena** | Agentic web/assistant | Task → browser/tool actions → final answer | Green: agent actions + final answer. Grey: page content, tool observations | SFT (browsing traces) → **RLVR** |
| **HumanEval / MBPP** | Coding (floor) | (signature+docstring) or NL instruction → function body | Green: generated code. Grey: prompt/asserts | **SFT** (instr→code); optional RLVR |
| **LiveCodeBench** | Coding (contamination-free) | Contest problem → full solution; + (buggy+failing test)→fix | Green: solution/repair. Grey: statement, sample I/O, test feedback | SFT → **RLVR** (test executor); use *recent* problems |
| **GSM8K** | Reasoning (math, floor) | Word problem → CoT + numeric answer | Green: CoT + final number. Grey: problem | SFT (CoT) → **RLVR** (exact-match) |
| **MATH** | Reasoning (math) | Problem → long-CoT + \boxed{} answer | Green: CoT + boxed answer. Grey: problem | SFT (long-CoT) → **RLVR** |
| **AIME** | Reasoning (olympiad) | Problem → extended CoT + integer 0–999 | Green: CoT + integer. Grey: problem | **RLVR** flagship (extended CoT) |
| **GPQA-Diamond** | Reasoning (expert STEM) | Question + options → CoT + chosen letter | Green: CoT + choice. Grey: question, options | SFT (expert CoT) → **RLVR** |
| **MILU** | Indic understanding (native) | Indic exam MCQ → answer (+CoT) | Green: answer/CoT. Grey: question+options | **Pretraining** (native Indic) + SFT |
| **IndicMMLU-Pro** | Indic understanding (reasoning) | Translated MMLU-Pro MCQ (9 langs) → answer | Green: answer/CoT. Grey: question+options | **Pretraining** + SFT |
| **IndicGenBench** | Indic generation | Parallel pairs → translation/summary/QA span (29 langs) | Green: generated target text. Grey: source, passage, question | **SFT** (parallel); optional preference |
| **IndicQA / Indic-QA** | Indic grounded QA | (passage + question) → answer span/text (11 langs) | Green: answer only. Grey: passage + question (RAG-shaped) | **SFT** (grounded QA) |
| **IndQA** (Indic reasoning) | Indic cultural reasoning | Indic/Hinglish prompt → free-form answer + rubric | Green: generated answer. Grey: prompt (rubric = grader) | SFT → **preference / RLVR-with-rubric** |
| **Indic coding/agentic** | **GAP** | *(To build)* Indic/Hinglish query + schemas/policy/issue → calls/code/agent turns | Green: agent's calls/code. Grey: tool responses, user turns, policy | SFT → RLVR — **same maps as English, Indic prompts** |

---

## 6. Mixture take-aways for Bharat

1. **Two loss-map archetypes cover almost everything.** (a) *Verifiable-outcome agentic/coding/math*: green only on the model's own emitted actions/answer, grey on every environment/tool/user token, finished with **RLVR** on a programmatic verifier. (b) *Understanding/generation/QA*: green on the target text, grey on prompt/context, learned mostly by **SFT** (and, for Indic knowledge, **pretraining coverage**).
2. **RLVR is the shared engine** for coding + math + agentic — invest in verifiers (sandboxed test runners, stateful tool sims, boxed-answer/rubric graders) as first-class infra.
3. **Never put loss on grey tokens** (tool outputs, retrieved passages, simulated-user turns) — the single most important structural rule for agentic SFT.
4. **"Native Indic" + "agentic" is an unclaimed intersection.** Port English agentic/coding loss maps to Indic/Hinglish prompts, policies, issues. Building Indic BFCL / Indic τ-bench / Indic-instruction code corpora (and matching verifiers) is the concrete data move that turns positioning into a measurable lead.
5. **Controllable reasoning** maps onto MATH/AIME/GPQA data with thinking-budget-annotated CoT (train with variable-length/toggleable CoT, reward correct final answers regardless of visible reasoning length).

---

## Sources

- SWE-bench Verified — [swebench.com/verified](https://www.swebench.com/verified.html), [OpenAI](https://openai.com/index/introducing-swe-bench-verified/), [Epoch AI](https://epoch.ai/benchmarks/swe-bench-verified)
- BFCL — [Gorilla leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), [ICML 2025 (PMLR)](https://proceedings.mlr.press/v267/patil25a.html)
- τ-bench / τ²-bench — [arXiv 2406.12045](https://arxiv.org/abs/2406.12045), [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
- Terminal-Bench — [tbench.ai](https://www.tbench.ai/), [Artificial Analysis v2.1](https://artificialanalysis.ai/evaluations/terminalbench-v2-1)
- GAIA / WebArena — [GAIA OpenReview](https://openreview.net/forum?id=fibxvahvs3), [HAL GAIA leaderboard](https://hal.cs.princeton.edu/gaia)
- HumanEval / MBPP — [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/docs/README.md)
- LiveCodeBench — [livecodebench.github.io](https://livecodebench.github.io/), [arXiv 2403.07974](https://arxiv.org/html/2403.07974v2)
- Reasoning & RLVR — [Phi-4-reasoning report](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf), [awesome-RLVR](https://github.com/opendilab/awesome-RLVR)
- MILU — [arXiv 2411.02538](https://arxiv.org/abs/2411.02538), [AI4Bharat/MILU](https://github.com/AI4Bharat/MILU)
- IndicMMLU-Pro — [arXiv 2501.15747](https://arxiv.org/html/2501.15747v1)
- IndicGenBench — [arXiv 2404.16816](https://arxiv.org/abs/2404.16816)
- IndicQA — [ai4bharat/IndicQA (HF)](https://huggingface.co/datasets/ai4bharat/IndicQA)
- IndQA — [OpenAI: Introducing IndQA](https://openai.com/index/introducing-indqa/)

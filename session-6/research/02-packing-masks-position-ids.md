# Dossier 02 — Packing, Loss Masks, Attention Masks, and Position IDs

> **The question this dossier answers:** once documents are tokenized, how do we turn them
> into fixed-length training windows **without wasting compute on padding** and **without
> letting one document leak into another through attention** — and how do we mark exactly
> which token positions the model should *learn from*? This is Session 6 §3 (documents →
> sequences), §4 (padding), and §5 (packing policies), and it is the single largest graded
> block: *"Packing, masks and batch correctness — 150 pts."*

> Note on sourcing: the consolidated web dossier for this specific topic was lost to a
> mid-stream API error during research, so the citations here are the canonical primary
> sources I rely on from domain knowledge (FlashAttention repo, the two packing papers, HF
> docs, Megatron). They should be re-verified against the live pages before final submission,
> but the mechanisms are stable and widely documented.

---

## 1. Vocabulary (the shape of one training example)

- **token** — integer id from the frozen tokenizer (Session 2 contract).
- **sequence** — a fixed-length window (e.g. 4 096 / 8 192 / 32 768 tokens).
- **sample** — one training example. In pretraining, usually one sequence. In SFT/agentic,
  a structured record (prompt / response / tool-observation / mask / labels).
- **microbatch** → **global batch** → **optimizer step** (gradient accumulation across GPUs).
- **loss-bearing token** vs **padding** vs **context-only token** — the GPU computes over
  *every* position, but the model learns only where **loss is applied**. The true throughput
  metric is therefore **useful loss-bearing tokens/sec**, not raw tokens/sec (Dossier 06).

---

## 2. Padding, and why it is wasted budget

Training wants fixed shapes; text does not arrive in fixed shapes. Padding a 2 700-token
example up to 8 192 "solves the shape problem but creates a compute problem": those pad
positions "occupy memory, pass through parts of the model, and reduce the number of useful
tokens processed per second" (assignment §4). Even when the **loss mask** zeroes pad
positions, dense attention still spends memory bandwidth and FLOPs carrying them **unless the
kernel explicitly skips them** (see varlen attention, §4 below).

Kinds of padding to be aware of: **right** (after real tokens, common when batching short
examples), **left** (before real tokens, common in batched *inference*), **batch-level** (pad
to the longest example in the batch), **fixed-context** (pad every sample to full context),
and **within-pack holes** (unused tail after packing). At scale, all of these are burned
training budget.

Three student questions the assignment poses, with the correct answers:

1. **Can we cut a document mid-line?** For plain next-token pretraining, usually **yes** —
   the model learns local continuation across billions of spans. But careless cuts damage
   **code blocks, tables, proofs, agent trajectories, and instruction/response pairs** where
   structure carries meaning.
2. **Can we fill the tail with a different topic?** For plain pretraining, **yes**, with an
   **EOS boundary** between documents. For SFT / agentic / reasoning, unrelated samples in the
   same attention-visible window "can teach unnatural transitions unless masks and boundaries
   isolate them."
3. **Does the boundary matter if we only predict the next token?** **Yes** — without explicit
   boundaries "the model may learn that unrelated text is a natural continuation." Boundaries
   prevent accidental cross-document continuations.

---

## 3. Packing policies (the §5 taxonomy) and when each is safe

| Policy | Mechanism | Utilization | Safe for |
| --- | --- | --- | --- |
| **pad-only** | one sample per window, pad the rest | worst | anything (but wasteful) |
| **concatenate-and-chop** | join docs with EOS, cut fixed windows | ~100% | plain pretraining |
| **greedy (first-fit)** | drop each example into the first window with room | good | order-dependent; mixed data |
| **best-fit / bin-packing** | sort/bucket by length, place in tightest slot | best | many short examples |
| **structure-preserving** | packing rules that keep SFT/tool/agentic samples intact | good | SFT / agentic |
| **long-context** | handled separately; every unused slot is high-value waste | — | long-context stage |

Rules that fall out of it: web text tolerates concatenation; **code** tolerates spans but
benefits from preserving file/function boundaries; **agentic** trajectories must preserve the
order of observation→call→answer; **reasoning** traces need enough room to *finish the
argument*; and — the firewall rule — **evaluation shards must never be casually packed into
training** (Dossier 05).

**Two papers worth citing** for the "packing done right" claims:

- *Efficient Sequence Packing without Cross-Contamination* (Krell et al., 2021) — packing
  BERT-style sequences with a **block-diagonal attention mask** removes cross-example
  contamination while recovering the throughput that naive padding wastes.
  <https://arxiv.org/abs/2107.02027>
- *Fewer Truncations Improve Language Modeling* (Ding et al., 2024) — "best-fit packing"
  reduces harmful truncations and measurably improves downstream quality vs concatenate-and-chop.
  <https://arxiv.org/abs/2404.10830>

---

## 4. The three masks, done correctly

When multiple documents share one window, three things must be computed per position.

### (a) Loss mask — *where the model learns*

- **Pretraining:** loss on (almost) every next-token position; the only masked positions are
  padding and, by convention, positions after the final EOS.
- **SFT:** **loss only on the assistant response tokens**; the prompt is context (masked).
- **Agentic / tool-use:** loss only on the model's **planning, tool-call, and final-answer**
  tokens — **never** on tool observations / environment output / retrieved passages. Putting
  loss on these "grey" tokens teaches the model to **hallucinate the environment** (it learns
  to *predict* tool output instead of *calling* the tool). This is the single most important
  structural rule and echoes Session 5's loss-map work.

### (b) Attention mask — *what each token may look at*

Packing N documents into one window requires a **block-diagonal (document-level) causal
mask**: position `i` may attend to position `j` **only if** `j ≤ i` **and** `doc_id[i] ==
doc_id[j]`. Without it, the model attends across the EOS into an unrelated document and learns
false continuations. Two implementation routes:

- **Dense boolean mask** (toy scale, what we will build): materialize an `L×L` mask that is
  the elementwise AND of the causal lower-triangle and the same-document indicator. Trivial to
  verify, `O(L²)` memory — fine for small `L`.
- **Varlen / "unpadded" attention** (production): FlashAttention's `flash_attn_varlen_*`
  takes **`cu_seqlens`** (cumulative sequence lengths — the document boundaries within the
  packed buffer) and computes attention **per segment**, never crossing boundaries and never
  spending FLOPs on padding. This is how packing is made *both* correct *and* fast.
  <https://github.com/Dao-AILab/flash-attention> · HF "attention mask / packing" docs:
  <https://huggingface.co/docs/transformers/en/glossary#attention-mask> and the
  `DataCollatorForCompletionOnlyLM` / packing utilities in TRL.

### (c) Position IDs — *where each token sits*

Position ids must **reset to 0 at each document boundary** inside a packed window, so
document 2 starts at position 0, not at position `len(doc1)`. With RoPE, position ids drive
the rotary phase; if they don't reset, the second document is rotated as if it were a
continuation of the first, corrupting its positional signal. `cu_seqlens`-based kernels reset
positions implicitly per segment; a dense implementation builds `position_ids` explicitly by
restarting the counter at every boundary.

---

## 5. Worked micro-example (what our packer/mask builder must produce)

Pack two documents `A=[a0,a1,a2,EOS]` and `B=[b0,b1,EOS]` into a window of length 8
(one pad at the end), pretraining mode:

```
tokens        : a0 a1 a2 EOS b0 b1 EOS PAD
doc_id        :  0  0  0   0  1  1   1  -1
position_ids  :  0  1  2   3  0  1   2   0     # reset at boundary
loss_mask     :  1  1  1   1  1  1   1   0     # (SFT would zero the prompt span)
attention_mask (row i can see col j):
              a0 a1 a2 E0 b0 b1 E1 P
        a0  [  1  0  0  0  0  0  0 0 ]
        a1  [  1  1  0  0  0  0  0 0 ]
        a2  [  1  1  1  0  0  0  0 0 ]
        E0  [  1  1  1  1  0  0  0 0 ]
        b0  [  0  0  0  0  1  0  0 0 ]   # cannot see doc A
        b1  [  0  0  0  0  1  1  0 0 ]
        E1  [  0  0  0  0  1  1  1 0 ]
        P   [  0  0  0  0  0  0  0 0 ]   # padding attends to nothing, learns nothing
```

Correctness invariants our tests assert:
- No attention edge crosses a `doc_id` boundary (block-diagonal ✔).
- `position_ids` restart at every boundary ✔.
- `loss_mask` is 0 on padding and (in SFT mode) on prompt tokens ✔.
- **Utilization** = loss-bearing tokens / window size = 7/8 here; reported per batch.

---

## 6. Minimal implementation recipe (what we will build)

1. **Packer** with switchable policy: `concat_chop` (pretraining), `best_fit` (SFT/short),
   `structure_preserving` (agentic — never split a trajectory, never reorder turns),
   `long_context` (dedicated stage). Deterministic cursor so the same shard always packs the
   same way (feeds Dossier 01's replay guarantee).
2. **Mask builder** returning `{input_ids, labels(=loss-masked), attention_mask(block-diag
   causal), position_ids(per-doc reset), doc_ids, lane}` for a window.
3. **Mode switch** `pretrain | sft | agentic` that changes *only* the loss mask (and the
   structure-preserving rule), making the "shape of the training example" visible per §3.
4. **Reported metrics per batch:** utilization %, wasted (pad) positions, boundary count,
   loss-bearing token count — all reconstructable from the saved batch (graded: numbers must
   be reproducible, §Evaluation).
5. **Every packed batch is content-hashed** (`batch_hash` over input_ids+labels+masks+
   position_ids) so resume/replay can prove byte-for-byte identity (Dossier 01/04).

---

## 7. Sources

- *Efficient Sequence Packing without Cross-Contamination*, Krell et al. — <https://arxiv.org/abs/2107.02027>
- *Fewer Truncations Improve Language Modeling*, Ding et al. — <https://arxiv.org/abs/2404.10830>
- FlashAttention (varlen / `cu_seqlens`) — <https://github.com/Dao-AILab/flash-attention>
- HF attention-mask glossary — <https://huggingface.co/docs/transformers/en/glossary#attention-mask>
- Megatron-Core datasets (concat-and-chop baseline) — <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/readme.md>

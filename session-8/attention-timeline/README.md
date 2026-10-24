# The Evolution of Attention — a chronological timeline

An interactive, single-file web app that lays out **every attention mechanism from
Session 8 in the order it was actually launched**, and explains each one as the
answer to a problem that existed at that moment. Zero build, zero dependencies —
one `index.html` you can open locally or drag onto Netlify.

**Live link:** https://era-v5-s8-attn-timeline.netlify.app

## Why chronological

Grouped by family, these mechanisms look like a menu. Sorted by date, they look
like an argument the field is having with itself: it wants **exactness** (cut
compute), then it wants **memory back** (shrink the KV cache), then it wants
**length** (fix position encoding), then it wants **memory back again** (throw out
the quadratic core for a recurrent state). You can't see that oscillation from a
list — only from a date axis. That's the whole point of the assignment, and the
app's closing section ("What the timeline shows") spells out the four things the
ordering reveals.

Each card gives: the **problem at that moment**, what the mechanism **buys you**,
what it **costs you** (honestly — every one is a trade), and **when you'd actually
pick it**. Color-codes by conceptual family; every card also carries a text tag, so
identity never rests on color alone (accessibility).

## Run it

```bash
cd session-8/attention-timeline
python3 -m http.server 8000     # open http://localhost:8000/index.html
```
Or just double-click `index.html` — it has no external requests, so `file://` works
too. Deploy = drag this folder onto [Netlify Drop](https://app.netlify.com/drop).

## Date sources

**Every date is the first public appearance** — the arXiv **v1** submission date
(verified against each paper's submission history), except NTK-aware scaling, which
first appeared as a community Reddit post and has no paper. Verified October 2026.

| # | Date | Mechanism | Source (primary) |
|---|------|-----------|------------------|
| 1 | 2017-06-12 | Scaled dot-product attention; sinusoidal + learned-absolute position | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 2 | 2019-04-23 | Sparse Transformers (strided/fixed) | [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) |
| 3 | 2019-11-06 | Multi-Query Attention (MQA) | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| 4 | 2019-12-25 | Explicit Sparse Transformer (top-k) | [arXiv:1912.11637](https://arxiv.org/abs/1912.11637) |
| 5 | 2020-04-10 | Longformer (sliding window) | [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) |
| 6 | 2020-06-29 | Linear attention (Transformers are RNNs) | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |
| 7 | 2021-02-22 | Delta rule / Fast Weight Programmers | [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) |
| 8 | 2021-04-20 | RoPE (RoFormer) | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| 9 | 2021-08-27 | ALiBi | [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) |
| 10 | 2023-05-22 | Grouped-Query Attention (GQA) | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| 11 | 2023-06-28 | NTK-aware scaled RoPE | [r/LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) |
| 12 | 2023-08-31 | YaRN | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| 13 | 2023-09-29 | StreamingLLM / attention sinks | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| 14 | 2024-05-07 | Multi-head Latent Attention (MLA, DeepSeek-V2) | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| 15 | 2024-06-10 | DeltaNet (parallelized delta rule) | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) |
| 16 | 2024-12-09 | Gated DeltaNet | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| 17 | 2025-02-16 | Native Sparse Attention (NSA, DeepSeek) | [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) |
| 18 | 2025-12-13 | DroPE (dropping positional embeddings, Sakana AI) | [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) |

### Notes & caveats on the dates

- **Sinusoidal, learned-absolute, and vanilla attention share one date** (2017-06-12)
  because they all debuted in *Attention Is All You Need*. They're shown as three
  separate cards because the assignment asks for them separately.
- **NTK-aware scaling (#11)** has no paper — it originated as a June 2023
  r/LocalLLaMA post by user *bloc97*. The date is the post date; it's the one entry
  not backed by an arXiv v1, and it's flagged as community-sourced in the app.
- **NTK date precision:** community post, late June 2023; we cite 2023-06-28. If a
  grader needs an exact day, the Reddit thread timestamp is the authority.
- **DroPE vs DRoPE — the trap.** There are *two* different papers:
  - **DroPE** (Sakana AI, "dropping positional embeddings", [arXiv:2512.12167](https://arxiv.org/abs/2512.12167), Dec 2025) — **this is the one in the timeline**, because it continues the positional/long-context thread the session is about.
  - **DRoPE** (directional RoPE for autonomous-driving agents, [arXiv:2503.15029](https://arxiv.org/abs/2503.15029), Mar 2025) — unrelated; a common confusion the instructor explicitly warned about.
- **Longformer** is the sliding-window citation; sliding window later became a
  production default in Mistral 7B (Sep 2023), noted on the card.

## Q2 — a mechanism not covered (with sourced date)

The headline omission is **FlashAttention** — "Fast and Memory-Efficient *Exact*
Attention with IO-Awareness," Dao et al., [arXiv:2205.14135](https://arxiv.org/abs/2205.14135),
**v1 2022-05-27**. It belongs on the timeline because it's the odd one out: every
other 2019-2022 entry pays the bill by *approximating* attention (sparse, linear,
windowed), while FlashAttention keeps attention mathematically exact and instead
attacks the *memory-IO* cost by tiling the computation to stay in SRAM. It's the
reason training long-context exact attention stayed viable at all — much of the
later long-context work assumes it underneath.

Two further not-covered mechanisms, both verified:
- **Ring Attention** — "…with Blockwise Transformers for Near-Infinite Context,"
  [arXiv:2310.01889](https://arxiv.org/abs/2310.01889), **2023-10-03**: shards the
  sequence across devices so context length scales with the device count.
- **Infini-attention** — "Leave No Context Behind,"
  [arXiv:2404.07143](https://arxiv.org/abs/2404.07143), **2024-04-10**: adds a
  compressive memory to a vanilla block for bounded-memory infinite context.

(See `../session-notes.md` for the full write-up of what the timeline shows.)

## Accessibility & design

Built per the project's dataviz method: categorical hues assigned in fixed palette
order (slots 5,1,2,3,4), **validated with the skill's checker in both light and dark
modes** — all CVD / normal-vision / lightness / chroma gates pass. Three light-mode
hues sit below 3:1 surface contrast, so the **relief rule** applies: every card and
legend chip carries a visible text label, never color alone. Light/dark toggle plus
`color-scheme` for native controls.

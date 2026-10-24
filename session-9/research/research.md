# Session 9 — Research notes

Background facts I relied on, with the reasoning so the harness is defensible.

## Cross-entropy for next-token prediction
For a target token `y` and logits `z`, the per-token loss is
`-log softmax(z)[y] = -z[y] + logsumexp(z)`. Averaged over contributing tokens it is the
scalar the optimiser minimises. PyTorch's `F.cross_entropy(logits, targets)` does exactly this
(log-softmax + NLL fused, numerically stable). The targets are the **next** token, so a model
seeing positions `0..T-1` predicts positions `1..T` — the "shift".

## The shift, and why strings matter
`logits[:, :-1]` are predictions *for* `tokens[:, 1:]`. If you instead align `logits[:, 1:]`
with `tokens[:, :-1]` (shift the wrong way), the model is trained to predict the token it just
saw — trivially learnable, so the loss still falls beautifully. The only robust check is to
decode both columns to strings and read them. Hence requirement 2's assert
`target[i] == input[i+1]` over decoded words.

## Padding and `ignore_index`
When batching variable-length sequences you pad to the max length. Those `<pad>` targets carry
no signal; if included, the mean loss is diluted and gradient leaks into padding. `ignore_index=
PAD_ID` drops them from both numerator and denominator, so the contributing-token **count**
visibly changes (10 → 7 here). Watching the count is the proof the mask took effect.

## Document packing and boundary masking
To avoid wasting positions, corpora are packed: `docA <eos> docB …` in one sequence. But the
target that predicts docB's first token *from docA's `<eos>`* is a cross-document guess with no
learnable signal. Leaving it in adds a large, irreducible term to the loss. On a trained model
that single boundary term is ~9.6 nats here (vs ~5 within-doc), so masking it (ignore_index on
the boundary target) measurably lowers the mean. Real pipelines either mask boundaries or reset
attention at document edges.

## Perplexity ≈ vocab at init
Perplexity = `exp(CE)`. A model that knows nothing should assign ~uniform probability `1/V`, so
CE ≈ `ln(V)` and PPL ≈ `V`. This is the cheapest end-to-end sanity check that the whole pipeline
(shapes, shift, softmax, averaging) is wired right. **Caveat:** it only holds if the untrained
model is actually near-uniform. Default `nn.Embedding` init is N(0,1); tying it to the output
head produces large logits → over-confident → PPL explodes. Initialising embeddings/head with
small std (0.02, as GPT-2 does) restores near-uniformity. So this check doubles as an init check.

## Weight tying
Tying the input embedding and output projection (`head.weight = tok.weight`, Press & Wolf 2016;
used in GPT-2) saves `V × D` parameters and often improves quality by sharing the token
geometry. Here that's `14 × 64 = 896` parameters: tied 105,088 vs untied 105,984.

## Chunked cross-entropy (memory)
The peak-memory villain in LM training is the `[N, V]` logits/softmax activation (N = tokens in
the batch, V = vocab). For large V it dwarfs the model. Computing the loss in row-chunks — logits
and log-softmax for `N/k` rows at a time, accumulating the summed loss — makes the peak logits
tensor `k×` smaller for an *identical* result. This is the idea behind fused/chunked CE kernels
(e.g. Liger-Kernel's fused linear-cross-entropy). Here V=14 so the absolute saving is tiny, but
the ratio (16×) demonstrates the mechanism; the win scales with V≈50k in real models.

## Multi-token prediction (Part 2)
Predicting several future tokens from one hidden state (Gloeckle et al. 2024, "Better & Faster
LLMs via Multi-token Prediction"; also the DeepSeek-V3 MTP objective) uses extra heads for t+1,
t+2, …. The near head is always at least as easy as the far head: on sequential data the recent
token pins the next one, while farther tokens require marginalising over the unknown tokens
between, which raises the irreducible entropy. On a first-order Markov chain this is exact — the
reason the harness uses that corpus so loss2 > loss1 holds robustly and matches the explanation.
Multi-token heads also power **speculative decoding**, where the near prediction is trusted more.

## Sources (concepts)
- Press & Wolf 2016, *Using the Output Embedding to Improve Language Models* (weight tying).
- Radford et al. 2019, GPT-2 (small-std init; tied head).
- Gloeckle et al. 2024, *Better & Faster LLMs via Multi-token Prediction*.
- DeepSeek-V3 technical report (MTP objective).
- Liger-Kernel / fused-linear-cross-entropy (chunked CE for peak-memory reduction).

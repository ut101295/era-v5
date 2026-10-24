"""
Session 9 — the loss harness, made correct and observable.

Script form, used to lock in the seven Part-1 numbers and the two Part-2 losses
before they go into the notebook. Every section is numbered to match the
assignment. Run top to bottom:

    python harness.py

The whole point: the few lines between model output and the scalar loss are where
silent bugs live. So we PRINT THE STRINGS, not the ids.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device = {DEVICE}\n")

# ---------------------------------------------------------------------------
# 0. A tiny, self-contained word-level tokenizer.
#    Word-level on purpose: when we print inputs beside targets in step 2, the
#    shift must be obvious to a human. Whole words make the off-by-one visible;
#    a wall of subword ids does not.
# ---------------------------------------------------------------------------
PAD, EOS = "<pad>", "<eos>"

WORDS = ["the", "cat", "dog", "fox", "ran", "sat", "jumped",
         "over", "under", "near", "log", "mat"]      # 12 content words
itos = [PAD, EOS] + WORDS
stoi = {w: i for i, w in enumerate(itos)}
VOCAB = len(itos)                                     # 14
PAD_ID, EOS_ID = stoi[PAD], stoi[EOS]

# Training data is drawn from a PEAKED FIRST-ORDER MARKOV CHAIN over the content
# words. This matters for Part 2: because each word strongly predicts the *next*
# one (order-1 dependence), knowing x_t pins x_{t+1} tightly but leaves x_{t+2}
# much less certain (it must marginalise over the unknown x_{t+1}). That is the
# real reason a t+2 head sits above a t+1 head — and it only shows up when the
# data actually has sequential dependence, not in a bag of independent slots.
_rng = random.Random(SEED)
_succ = {w: _rng.sample(range(len(WORDS)), 2) for w in range(len(WORDS))}  # 2 favoured next-words each


def _gen_sentence(length=11):
    cur = _rng.randrange(len(WORDS))
    out = [cur]
    for _ in range(length - 1):
        cur = _rng.choice(_succ[cur]) if _rng.random() < 0.9 else _rng.randrange(len(WORDS))
        out.append(cur)
    return [WORDS[i] for i in out]


CORPUS = [" ".join(_gen_sentence()) for _ in range(400)]  # 400 Markov sentences, seeded


def encode(text):
    return [stoi[w] for w in text.split()]


def decode_ids(ids):
    return [itos[i] for i in ids]


def pad_batch(rows, device=DEVICE):
    ml = max(len(r) for r in rows)
    return torch.tensor([r + [PAD_ID] * (ml - len(r)) for r in rows], device=device)


print(f"vocab size = {VOCAB}  (incl. {PAD!r}={PAD_ID}, {EOS!r}={EOS_ID})\n")


# ---------------------------------------------------------------------------
# A tiny causal transformer. Architecture is deliberately minimal — the
# assignment is about the LOSS, not the model. It just has to produce logits.
#
# NOTE the small-std init. A randomly-initialised LM should sit at loss ~ ln(V)
# (uniform guessing). That only happens if the output logits start NEAR ZERO.
# Default nn.Embedding init is N(0,1); tied to the head that makes logits huge
# and the untrained model wildly over-confident (loss >> ln V). GPT-2-style
# std=0.02 init is what puts an untrained model at ~ln(V). This is itself one of
# the correctness checks in step 5.
# ---------------------------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab, d_model=64, n_heads=4, n_layers=2, max_T=64, tie=True):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_T, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self._init_small()
        if tie:
            self.head.weight = self.tok.weight  # weight tying (after init)
        self.max_T = max_T

    def _init_small(self):
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        h = self.tok(tokens) + self.pos(pos)[None]
        causal = torch.triu(torch.ones(T, T, device=tokens.device, dtype=torch.bool), 1)
        h = self.blocks(h, mask=causal)
        return self.norm(h)  # [B, T, d_model]  (hidden states, pre-head)


def quick_train(model, data, steps=300, lr=3e-3):
    """Train next-token prediction with padding masked. Returns the model."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        lg = model.head(model(data))[:, :-1].reshape(-1, VOCAB)
        tg = data[:, 1:].reshape(-1)
        loss = F.cross_entropy(lg, tg, ignore_index=PAD_ID)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


torch.manual_seed(SEED)
model = TinyLM(VOCAB, tie=True).to(DEVICE)
model.eval()
output_head = model.head  # the assignment's `output_head`


# ===========================================================================
# PART 1
# ===========================================================================

# --- (1) Print every tensor shape, with the meaning of each dimension -------
print("=" * 70)
print("(1) SHAPES — every tensor and what each dimension means")
print("=" * 70)

batch = ["the cat ran over the log", "the dog sat near the mat"]
tokens = pad_batch([encode(s) for s in batch])  # equal length here
print(f"tokens        {tuple(tokens.shape)}   [B=batch, T=sequence length]")

hidden = model(tokens)
print(f"hidden        {tuple(hidden.shape)}   [B, T, d_model=hidden width]")

logits = output_head(hidden)
print(f"logits        {tuple(logits.shape)}   [B, T, V=vocab; score per next-token]")

shift_logits = logits[:, :-1]
shift_targets = tokens[:, 1:]
print(f"shift_logits  {tuple(shift_logits.shape)}   [B, T-1, V] — drop last position (no target)")
print(f"shift_targets {tuple(shift_targets.shape)}   [B, T-1]   — drop first token (nothing predicts it)")

flat_logits = shift_logits.reshape(-1, VOCAB)
flat_targets = shift_targets.reshape(-1)
print(f"flat_logits   {tuple(flat_logits.shape)}  [B*(T-1), V] — flattened for cross_entropy")
print(f"flat_targets  {tuple(flat_targets.shape)}      [B*(T-1)]   — flattened targets\n")


# --- (2) Verify the shift by printing STRINGS, inputs beside targets --------
print("=" * 70)
print("(2) SHIFT CHECK — input word → the word the model must predict")
print("=" * 70)
print("If this reads as a sensible 'next word' column, the shift is correct.\n")
for b, s in enumerate(batch):
    print(f"  sequence {b}: {s!r}")
    ins = decode_ids(tokens[b, :-1].tolist())
    tgs = decode_ids(tokens[b, 1:].tolist())
    for i, t in zip(ins, tgs):
        print(f"      input {i!r:>8}   ->   target {t!r}")
    print()
assert torch.equal(shift_targets, tokens[:, 1:]), "target must be input shifted left by one"
print("  assert passed: target[i] == input[i+1]  (the shift is a +1 slice)\n")


# --- (3) Mask padding and confirm the contributing-token count changes ------
print("=" * 70)
print("(3) PADDING MASK — contributing-token count must drop when we ignore <pad>")
print("=" * 70)

padded = pad_batch([encode("the cat ran over the log"), encode("the dog sat")])
print(f"padded batch shape {tuple(padded.shape)}:")
for row in padded.tolist():
    print(f"    {decode_ids(row)}")

h = model(padded)
lg = output_head(h)[:, :-1].reshape(-1, VOCAB)
tg = padded[:, 1:].reshape(-1)

loss_nomask = F.cross_entropy(lg, tg)                        # counts <pad> targets
loss_masked = F.cross_entropy(lg, tg, ignore_index=PAD_ID)  # ignores <pad> targets
n_all = tg.numel()
n_real = (tg != PAD_ID).sum().item()
print(f"\n  contributing tokens  no-mask: {n_all}   masked: {n_real}  "
      f"(dropped {n_all - n_real} <pad> targets)")
print(f"  loss  no-mask: {loss_nomask.item():.4f}   masked: {loss_masked.item():.4f}")
print("  -> the mean is now taken over real tokens only; <pad> no longer dilutes it.\n")


# --- (4) Pack two documents, mask the boundary (on a TRAINED model) ----------
print("=" * 70)
print("(4) DOCUMENT PACKING — mask the cross-document boundary target")
print("=" * 70)
# Use a trained model so the numbers tell the real story: a trained LM predicts
# WITHIN a document confidently (low loss), but 'what document comes next' after
# <eos> is genuinely unpredictable — that boundary term is spuriously high and we
# must not train on it. (On an untrained model every term is ~ln V and the gap
# would be invisible, which is exactly why we train first here.)
torch.manual_seed(SEED)
trained = quick_train(TinyLM(VOCAB, tie=True).to(DEVICE), pad_batch([encode(s) + [EOS_ID] for s in CORPUS]))

docA, docB = encode("the cat ran over the log"), encode("the fox jumped under the mat")
packed = torch.tensor([docA + [EOS_ID] + docB], device=DEVICE)
print(f"  packed: {decode_ids(packed[0].tolist())}")

with torch.no_grad():
    lg = trained.head(trained(packed))[:, :-1].reshape(-1, VOCAB)
tg = packed[:, 1:].clone()
boundary = len(docA)  # index in the shifted targets where docB[0] is predicted from <eos>
print(f"  boundary target = predict {itos[tg[0, boundary].item()]!r} from <eos> "
      f"(cross-document — meaningless)")

loss_before = F.cross_entropy(lg, tg.reshape(-1))
tg_masked = tg.clone()
tg_masked[0, boundary] = PAD_ID
loss_after = F.cross_entropy(lg, tg_masked.reshape(-1), ignore_index=PAD_ID)
boundary_term = F.cross_entropy(lg[boundary:boundary + 1], tg[:, boundary])
print(f"\n  loss before masking boundary: {loss_before.item():.4f}")
print(f"  loss after  masking boundary: {loss_after.item():.4f}")
print(f"  (the single boundary term alone = {boundary_term.item():.4f} — far above the rest)")
print("  -> masking removes a large, unlearnable cross-document term from the mean.\n")


# --- (5) Perplexity: untrained model sits near vocab size -------------------
print("=" * 70)
print("(5) PERPLEXITY — an untrained model should sit near |vocab|")
print("=" * 70)
torch.manual_seed(SEED + 1)
fresh = TinyLM(VOCAB, tie=True).to(DEVICE)
fresh.eval()
with torch.no_grad():
    allseq = pad_batch([encode(s) + [EOS_ID] for s in CORPUS[:32]])
    lg = fresh.head(fresh(allseq))[:, :-1].reshape(-1, VOCAB)
    tg = allseq[:, 1:].reshape(-1)
    ce = F.cross_entropy(lg, tg, ignore_index=PAD_ID)
ppl = math.exp(ce.item())
print(f"  untrained cross-entropy = {ce.item():.4f}")
print(f"  untrained perplexity    = {ppl:.2f}   (vocab size = {VOCAB})")
print(f"  ln(vocab) = {math.log(VOCAB):.4f}   exp(ln vocab) = {VOCAB}  <- target")
print("  -> PPL ~ |vocab| means the model is guessing ~uniformly: the harness is sane.\n")


# --- (6) Tied vs untied head parameter counts -------------------------------
print("=" * 70)
print("(6) TIED vs UNTIED output head — parameter counts on THIS config")
print("=" * 70)
tied_model = TinyLM(VOCAB, tie=True)
untied_model = TinyLM(VOCAB, tie=False)
tied_n = sum(p.numel() for p in tied_model.parameters())
untied_n = sum(p.numel() for p in untied_model.parameters())
head_n = untied_model.head.weight.numel()
print(f"  separate head weight = d_model x vocab = {head_n}")
print(f"  total params  tied:   {tied_n:,}")
print(f"  total params  untied: {untied_n:,}")
print(f"  untied costs {untied_n - tied_n:,} extra params "
      f"(one {tuple(untied_model.head.weight.shape)} matrix)")
print("  -> tying reuses the embedding matrix as the head: zero extra params.\n")


# --- (7) Peak memory: ordinary CE vs a hand-written chunked CE ---------------
print("=" * 70)
print("(7) MEMORY — ordinary cross-entropy vs chunked cross-entropy")
print("=" * 70)
# The memory villain is the full logits activation [N, V]. Chunking never
# materialises all of it at once. We report the largest logits tensor each path
# holds (exact, device-independent) and, on CUDA, the real measured peak.
N = 4096
hid = torch.randn(N, 64, device=DEVICE)
W = torch.randn(64, VOCAB, device=DEVICE)
tgt = torch.randint(0, VOCAB, (N,), device=DEVICE)


def ordinary_ce(hid, W, tgt):
    logits = hid @ W                      # [N, V] <- full materialisation
    return F.cross_entropy(logits, tgt), logits.element_size() * logits.nelement()


def chunked_ce(hid, W, tgt, chunks=16):
    size = math.ceil(hid.shape[0] / chunks)
    total, count, peak = 0.0, 0, 0
    for i in range(0, hid.shape[0], size):
        lg = hid[i:i + size] @ W          # [chunk, V] <- only one chunk alive
        peak = max(peak, lg.element_size() * lg.nelement())
        total += F.cross_entropy(lg, tgt[i:i + size], reduction="sum").item()
        count += lg.shape[0]
    return total / count, peak


def measure(fn):
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        out = fn()
        return out, torch.cuda.max_memory_allocated()
    return fn(), None


(loss_o, bytes_o), peak_o = measure(lambda: ordinary_ce(hid, W, tgt))
(loss_c, bytes_c), peak_c = measure(lambda: chunked_ce(hid, W, tgt, chunks=16))
print(f"  N (rows) = {N}, V = {VOCAB}")
print(f"  ordinary: largest logits tensor = {bytes_o/1e6:.3f} MB   loss={loss_o:.4f}")
print(f"  chunked : largest logits tensor = {bytes_c/1e6:.3f} MB   loss={loss_c:.4f}  (16 chunks)")
print(f"  logits-activation peak ratio = {bytes_o / bytes_c:.1f}x smaller")
if peak_o is not None:
    print(f"  [CUDA measured peak] ordinary {peak_o/1e6:.1f} MB vs chunked {peak_c/1e6:.1f} MB")
print("  -> identical loss, but chunked never holds the full [N,V] logits at once.")
print("     (The real-world win is largest with a big vocab, e.g. V=50k.)\n")


# ===========================================================================
# PART 2 — a second output head predicting token t+2
# ===========================================================================
print("=" * 70)
print("PART 2 — second head predicts t+2; watch its loss vs the t+1 head")
print("=" * 70)


class TwoHeadLM(TinyLM):
    def __init__(self, vocab, **kw):
        super().__init__(vocab, **kw)
        self.head2 = nn.Linear(self.tok.embedding_dim, vocab, bias=False)  # untied
        nn.init.normal_(self.head2.weight, std=0.02)


torch.manual_seed(SEED)
m2 = TwoHeadLM(VOCAB, tie=True).to(DEVICE)
opt = torch.optim.AdamW(m2.parameters(), lr=3e-3)
data = pad_batch([encode(s) + [EOS_ID] for s in CORPUS])


def two_head_losses(model, tokens):
    h = model.forward(tokens)
    l1 = model.head(h)[:, :-1].reshape(-1, VOCAB)   # predict t+1
    l2 = model.head2(h)[:, :-2].reshape(-1, VOCAB)  # predict t+2
    t1 = tokens[:, 1:].reshape(-1)
    t2 = tokens[:, 2:].reshape(-1)
    loss1 = F.cross_entropy(l1, t1, ignore_index=PAD_ID)
    loss2 = F.cross_entropy(l2, t2, ignore_index=PAD_ID)
    return loss1, loss2


print(f"  {'step':>5} {'loss1(t+1)':>12} {'loss2(t+2)':>12} {'sum':>10}")
m2.train()
for step in range(301):
    loss1, loss2 = two_head_losses(m2, data)
    loss = loss1 + loss2
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"  {step:>5} {loss1.item():>12.4f} {loss2.item():>12.4f} {loss.item():>10.4f}")

print("\n  What happens: both losses fall together, but loss2 (t+2) stays clearly ABOVE")
print("  loss1 (t+1), and the gap widens as training sharpens the one-step predictor.")
print("  Why: the data is an order-1 chain, so the last token x_t strongly pins x_{t+1}")
print("  (low loss1), but x_{t+2} can only be reached by marginalising over the still-")
print("  unknown x_{t+1}, which is more spread out (higher loss2). The far head converges")
print("  to a higher floor = the extra irreducible uncertainty of the longer jump. This")
print("  is the same reason speculative-decoding / multi-token heads trust near tokens")
print("  more than far ones. (It's the robust tendency for sequential data, not a")
print("  theorem for every possible distribution.)\n")
print("done.")

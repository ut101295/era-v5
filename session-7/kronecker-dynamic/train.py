"""
Proof 3 & 4 for Problem #3 — train a small causal Transformer and show that the
32-byte wall makes long words UNLEARNABLE, while the functional-position codec
makes them learnable.

The task
--------
Each "word" is stem + suffix, drawn from inflectional families across scripts and
stem lengths. The model reads a 3-word sequence and must predict the SUFFIX CLASS
(0..7) of the FIRST word, read out at the last position (so attention has to carry
the information forward). The suffix is what distinguishes members of a family, and
for long words that distinguishing byte lives PAST position 32.

Why this is a fair proof
------------------------
Everything except the input codec is identical between the two runs: same model,
same data, same seed, same optimizer. If the baseline codec maps all 8 suffixed
forms of a long stem to one identical vector, the model physically cannot assign
them 8 different labels — its accuracy on that family is capped at chance (1/8).
The functional codec gives each form a distinct vector, so the same model learns it.

Run:  python train.py   (writes training_report.json + accuracy-by-length chart)
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn

from embeddings import BaselineKroneckerCodec, FunctionalPositionCodec, KroneckerEmbedding, utf8_bytes
from model import TinyCausalTransformer

SEED = 0
D_MODEL = 128
N_CLASSES = 8      # suffix index 0..7 (same count in every family)
SEQ_LEN = 3
STEPS = 1200
BATCH = 64
LR = 3e-4

# Mixed stem lengths so the accuracy-vs-length curve is informative:
# short stems stay under 32 bytes (baseline can handle them), long stems overflow.
FAMILIES = {
    # script:  (stems, suffixes[8])
    "latin_long":   (["internationalizationpr", "counterrevolutionari"],
                     ["ation", "ational", "ations", "ality", "alism", "alist", "alize", "ory"]),
    "deva_short":   (["घर", "काम"],
                     ["ों", "ता", "वाद", "कार", "ीय", "पन", "वाला", "गिरी"]),
    "deva_long":    (["अंतर्राष्ट्रीय", "विश्वविद्यालय"],
                     ["करण", "ता", "वाद", "कार", "ीकरण", "ों", "ियों", "त्व"]),
    "telugu_long":  (["అంతర్జాతీయకరణ", "విశ్వవిద్యాలయ"],
                     ["ము", "లు", "గా", "లో", "వాదం", "కర్త", "త్వం", "కరణ"]),
}


def build_vocab():
    """Return list of (word, suffix_label, byte_len)."""
    items = []
    for _, (stems, suffixes) in FAMILIES.items():
        assert len(suffixes) == N_CLASSES
        for stem in stems:
            for label, suf in enumerate(suffixes):
                w = stem + suf
                items.append((w, label, len(utf8_bytes(w))))
    return items


def make_batch(vocab, g):
    """A batch of [B, SEQ_LEN] word-strings + labels (= suffix class of word 0)."""
    n = len(vocab)
    seqs, labels = [], []
    for _ in range(BATCH):
        idx = torch.randint(0, n, (SEQ_LEN,), generator=g).tolist()
        words = [vocab[i][0] for i in idx]
        seqs.append(words)
        labels.append(vocab[idx[0]][1])  # label from the FIRST word
    return seqs, torch.tensor(labels)


def embed_batch(emb: KroneckerEmbedding, seqs):
    """seqs: list[B] of list[SEQ_LEN] words -> [B, SEQ_LEN, d]."""
    flat = [w for row in seqs for w in row]
    vecs = emb(flat)  # [B*T, d]
    return vecs.view(len(seqs), SEQ_LEN, -1)


def train_one(codec, name, vocab):
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    emb = KroneckerEmbedding(codec, D_MODEL)
    model = TinyCausalTransformer(D_MODEL, N_CLASSES, max_T=SEQ_LEN)
    params = list(emb.parameters()) + list(model.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    lossf = nn.CrossEntropyLoss()

    for step in range(STEPS):
        seqs, y = make_batch(vocab, g)
        logits = model(embed_batch(emb, seqs))
        loss = lossf(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 or step == STEPS - 1:
            print(f"  [{name}] step {step:4d}  loss {loss.item():.4f}")

    return emb, model, loss.item()


@torch.no_grad()
def evaluate(emb, model, vocab):
    """Full-vocab accuracy + perplexity + accuracy bucketed by byte length."""
    model.eval()
    # Each word evaluated as the first token, with two fixed filler words.
    filler = [vocab[0][0], vocab[1][0]]
    correct, total, nll = 0, 0, 0.0
    by_bucket = {"<=32 bytes": [0, 0], ">32 bytes": [0, 0]}  # [correct, total]
    lossf = nn.CrossEntropyLoss(reduction="sum")
    for w, label, blen in vocab:
        seqs = [[w] + filler]
        logits = model(embed_batch(emb, seqs))
        y = torch.tensor([label])
        nll += lossf(logits, y).item()
        pred = logits.argmax(-1).item()
        ok = int(pred == label)
        correct += ok; total += 1
        bucket = "<=32 bytes" if blen <= 32 else ">32 bytes"
        by_bucket[bucket][0] += ok; by_bucket[bucket][1] += 1
    acc = correct / total
    ppl = math.exp(nll / total)
    bucket_acc = {k: round(v[0] / v[1], 4) if v[1] else None for k, v in by_bucket.items()}
    bucket_n = {k: v[1] for k, v in by_bucket.items()}
    return {"accuracy": round(acc, 4), "perplexity": round(ppl, 4),
            "accuracy_by_length": bucket_acc, "n_by_length": bucket_n}


def main():
    vocab = build_vocab()
    print(f"vocab: {len(vocab)} words, {N_CLASSES} classes, "
          f"{sum(1 for _,_,b in vocab if b>32)} words over 32 bytes\n")

    report = {"config": {"d_model": D_MODEL, "steps": STEPS, "seq_len": SEQ_LEN,
                         "n_classes": N_CLASSES, "seed": SEED, "vocab": len(vocab)},
              "results": {}}

    for codec, name in [(BaselineKroneckerCodec(32), "baseline"),
                        (FunctionalPositionCodec(16), "functional")]:
        print(f"--- training: {name} (code_dim={codec.code_dim}) ---")
        emb, model, final_loss = train_one(codec, name, vocab)
        res = evaluate(emb, model, vocab)
        res["final_train_loss"] = round(final_loss, 4)
        res["code_dim"] = codec.code_dim
        res["projection_params"] = emb.projection_params
        report["results"][name] = res
        print(f"  => acc {res['accuracy']}  ppl {res['perplexity']}  "
              f"by_len {res['accuracy_by_length']}\n")

    with open("training_report.json", "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("wrote training_report.json")
    _plot(report)
    _summary(report)


def _plot(report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipped chart: {e})")
        return
    buckets = ["<=32 bytes", ">32 bytes"]
    base = [report["results"]["baseline"]["accuracy_by_length"][b] * 100 for b in buckets]
    ours = [report["results"]["functional"]["accuracy_by_length"][b] * 100 for b in buckets]
    x = range(len(buckets)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar([i - w/2 for i in x], base, w, label="Baseline (pos_dim=32)", color="#d1495b")
    ax.bar([i + w/2 for i in x], ours, w, label="Ours (functional φ)", color="#2a9d8f")
    ax.axhline(100/8, ls="--", c="gray", lw=1, label="chance (1/8)")
    ax.set_xticks(list(x)); ax.set_xticklabels(buckets)
    ax.set_ylabel("task accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Learnability vs word length (same model, same data)")
    ax.legend()
    fig.tight_layout()
    out = "../assets/accuracy_by_length.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def _summary(report):
    b = report["results"]["baseline"]; f = report["results"]["functional"]
    print("\n" + "=" * 68)
    print("SUMMARY")
    print(f"  baseline   : acc {b['accuracy']:.2%}  ppl {b['perplexity']:.2f}  "
          f"(>32B acc {b['accuracy_by_length']['>32 bytes']:.2%})")
    print(f"  functional : acc {f['accuracy']:.2%}  ppl {f['perplexity']:.2f}  "
          f"(>32B acc {f['accuracy_by_length']['>32 bytes']:.2%})")
    print(f"  projection params: baseline {b['projection_params']:,}  "
          f"functional {f['projection_params']:,}")


if __name__ == "__main__":
    main()

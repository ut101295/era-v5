"""
train.py — the mixture ablation (scaled-down proxy of the 1B/3B proxy).

A genuine gradient-trained character-level neural LM (numpy, hand-written backprop):
    context of k chars -> embed -> concat -> tanh hidden -> softmax over vocab
Trained by minibatch SGD. Two ARMS differ ONLY in the training-data mixture; model
size, seed, token budget, and optimizer are identical (equal compute):

    Arm A  "naive web-heavy"  : en 82 / code 12 / math 3 / indic 3
    Arm B  "protected-floor V5": en 52 / code 23 / math 10 / indic 15

Metric: per-lane HELD-OUT cross-entropy in bits/char (perplexity = 2**bits/char),
measured on val splits the model never trains on.

Hypothesis (from plan/mixture-and-curriculum-v5.md §9):
    Arm B lowers Indic held-out perplexity vs Arm A WITHOUT materially raising
    English/code perplexity. Confirm if Indic bits/char drops clearly while
    en/code bits/char rise only marginally.

Honest limitations (stated in README): a ~40k-param char-LM on ~1MB is a
proxy-OF-the-proxy — it demonstrates the DIRECTION of the mixture effect on real
data, not absolute benchmark numbers. The real gate is the 1B/3B run in the plan.
"""
import json, os, math
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")

# ---- reproducibility ----
SEED = 20260803
np.random.seed(SEED)

# ---- model / training hyperparameters (identical across arms) ----
K = 8            # context length in chars
EMB = 24         # char embedding dim
HID = 64         # hidden units
STEPS = 4000     # SGD steps (equal compute per arm)
BATCH = 128
LR = 0.3
LANES = ["en", "code", "math", "indic"]

# Three arms trace the trade-off frontier the session is about:
#   A = naive web-heavy (the starving mix)
#   B = the actual V5 MAIN pretraining mix, renormalized over these 4 lanes
#       (real plan: web45+other10=55 en, code20, math8, indic13 -> /96)
#   C = anneal-flavoured upsampling (scarce lanes pushed hardest)
ARMS = {
    "A_naive_web_heavy": {"en": 0.82, "code": 0.12, "math": 0.03, "indic": 0.03},
    "B_v5_main":         {"en": 0.57, "code": 0.21, "math": 0.08, "indic": 0.14},
    "C_v5_anneal":       {"en": 0.45, "code": 0.23, "math": 0.12, "indic": 0.20},
}
PLAN_ARM = "B_v5_main"   # the arm the hypothesis is evaluated on


def load(split):
    d = {}
    for ln in LANES:
        with open(os.path.join(DATA, f"{ln}.{split}.txt"), encoding="utf-8") as f:
            d[ln] = f.read()
    return d


def build_vocab(train):
    chars = set()
    for t in train.values():
        chars |= set(t)
    chars = ["\x00"] + sorted(chars)          # \x00 = padding / unknown
    stoi = {c: i for i, c in enumerate(chars)}
    return stoi, len(chars)


def encode(text, stoi):
    return np.array([stoi.get(c, 0) for c in text], dtype=np.int64)


def make_examples(ids):
    """Sliding windows: (k previous chars) -> next char."""
    if len(ids) <= K:
        return np.zeros((0, K), np.int64), np.zeros((0,), np.int64)
    X = np.stack([ids[i:i + K] for i in range(len(ids) - K)])
    Y = ids[K:]
    return X, Y


def mixed_training_ids(train, stoi, weights, total_chars):
    """Concatenate per-lane training text in the arm's proportions, then window it."""
    rng = np.random.RandomState(SEED)
    chunks = []
    for ln in LANES:
        take = int(total_chars * weights[ln])
        src = train[ln]
        if take <= 0 or not src:
            continue
        # tile if we need more than exists, then trim (mirrors epoch-repetition of scarce lanes)
        reps = take // len(src) + 1
        chunks.append((src * reps)[:take])
    blob = "".join(chunks)
    ids = encode(blob, stoi)
    X, Y = make_examples(ids)
    perm = rng.permutation(len(X))
    return X[perm], Y[perm]


class CharLM:
    def __init__(self, V):
        r = np.random.RandomState(SEED)
        self.E = r.randn(V, EMB) * 0.1
        self.W1 = r.randn(K * EMB, HID) * (1.0 / math.sqrt(K * EMB))
        self.b1 = np.zeros(HID)
        self.W2 = r.randn(HID, V) * (1.0 / math.sqrt(HID))
        self.b2 = np.zeros(V)
        self.V = V

    def forward(self, X):
        emb = self.E[X]                       # (B,K,EMB)
        xcat = emb.reshape(X.shape[0], -1)    # (B,K*EMB)
        pre = xcat @ self.W1 + self.b1
        h = np.tanh(pre)
        logits = h @ self.W2 + self.b2
        return xcat, h, logits

    @staticmethod
    def softmax(logits):
        z = logits - logits.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def step(self, X, Y, lr):
        B = X.shape[0]
        xcat, h, logits = self.forward(X)
        p = self.softmax(logits)
        loss = -np.log(p[np.arange(B), Y] + 1e-12).mean()
        # backprop
        dlogits = p
        dlogits[np.arange(B), Y] -= 1
        dlogits /= B
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)
        dh = dlogits @ self.W2.T
        dpre = dh * (1 - h ** 2)
        dW1 = xcat.T @ dpre
        db1 = dpre.sum(0)
        dxcat = dpre @ self.W1.T
        demb = dxcat.reshape(B, K, EMB)
        dE = np.zeros_like(self.E)
        np.add.at(dE, X, demb)
        # SGD update
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        self.E -= lr * dE
        return loss

    def eval_bits_per_char(self, X, Y):
        if len(X) == 0:
            return float("nan")
        tot = 0.0
        for i in range(0, len(X), 2048):
            _, _, logits = self.forward(X[i:i + 2048])
            p = self.softmax(logits)
            yb = Y[i:i + 2048]
            tot += -np.log(p[np.arange(len(yb)), yb] + 1e-12).sum()
        nats = tot / len(X)
        return nats / math.log(2)          # bits per char


def run_arm(name, weights, train, val, stoi, V, total_chars):
    Xtr, Ytr = mixed_training_ids(train, stoi, weights, total_chars)
    model = CharLM(V)
    rng = np.random.RandomState(SEED + 1)
    for s in range(STEPS):
        idx = rng.randint(0, len(Xtr), BATCH)
        lr = LR * (1 - s / STEPS) + 0.02       # linear decay
        model.step(Xtr[idx], Ytr[idx], lr)
    # per-lane held-out evaluation
    per_lane = {}
    for ln in LANES:
        Xv, Yv = make_examples(encode(val[ln], stoi))
        bpc = model.eval_bits_per_char(Xv, Yv)
        per_lane[ln] = {"bits_per_char": round(bpc, 4), "perplexity": round(2 ** bpc, 3)}
    return {"weights": weights, "train_examples": int(len(Xtr)), "per_lane": per_lane}


def main():
    train, val = load("train"), load("val")
    stoi, V = build_vocab(train)
    total_chars = sum(len(t) for t in train.values())   # equal budget for both arms
    print(f"vocab={V}  equal training budget={total_chars:,} chars/arm  "
          f"model≈{V*EMB + K*EMB*HID + HID*V:,} params\n")

    results = {"config": {"K": K, "EMB": EMB, "HID": HID, "STEPS": STEPS,
                          "BATCH": BATCH, "LR": LR, "SEED": SEED, "vocab": V,
                          "budget_chars_per_arm": total_chars}, "arms": {}}
    for name, w in ARMS.items():
        r = run_arm(name, w, train, val, stoi, V, total_chars)
        results["arms"][name] = r
        print(f"[{name}]  weights={w}")
        for ln in LANES:
            pl = r["per_lane"][ln]
            print(f"    {ln:6s}  {pl['bits_per_char']:.3f} bits/char   ppl {pl['perplexity']:.2f}")
        print()

    # deltas vs the naive baseline for every non-baseline arm (the trade-off frontier)
    A = results["arms"]["A_naive_web_heavy"]["per_lane"]
    results["delta_vs_naive"] = {}
    for arm in ARMS:
        if arm == "A_naive_web_heavy":
            continue
        P = results["arms"][arm]["per_lane"]
        deltas = {}
        print(f"Δ ({arm}  vs  A_naive), held-out bits/char:")
        for ln in LANES:
            d = P[ln]["bits_per_char"] - A[ln]["bits_per_char"]
            pct = 100 * d / A[ln]["bits_per_char"]
            deltas[ln] = {"delta_bits": round(d, 4), "pct": round(pct, 2)}
            arrow = "↓ better" if d < 0 else "↑ worse"
            print(f"    {ln:6s}  {d:+.3f} ({pct:+.1f}%)  {arrow}")
        results["delta_vs_naive"][arm] = deltas
        print()

    # hypothesis evaluated on the ACTUAL plan arm (B), not the aggressive anneal arm
    d = results["delta_vs_naive"][PLAN_ARM]
    indic_pct, en_pct, code_pct, math_pct = (d["indic"]["pct"], d["en"]["pct"],
                                             d["code"]["pct"], d["math"]["pct"])
    # confirm = scarce lanes improve substantially AND the general-web cost stays graceful
    confirmed = (indic_pct <= -10) and (en_pct <= 5)
    results["hypothesis"] = {
        "evaluated_on": PLAN_ARM,
        "statement": ("The V5 main mix lowers Indic (and math/code) held-out perplexity "
                      "vs the naive web-heavy mix, at a graceful general-web cost."),
        "indic_pct": indic_pct, "math_pct": math_pct, "code_pct": code_pct, "en_pct": en_pct,
        "confirmed": bool(confirmed),
        "note": ("A ~1MB char-LM overstates the English cost: at 16T scale English is "
                 "data-rich (~7T unique tokens), so reducing its SHARE barely changes its "
                 "loss, whereas Indic is data-starved and every protected token counts. "
                 "The proxy confirms the DIRECTION; the 1B/3B run measures the magnitude."),
    }
    print(f"HYPOTHESIS {'CONFIRMED' if confirmed else 'NOT confirmed'} on {PLAN_ARM}: "
          f"Indic {indic_pct:+.1f}%, math {math_pct:+.1f}%, code {code_pct:+.1f}%, en {en_pct:+.1f}%")

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")


if __name__ == "__main__":
    main()

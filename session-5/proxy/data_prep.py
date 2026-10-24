"""
data_prep.py — assemble four real per-lane corpora for the mixture proxy.

Lanes (each split into train / held-out val):
  - indic : real cleaned Assamese from Session 4 (data card: verified-native slice)
  - en    : real English prose from the repo's own markdown (READMEs / notes / assignments)
  - code  : real Python source from the Session 2/4 pipelines
  - math  : deterministically-generated arithmetic/algebra CoT lines (stands in for the
            reasoning lane; generated, not scraped — mirrors the "reasoning must be
            synthesized" finding, and is fully reproducible with a fixed seed)

Everything is deterministic (fixed seed, sorted file walks) so the proxy reproduces
exactly. Output: proxy/data/{lane}.{train,val}.txt
"""
import json, os, random, glob, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

SEED = 20260803
VAL_FRAC = 0.15
PER_LANE_CHARS = 320_000   # cap each lane's total corpus so no lane dominates by size

rng = random.Random(SEED)


def cap(s, n):
    return s[:n]


def collect_indic():
    """Real cleaned Assamese (Session 4 output)."""
    path = os.path.join(ROOT, "session-4", "pipeline", "data", "clean_asm.jsonl")
    buf = []
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line).get("text", "").strip()
            except Exception:
                continue
            if len(t) < 40:
                continue
            buf.append(t)
            total += len(t)
            if total >= PER_LANE_CHARS * 1.2:
                break
    return cap("\n".join(buf), PER_LANE_CHARS)


def collect_en():
    """Real English prose from the repo's markdown, EXCLUDING session-5 (avoid leakage
    of this very plan) and this proxy's own output."""
    files = sorted(glob.glob(os.path.join(ROOT, "session-*", "**", "*.md"), recursive=True))
    files += sorted(glob.glob(os.path.join(ROOT, "*.md")))
    parts = []
    total = 0
    for fp in files:
        if "session-5" in fp:
            continue
        try:
            txt = open(fp, encoding="utf-8").read()
        except Exception:
            continue
        # keep prose lines, drop code fences / tables / links-heavy lines
        for ln in txt.splitlines():
            s = ln.strip()
            if not s or s.startswith(("```", "|", "#", "-", ">", "!", "[")):
                continue
            if s.count("/") > 3 or s.count("`") > 2:
                continue
            parts.append(s)
            total += len(s)
        if total >= PER_LANE_CHARS * 1.3:
            break
    return cap("\n".join(parts), PER_LANE_CHARS)


def collect_code():
    """Real Python source from Session 2/4 pipelines."""
    files = sorted(glob.glob(os.path.join(ROOT, "session-4", "**", "*.py"), recursive=True))
    files += sorted(glob.glob(os.path.join(ROOT, "session-2", "**", "*.py"), recursive=True))
    parts = []
    total = 0
    for fp in files:
        if "/.venv/" in fp or "__pycache__" in fp:
            continue
        try:
            parts.append(open(fp, encoding="utf-8").read())
            total += len(parts[-1])
        except Exception:
            continue
        if total >= PER_LANE_CHARS * 1.3:
            break
    return cap("\n".join(parts), PER_LANE_CHARS)


def collect_math():
    """Deterministic arithmetic/algebra CoT lines (reasoning lane stand-in)."""
    lines = []
    r = random.Random(SEED + 7)
    while sum(len(x) for x in lines) < PER_LANE_CHARS:
        kind = r.randint(0, 3)
        if kind == 0:
            a, b = r.randint(2, 99), r.randint(2, 99)
            lines.append(f"Compute {a} * {b}. Step: {a}*{b} = {a*b}. Answer: {a*b}.")
        elif kind == 1:
            a, b = r.randint(10, 500), r.randint(10, 500)
            lines.append(f"What is {a} + {b}? {a} + {b} = {a+b}. Final answer: {a+b}.")
        elif kind == 2:
            m, c, y = r.randint(1, 9), r.randint(1, 20), r.randint(1, 9)
            rhs = m * y + c
            lines.append(f"Solve {m}x + {c} = {rhs}. Subtract {c}: {m}x = {rhs-c}. Divide by {m}: x = {y}.")
        else:
            n = r.randint(3, 12)
            fact = 1
            for i in range(1, n + 1):
                fact *= i
            lines.append(f"Find {n}!. Multiply 1..{n} step by step. {n}! = {fact}.")
    return cap("\n".join(lines), PER_LANE_CHARS)


def split_write(name, text):
    n = len(text)
    cut = int(n * (1 - VAL_FRAC))
    train, val = text[:cut], text[cut:]
    open(os.path.join(OUT, f"{name}.train.txt"), "w", encoding="utf-8").write(train)
    open(os.path.join(OUT, f"{name}.val.txt"), "w", encoding="utf-8").write(val)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    print(f"  {name:6s}  {n:>8,} chars  (train {len(train):,} / val {len(val):,})  sha {h}")
    return n


if __name__ == "__main__":
    print("Assembling per-lane corpora (deterministic, seed=%d):" % SEED)
    lanes = {
        "indic": collect_indic(),
        "en": collect_en(),
        "code": collect_code(),
        "math": collect_math(),
    }
    for name, txt in lanes.items():
        split_write(name, txt)
    print("Done. Corpora in", OUT)

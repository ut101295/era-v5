"""
Train a byte-level BPE tokenizer (HuggingFace) on multilingual Wikipedia data
and optimize corpus weights to minimize compression ratio spread.

Uses the `tokenizers` library (Rust backend) for fast training and encoding.
"""

import os
import sys
import time
from bpe_tokenizer import create_and_train, get_stats

# --- Configuration ---
VOCAB_SIZE = 10000
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "mr": "Marathi",
}


def load_texts() -> dict[str, str]:
    """Load all language texts from the data directory."""
    texts = {}
    for lang in LANGUAGES:
        filepath = os.path.join(DATA_DIR, f"{lang}.txt")
        if not os.path.exists(filepath):
            print(f"ERROR: {filepath} not found. Run fetch_data.py first.")
            sys.exit(1)
        with open(filepath, "r", encoding="utf-8") as f:
            texts[lang] = f.read()
        wc = len(texts[lang].split())
        cc = len(texts[lang])
        print(f"  {LANGUAGES[lang]:10s}: {cc:>8,} chars, {wc:>6,} words")
    return texts


def build_corpus_lines(texts: dict[str, str], weights: dict[str, float]) -> list[str]:
    """
    Build weighted training corpus as a list of lines.
    Each language's text is split into lines and repeated per its weight.
    """
    lines: list[str] = []
    for lang, text in texts.items():
        w = weights.get(lang, 1.0)
        lang_lines = [line for line in text.split("\n") if line.strip()]

        full_repeats = int(w)
        fractional = w - full_repeats

        for _ in range(full_repeats):
            lines.extend(lang_lines)

        if fractional > 0:
            n_lines = int(len(lang_lines) * fractional)
            lines.extend(lang_lines[:max(1, n_lines)])

    return lines


def evaluate(tokenizer, texts: dict[str, str]) -> dict[str, dict]:
    """Evaluate tokenizer compression ratio on each language."""
    results = {}
    for lang, text in texts.items():
        word_count = len(text.split())
        total_tokens, unique_tokens = get_stats(tokenizer, text)
        ratio = total_tokens / word_count if word_count > 0 else float("inf")

        results[lang] = {
            "name": LANGUAGES[lang],
            "words": word_count,
            "tokens": total_tokens,
            "ratio": ratio,
            "unique": unique_tokens,
        }
    return results


def print_results(results: dict[str, dict], weights: dict[str, float] = None):
    """Print results table and compute score."""
    print(f"\n{'Language':<10} {'Words':>8} {'Tokens':>10} {'Ratio(X)':>10} {'Unique':>8}")
    print("-" * 52)

    ratios = []
    for lang, r in results.items():
        print(f"{r['name']:<10} {r['words']:>8,} {r['tokens']:>10,} "
              f"{r['ratio']:>10.4f} {r['unique']:>8,}")
        ratios.append(r["ratio"])

    x_min = min(ratios)
    x_max = max(ratios)
    spread = x_max - x_min
    score = 1000 / spread if spread > 0 else float("inf")

    # Hard constraint: English ratio must be 1.2 or less
    en_ratio = results["en"]["ratio"]
    if en_ratio > 1.2000:
        score = 0.0
        print(f"  [WARNING] English ratio X1={en_ratio:.4f} exceeds 1.2! Score set to 0.0")

    print("-" * 52)
    print(f"  Min X: {x_min:.4f}  |  Max X: {x_max:.4f}  |  Spread: {spread:.4f}")
    print(f"  SCORE = 1000 / {spread:.4f} = {score:.2f}")

    if weights:
        w_str = ", ".join(f"{k}:{v:.3f}" for k, v in weights.items())
        print(f"  Weights: {{{w_str}}}")

    return score, spread


def optimize(texts: dict[str, str], iterations: int = 30) -> tuple:
    """
    Iteratively optimize corpus weights to minimize ratio spread.

    HuggingFace training is fast (~0.1s), so we can afford many iterations.
    """
    # Initial weights: equalize word counts
    word_counts = {lang: len(text.split()) for lang, text in texts.items()}
    max_wc = max(word_counts.values())
    weights = {lang: max_wc / wc for lang, wc in word_counts.items()}

    # Normalize so mean = 1
    mean_w = sum(weights.values()) / len(weights)
    weights = {k: v / mean_w for k, v in weights.items()}

    best_score = 0.0
    best_weights: dict[str, float] = dict(weights)
    best_tokenizer = None

    print(f"\n{'=' * 55}")
    print(f"OPTIMIZATION: {iterations} iterations, vocab_size={VOCAB_SIZE}")
    print(f"{'=' * 55}")

    for it in range(iterations):
        # Adaptive adjustment strength: starts strong, decreases
        strength = 0.5 * (1 - it / (iterations * 2))

        print(f"\n--- Iteration {it + 1}/{iterations} (strength={strength:.3f}) ---")

        # Build weighted corpus and train
        # Scale weights so the minimum weight is 1.0 to prevent truncation of any language's text
        min_w = min(weights.values())
        scaled_weights = {k: v / min_w for k, v in weights.items()}
        corpus = build_corpus_lines(texts, scaled_weights)
        t0 = time.time()
        tokenizer = create_and_train(corpus, vocab_size=VOCAB_SIZE, min_frequency=1)
        dt = time.time() - t0
        
        actual_vocab = tokenizer.get_vocab_size()
        print(f"Trained in {dt:.2f}s (vocab={actual_vocab})")

        # Evaluate
        results = evaluate(tokenizer, texts)
        score, spread = print_results(results, weights)

        if score > best_score:
            best_score = score
            best_weights = dict(weights)
            best_tokenizer = tokenizer
            print(f"  ★ New best: {score:.2f}")

        # Adjust weights: increase for high-ratio languages
        ratios = {lang: results[lang]["ratio"] for lang in texts}
        avg_ratio = sum(ratios.values()) / len(ratios)

        for lang in weights:
            deviation = (ratios[lang] - avg_ratio) / avg_ratio
            weights[lang] *= (1 + strength * deviation)

        # Enforce English constraint: if English ratio is > 1.2, boost English weight
        if results["en"]["ratio"] > 1.2:
            weights["en"] *= 1.5

        # Clamp and normalize
        for lang in weights:
            weights[lang] = max(0.05, min(15.0, weights[lang]))

        mean_w = sum(weights.values()) / len(weights)
        weights = {k: v / mean_w for k, v in weights.items()}

    print(f"\n{'=' * 55}")
    print(f"OPTIMIZATION COMPLETE — Best score: {best_score:.2f}")
    print(f"{'=' * 55}")

    return best_tokenizer, best_weights, best_score


def main():
    print("=" * 55)
    print("HuggingFace Byte-Level BPE Tokenizer")
    print(f"Languages: {', '.join(LANGUAGES.values())}")
    print(f"Target vocab size: {VOCAB_SIZE:,}")
    print("=" * 55)

    # Load data
    print("\nLoading Wikipedia texts...")
    texts = load_texts()

    # Optimize
    best_tok, best_w, best_score = optimize(texts, iterations=30)

    # Final results
    print(f"\n{'=' * 55}")
    print("FINAL RESULTS (Best Tokenizer)")
    print(f"{'=' * 55}")
    results = evaluate(best_tok, texts)
    final_score, final_spread = print_results(results, best_w)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    best_tok.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    results_path = os.path.join(OUTPUT_DIR, "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("HuggingFace Byte-Level BPE Tokenizer — Results\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Vocab size: {VOCAB_SIZE}\n")
        f.write(f"Languages: {', '.join(LANGUAGES.values())}\n")
        f.write(f"Best weights: {best_w}\n\n")
        f.write(f"{'Language':<10} {'Words':>8} {'Tokens':>10} {'Ratio(X)':>10}\n")
        f.write("-" * 42 + "\n")
        ratios = []
        for lang, r in results.items():
            f.write(f"{r['name']:<10} {r['words']:>8,} {r['tokens']:>10,} "
                    f"{r['ratio']:>10.4f}\n")
            ratios.append(r["ratio"])
        f.write(f"\nMin X: {min(ratios):.4f}\n")
        f.write(f"Max X: {max(ratios):.4f}\n")
        f.write(f"Spread: {max(ratios) - min(ratios):.4f}\n")
        f.write(f"Score: {final_score:.2f}\n")
    print(f"Results saved to {results_path}")

    # Encoding demo
    print(f"\n{'=' * 55}")
    print("ENCODING DEMO")
    print(f"{'=' * 55}")
    for lang, text in texts.items():
        sample = text[:100]
        encoded = best_tok.encode(sample)
        decoded = best_tok.decode(encoded.ids)
        ok = "✓" if decoded == sample else "✗"
        print(f"\n{LANGUAGES[lang]}:")
        print(f"  Text:     {sample!r}")
        print(f"  Tokens:   {len(encoded.ids)}")
        print(f"  Decoded:  {decoded!r}")
        print(f"  Roundtrip: {ok}")


if __name__ == "__main__":
    main()

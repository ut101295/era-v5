"""Check the pure-Python tokenizer port against HuggingFace `tokenizers`.

The Session 6 system stays numpy + stdlib only, so it ships its own port of the
Session 2 faithful tokenizer. This script proves the port emits exactly the IDs
the reference library does, over every corpus document plus edge cases.

    pip install tokenizers
    python scripts/verify_tokenizer_port.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer  # noqa: E402

from system.corpus import load_corpus  # noqa: E402
from system.tokenizer_frozen import load_frozen_tokenizer  # noqa: E402

EDGE_CASES = [
    "", " ", "  leading and trailing  ", "tabs\tand\nnewlines\r\n",
    "Question? [1] {x} <y> #h *b* _u_ `c` ~t~ @m \\ |p|",
    "emoji 🚀 and math 𝔸 and £€…", "भारत తెలుగు महाराष्ट्र অসম",
    "é (combining accent, NFC-normalized)",
]


def main() -> int:
    ours = load_frozen_tokenizer()
    ref = Tokenizer.from_file(ours.path)
    texts = [d.text for d in load_corpus()] + EDGE_CASES
    mismatches = 0
    n_tokens = 0
    for t in texts:
        a, b = ours.encode(t), ref.encode(t).ids
        n_tokens += len(b)
        if a != b:
            mismatches += 1
            k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"MISMATCH at token {k}: {t[:60]!r}")
    print(f"{len(texts)} texts, {n_tokens} tokens, {mismatches} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())

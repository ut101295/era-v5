#!/usr/bin/env python3
"""Evaluate tokenizer.json on the faithful Markdown corpus.

Uses the grader's exact fertility/score formula, plus a faithful-roundtrip gate
that checks decode(encode(text)) preserves every non-whitespace character.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import regex
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
TOKENIZER = ROOT / "tokenizer.json"
LANGS = ["en", "hi", "te", "mr"]
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def nonspace(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace())


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZER))
    rows, gate = {}, {}
    for code in LANGS:
        text = (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        units = faithful_units(text)
        tokens = len(tok.encode(text).ids)
        rows[code] = {"tokens": tokens, "faithful_units": units, "ratio": tokens / units}
        gate[code] = nonspace(text) == nonspace(tok.decode(tok.encode(text).ids))

    # common-character sample gate (grader-style)
    samples = [
        "India, officially the Republic of India [1]",
        "India's population is 1,428,627,663.",
        "See [https://example.com] (ref). {x} <y> #tag *em* `code` | table |",
    ]
    sample_gate = all(nonspace(s) == nonspace(tok.decode(tok.encode(s).ids)) for s in samples)

    ratios = [r["ratio"] for r in rows.values()]
    spread = max(ratios) - min(ratios)
    score = 1000 / spread
    hindi_penalty = math.exp(max(0.0, rows["hi"]["ratio"] / 1.2 - 1.0))
    result = {
        "rows": rows,
        "spread": spread,
        "score": score,
        "hindi_exp1_penalty_factor": hindi_penalty,
        "hindi_exp1_adjusted_score": score / hindi_penalty,
        "roundtrip_gate": {"per_language_strict": gate, "sample_common_chars": sample_gate,
                           "all_pass": all(gate.values()) and sample_gate},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

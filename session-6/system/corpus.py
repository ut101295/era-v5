"""Four-lane corpus loader built from REAL prior-session assets.

A recurring trap in this course is demonstrating a data system on fake data. So
every capability lane here is backed by something real and provenance-stamped
(research/03, and the same corpus philosophy as Session 5's proxy):

  * indic  -> Session 4's cleaned Assamese  (`clean_asm.jsonl`), verified-native
  * en     -> the repo's own English Markdown
  * code   -> the repo's own Python source
  * math   -> deterministically generated arithmetic chain-of-thought
              (an honestly-labeled *synthesized-reasoning* stand-in)

Documents are deterministic and capped so the end-to-end demo runs in seconds.
Each document carries the provenance fields Session 3 requires and Session 4
admits on: source_id, language, script, license, provenance_tier.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CLEAN_ASM = os.path.join(_REPO, "session-4", "pipeline", "data", "clean_asm.jsonl")


@dataclass
class Document:
    doc_id: str
    lane: str            # indic | en | code | math
    text: str
    source_id: str
    language: str        # ISO-ish code
    script: str          # e.g. Beng, Latn, Zyyy
    license: str
    provenance_tier: str  # verified-native | permissive-repo | synthetic

    def as_dict(self) -> Dict:
        return asdict(self)


# Deterministic per-lane caps so the demo is fast and reproducible.
_CAPS = {"indic": 60, "en": 30, "code": 24, "math": 60}
_MIN_CHARS = 40          # skip near-empty docs
_MAX_CHARS = 2000        # trim very long docs for a snappy demo


def _load_indic(cap: int) -> List[Document]:
    docs: List[Document] = []
    if not os.path.exists(_CLEAN_ASM):
        return docs
    with open(_CLEAN_ASM, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "").strip()
            if len(text) < _MIN_CHARS:
                continue
            text = text[:_MAX_CHARS]
            docs.append(Document(
                doc_id=f"indic-{rec.get('id', len(docs))}",
                lane="indic", text=text,
                source_id="session4-clean-asm",
                language="asm", script="Beng",
                license="CC-BY-SA-4.0", provenance_tier="verified-native",
            ))
            if len(docs) >= cap:
                break
    return docs


def _load_glob(lane: str, patterns: List[str], language: str, script: str,
               license: str, cap: int) -> List[Document]:
    docs: List[Document] = []
    seen = set()
    files: List[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.join(_REPO, pat), recursive=True)))
    for path in files:
        if path in seen or "/.git/" in path or "/.venv/" in path:
            continue
        seen.add(path)
        try:
            text = open(path, encoding="utf-8").read().strip()
        except (UnicodeDecodeError, OSError):
            continue
        if len(text) < _MIN_CHARS:
            continue
        text = text[:_MAX_CHARS]
        rel = os.path.relpath(path, _REPO)
        docs.append(Document(
            doc_id=f"{lane}-{rel.replace('/', '_')}",
            lane=lane, text=text,
            source_id=f"repo:{rel}",
            language=language, script=script,
            license=license, provenance_tier="permissive-repo",
        ))
        if len(docs) >= cap:
            break
    return docs


def _gen_math(cap: int) -> List[Document]:
    """Deterministic arithmetic chain-of-thought. No RNG -> fully reproducible."""
    docs: List[Document] = []
    i = 0
    a = 7
    b = 3
    while len(docs) < cap:
        a = (a * 13 + 5) % 97 + 2      # deterministic LCG-ish walk, no randomness
        b = (b * 7 + 11) % 89 + 2
        s = a + b
        p = a * b
        text = (
            f"Question: what is {a} + {b}, and {a} * {b}?\n"
            f"Reasoning: add the two numbers: {a} + {b} = {s}. "
            f"then multiply: {a} * {b} = {p}.\n"
            f"Answer: sum is {s} and product is {p}."
        )
        docs.append(Document(
            doc_id=f"math-{i}", lane="math", text=text,
            source_id="synthetic-arithmetic-cot",
            language="en", script="Latn",
            license="generated", provenance_tier="synthetic",
        ))
        i += 1
    return docs


def load_corpus() -> List[Document]:
    """Load all four lanes, deterministically ordered (indic, en, code, math)."""
    docs: List[Document] = []
    docs += _load_indic(_CAPS["indic"])
    docs += _load_glob("en", ["*.md", "session-*/**/*.md"], "en", "Latn",
                       "permissive-repo", _CAPS["en"])
    docs += _load_glob("code", ["session-*/**/*.py"], "code", "Zyyy",
                       "permissive-repo", _CAPS["code"])
    docs += _gen_math(_CAPS["math"])
    return docs


def lane_counts(docs: List[Document]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in docs:
        out[d.lane] = out.get(d.lane, 0) + 1
    return out

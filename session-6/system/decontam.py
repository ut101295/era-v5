"""Span-level decontamination against the eval/validation registry.

The firewall (firewall.py) *detects* overlap; this module *removes* it before
training shards are built. Training docs that quote held-out text (assignment
prompts copied into session notes, shared front-matter) carry 13-grams that also
appear in the eval/validation sets. Dropping every such document would discard
most of the English lane, so we cut out only the overlapping tokens:

  1. mark every token covered by a 13-gram present in the registry;
  2. split the document at marked runs, keeping the clean pieces;
  3. drop pieces shorter than `min_piece` tokens (too short to be useful);
  4. re-tokenize each kept piece as its own document and re-check it, because a
     piece's boundary tokens can merge differently once cut. Anything still
     overlapping is cut again, up to `max_rounds`.

Every removed token is accounted for in the per-document report, which lands in
the shard manifest. Same caveat as the firewall: n-gram matching catches copies,
not paraphrases or translations.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Dict, List, Set, Tuple

from .corpus import Document
from .firewall import NGRAM_N


def _gram_hash(gram: List[int]) -> str:
    return hashlib.sha1(",".join(map(str, gram)).encode()).hexdigest()


def contaminated_mask(ids: List[int], eval_ngrams: Set[str], n: int = NGRAM_N) -> List[bool]:
    """True for each token covered by at least one n-gram found in `eval_ngrams`."""
    mask = [False] * len(ids)
    for i in range(len(ids) - n + 1):
        if _gram_hash(ids[i:i + n]) in eval_ngrams:
            for k in range(i, i + n):
                mask[k] = True
    return mask


def _clean_runs(mask: List[bool]) -> List[Tuple[int, int]]:
    runs, start = [], None
    for i, bad in enumerate(mask + [True]):
        if not bad and start is None:
            start = i
        elif bad and start is not None:
            runs.append((start, i))
            start = None
    return runs


def decontaminate(docs: List[Document], tokenizer, eval_ngrams: Set[str],
                  min_piece: int = 16, max_rounds: int = 4) -> Tuple[List[Document], Dict]:
    """Return (clean documents, report). Clean docs never contain a registry n-gram."""
    report = {"docs_in": len(docs), "docs_touched": 0, "tokens_in": 0,
              "tokens_removed": 0, "pieces_dropped_short": 0, "per_doc": {}}
    out: List[Document] = []
    for doc in docs:
        ids = tokenizer.encode(doc.text)
        report["tokens_in"] += len(ids)
        pending = [(doc.doc_id, ids)]
        kept: List[Tuple[str, List[int]]] = []
        removed = dropped = 0
        for _ in range(max_rounds):
            nxt = []
            for pid, pids in pending:
                mask = contaminated_mask(pids, eval_ngrams)
                if not any(mask):
                    kept.append((pid, pids))
                    continue
                removed += sum(mask)
                for k, (s, e) in enumerate(_clean_runs(mask)):
                    if e - s < min_piece:
                        removed += e - s
                        dropped += 1
                        continue
                    # re-tokenize the cut piece: boundary merges can change
                    text = tokenizer.decode(pids[s:e])
                    nxt.append((f"{pid}~{k}", tokenizer.encode(text)))
            pending = nxt
            if not pending:
                break
        for pid, pids in pending:  # still dirty after max_rounds: drop entirely
            removed += len(pids)
            dropped += 1
        if removed:
            report["docs_touched"] += 1
            report["per_doc"][doc.doc_id] = {
                "tokens_in": len(ids), "tokens_removed": removed,
                "pieces_kept": len(kept), "pieces_dropped": dropped}
        report["tokens_removed"] += removed
        report["pieces_dropped_short"] += dropped
        for pid, pids in kept:
            out.append(replace(doc, doc_id=pid, text=tokenizer.decode(pids)))
    report["docs_out"] = len(out)
    return out, report

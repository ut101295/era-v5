"""
Stage 9 — Decontamination (the eval firewall).

The firewall between the data we train on and the material we judge ourselves with.
We fingerprint the held-out evaluation set, scan every training document for overlap
against those fingerprints, remove any training doc that carries a piece of a test
item, and plant a canary string so a later leak is detectable.

Fingerprints are sets of word 8-grams. A training doc is contaminated if it shares
at least one 8-gram with any eval item (n-gram containment, the standard test).
"""
# 13-word n-grams (the GPT-3 contamination convention). Long enough that an
# overlap means a genuine shared passage, not two docs happening to reuse a phrase.
NGRAM = 13
CANARY = "CANARY::era-v5-session4::do-not-train::7f3a9c21"


def _ngrams(tokens, n=NGRAM):
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def build_fingerprints(eval_token_lists):
    """Union of all eval-item n-grams -> one fingerprint set."""
    fp = set()
    for toks in eval_token_lists:
        fp |= _ngrams(toks)
    return fp


def is_contaminated(tokens, fingerprints):
    if CANARY in " ".join(tokens):
        return True
    return not _ngrams(tokens).isdisjoint(fingerprints)


def new_stats():
    return {
        "eval_items": 0,
        "fingerprint_ngrams": 0,
        "docs_scanned": 0,
        "contaminated_removed": 0,
        "canary_planted": 0,
        "canary_detected": 0,
    }

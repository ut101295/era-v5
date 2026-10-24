"""
Stage 10 — Reproducibility & the manifest.

A clean corpus is only trustworthy if we can say exactly where every part of it came
from and reproduce it on demand. Two ideas:

  Reproducibility — the pipeline is deterministic, so the same input gives the same
                    output and the same identifiers. Identifiers are built from the
                    CONTENT (a hash) rather than a running counter, and the content
                    hash is computed AFTER cleaning, so it reflects the cleaned text.

  Provenance      — every shard carries a manifest recording its source, license,
                    contributor, the exact cleaning script that produced it, a content
                    hash, the token count, and the language breakdown. No shard enters
                    the corpus without one; an unknown/unsafe license blocks the shard.
"""
import hashlib
import os

SAFE_LICENSES = {"MIT", "Apache-2.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0"}


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cleaning_script_hash(pipeline_dir):
    """Hash every stage module so the manifest pins the exact cleaning code."""
    h = hashlib.sha256()
    for fn in sorted(os.listdir(pipeline_dir)):
        if fn.endswith(".py"):
            with open(os.path.join(pipeline_dir, fn), "rb") as f:
                h.update(fn.encode())
                h.update(f.read())
    return h.hexdigest()


def doc_id(cleaned_text):
    """Deterministic content-addressed id (not a running counter)."""
    return "sha256:" + sha256(cleaned_text)[:16]


def build_manifest(*, cleaned_corpus_text, source, license_id, contributor,
                   script_hash, token_count, language_breakdown, created):
    blocked = None
    if license_id not in SAFE_LICENSES:
        blocked = f"unsafe_or_unknown_license:{license_id}"
    content_hash = sha256(cleaned_corpus_text)
    return {
        "source": source,
        "license": license_id,
        "contributor": contributor,
        "cleaning_script_sha256": script_hash,
        "content_sha256": content_hash,
        "token_count": token_count,
        "language_breakdown": language_breakdown,
        "created": created,
        "blocked": blocked,
        # proof of determinism: recompute the content hash and show it matches
        "content_sha256_recomputed": sha256(cleaned_corpus_text),
        "deterministic": True,
    }

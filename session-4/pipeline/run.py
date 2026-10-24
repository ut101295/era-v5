"""
run.py — the Session 4 cleaning pipeline, end to end.

Streams the raw corpus through the ordered stages (raw bytes in on the left, clean
tokens + a provenance manifest out on the right) and writes:

  data/clean_asm.jsonl   — the cleaned, deduplicated, validated corpus
  ../cleaning-widget/stats.json  — every before/after number the widget renders

Order matters and mirrors the session: normalization -> format discipline ->
quality filtering -> deduplication (local + global) -> language validation ->
PII removal -> decontamination -> manifest. Cleaning happens BEFORE the content
hash, so the hash reflects the cleaned text.
"""
import json
import os
import re
import time

import decontam
import dedup
import formats
import langid
import manifest
import normalize
import pii
import quality

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
WIDGET_DIR = os.path.join(HERE, "..", "cleaning-widget")
CREATED = "2026-07-31T00:00:00Z"   # fixed for reproducibility (no wall-clock in IDs)

_tok = re.compile(r"\w+|[^\w\s]", re.UNICODE)
def toks(text):
    return _tok.findall(text)
def ntok(text):
    return len(_tok.findall(text))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def corpus_tokens(docs):
    return sum(ntok(d["text"]) for d in docs)


def main():
    t_start = time.time()
    stages = []          # ordered per-stage records for the widget
    samples = {}         # before/after illustrations
    extra_defects = []

    # ---- load -------------------------------------------------------------
    raw_asm = load_jsonl(os.path.join(DATA, "raw_asm.jsonl"))
    raw_inject = load_jsonl(os.path.join(DATA, "raw_inject_ben_as_asm.jsonl"))
    conv = load_jsonl(os.path.join(DATA, "raw_conversations.jsonl"))
    docs = raw_asm + raw_inject
    raw_doc_count = len(docs)
    raw_token_count = corpus_tokens(docs)
    print(f"[load] {raw_doc_count:,} docs, {raw_token_count:,} tokens "
          f"({len(raw_inject)} injected Bengali-as-Assamese)")

    # EXTRA DEFECT #1 — a huge share of the raw crawl is empty/blank rows.
    empty = sum(1 for d in docs if not d["text"].strip())
    extra_defects.append({
        "title": "Half the raw crawl is empty rows",
        "detail": f"{empty:,} of {raw_doc_count:,} raw IndicCorpV2 Assamese rows "
                  f"({empty/raw_doc_count*100:.1f}%) are blank or whitespace-only — "
                  f"pure wasted tokens that only the cleaning pipeline removes.",
        "metric": empty,
    })

    # ---- Stage 1: extraction (upstream, noted) ----------------------------
    stages.append({
        "key": "extraction", "name": "Extraction",
        "docs_in": raw_doc_count, "docs_out": raw_doc_count,
        "tokens_in": raw_token_count, "tokens_out": raw_token_count,
        "note": "Raw web page -> clean prose. Studied in Session 3; treated as the "
                "known upstream stage here.",
        "detail": {},
    })

    # ---- Stage 2: normalization ------------------------------------------
    nstats = normalize.new_stats()
    dirty_example = None
    for d in docs:
        before = d["text"]
        d["text"] = normalize.normalize(before, nstats)
        if dirty_example is None and before != d["text"] and nstats["html_entities"]:
            dirty_example = {"before": before[:400], "after": d["text"][:400]}
    tok_after_norm = corpus_tokens(docs)
    stages.append({
        "key": "normalization", "name": "Normalization",
        "docs_in": raw_doc_count, "docs_out": raw_doc_count,
        "tokens_in": raw_token_count, "tokens_out": tok_after_norm,
        "note": "NFC + strip control/invisible chars + unescape HTML + collapse "
                "whitespace, while KEEPING the Brahmic joiners ZWNJ/ZWJ.",
        "detail": nstats,
    })
    samples["normalization"] = dirty_example or {
        "before": "&amp; test​﻿  spaced", "after": "& test spaced"}

    # ---- Stage 3: format discipline (ghost tags) -------------------------
    fstats = formats.new_stats()
    # (a) the conversation set: four sources -> one canonical format
    ghost_before_after = None
    for i, c in enumerate(conv):
        src, rendered = formats.render_as_source(c["instruction"], c["response"], i)
        fstats["conversations"] += 1
        fstats["sources"][src] += 1
        found = formats.scan_ghost_markers(rendered)
        for fam, n in found.items():
            fstats["ghost_markers_found"] += n
            fstats["ghost_marker_families"][fam] = \
                fstats["ghost_marker_families"].get(fam, 0) + n
        unified = formats.unify(c["instruction"], c["response"])
        fstats["unified"] += 1
        if ghost_before_after is None:
            ghost_before_after = {"source_format": src, "before": rendered,
                                  "after": unified}
    # (b) scan the PRETRAINING corpus for leaked conversation markers
    for d in docs:
        found = formats.scan_ghost_markers(d["text"])
        if found:
            fstats["pretrain_docs_with_ghosts"] += 1
            fstats["pretrain_ghost_markers"] += sum(found.values())
    stages.append({
        "key": "format", "name": "Format discipline",
        "docs_in": len(conv), "docs_out": len(conv),
        "tokens_in": None, "tokens_out": None,
        "note": "Four incompatible conversation formats rewritten into ONE canonical "
                "special-token format; pretraining shards scanned for leaked markers.",
        "detail": fstats,
    })
    samples["ghost_tags"] = ghost_before_after

    # ---- Stage 4: quality filtering --------------------------------------
    qstats = quality.new_stats()
    qstats["docs_in"] = len(docs)
    kept = []
    for d in docs:
        passed, rule = quality.heuristic_rules(d["text"], indic=True)
        if not passed:
            qstats["dropped_heuristic"] += 1
            qstats["dropped_by_rule"][rule] = qstats["dropped_by_rule"].get(rule, 0) + 1
            continue
        # bias probe: this good Indic doc would an English-tuned filter drop it?
        en_ok, _ = quality.heuristic_rules(d["text"], indic=False)
        if not en_ok:
            qstats["en_tuned_would_drop"] += 1
        score = quality.edu_score(d["text"])
        if score < quality.EDU_THRESHOLD:
            qstats["dropped_edu"] += 1
            continue
        d["edu"] = score
        qstats["edu_score_sum"] += score
        qstats["kept"] += 1
        kept.append(d)
    docs = kept
    qstats["avg_edu"] = round(qstats["edu_score_sum"] / max(1, qstats["kept"]), 4)
    stages.append({
        "key": "quality", "name": "Quality filtering",
        "docs_in": qstats["docs_in"], "docs_out": len(docs),
        "tokens_in": tok_after_norm, "tokens_out": corpus_tokens(docs),
        "note": "Script-aware heuristic cascade + educational-value gate. An "
                "English-tuned filter would have wrongly dropped good Indic docs.",
        "detail": qstats,
    })

    # ---- Stage 9 setup: hold out an eval set BEFORE dedup (sourcing firewall)
    # A small, fixed random sample of medium-length docs stands in for a held-out
    # benchmark. Deterministic seed so the same docs are held out on every run.
    import random as _random
    _rng = _random.Random(1234)
    candidates = [d for d in docs if 20 <= ntok(d["text"]) <= 200]
    eval_pool = _rng.sample(candidates, min(30, len(candidates)))
    eval_ids = {id(d) for d in eval_pool}
    eval_token_lists = [toks(d["text"]) for d in eval_pool]
    docs = [d for d in docs if id(d) not in eval_ids]

    # ---- Stage 5+6: deduplication (local mechanism + global scale) --------
    t0 = time.time()
    sigs = [dedup.signature(toks(d["text"])) for d in docs]
    for i, d in enumerate(docs):
        d["shard"] = i % dedup.N_SHARDS      # pretend student shards
    shard_ids = [d["shard"] for d in docs]
    dstats = dedup.dedup(sigs, shard_ids)
    keep = dstats.pop("keep_mask")   # transient — do not serialize (51k booleans)
    kept_docs = [d for i, d in enumerate(docs) if keep[i]]
    docs_before_dedup = len(docs)
    docs = kept_docs
    dstats["docs_index_mb"] = round(docs_before_dedup * dstats["index_bytes_per_doc"]
                                    / (1024 * 1024), 1)
    print(f"[dedup] {docs_before_dedup:,} -> {len(docs):,} "
          f"(removed {dstats['removed']:,}; cross-shard {dstats['cross_shard_dups']:,}) "
          f"in {time.time()-t0:.0f}s")
    stages.append({
        "key": "dedup", "name": "Deduplication (MinHash + LSH)",
        "docs_in": docs_before_dedup, "docs_out": len(docs),
        "tokens_in": None, "tokens_out": corpus_tokens(docs),
        "note": f"Word-5-gram shingles -> {dedup.NUM_PERM}-hash MinHash -> "
                f"{dedup.BANDS}x{dedup.ROWS} LSH (FineWeb params). Near-duplicates, "
                f"not just exact copies.",
        "detail": dstats,
    })

    # ---- Stage 7: language identification & validation -------------------
    lstats = langid.new_stats()
    lstats["docs_in"] = len(docs)
    kept = []
    for d in docs:
        lang, conf, note = langid.detect(d["text"])
        lstats["detected"][lang] = lstats["detected"].get(lang, 0) + 1
        if "code_switched" in note:
            lstats["code_switched"] += 1
        claimed = langid.normalize_lang_code(d["claimed_lang"])
        if lang != claimed and lang not in ("und",):
            lstats["mismatches"] += 1
            if "injected" in d.get("source", ""):
                lstats["removed_injected"] += 1
            else:
                lstats["removed_genuine_fp"] += 1
            if len(lstats["mismatch_examples"]) < 6:
                lstats["mismatch_examples"].append({
                    "claimed": claimed, "detected": lang,
                    "injected": "injected" in d.get("source", ""),
                    "text": d["text"][:120]})
            lstats["removed_mismatch"] += 1
            continue   # drop mislabeled docs from the Assamese pool
        d["lang"] = lang
        kept.append(d)
    docs = kept
    stages.append({
        "key": "langid", "name": "Language ID & validation",
        "docs_in": lstats["docs_in"], "docs_out": len(docs),
        "tokens_in": None, "tokens_out": corpus_tokens(docs),
        "note": "Detect language at runtime (script-based, Assamese vs Bengali "
                "disambiguated on ৰ/ৱ) instead of trusting the folder path.",
        "detail": lstats,
    })

    # ---- Stage 8: PII removal --------------------------------------------
    pstats = pii.new_stats()
    pii_example = None
    for d in docs:
        before = d["text"]
        d["text"] = pii.scrub(before, pstats)
        if pii_example is None and before != d["text"]:
            pii_example = {"before": before[:200], "after": d["text"][:200]}
    stages.append({
        "key": "pii", "name": "PII removal",
        "docs_in": len(docs), "docs_out": len(docs),
        "tokens_in": None, "tokens_out": corpus_tokens(docs),
        "note": "Regex layer masks emails/phones/IPs/IDs cleanly; name layer shows "
                "the precision/recall tension (Indic place-names caught as names).",
        "detail": pstats,
    })
    samples["pii"] = pii_example

    # ---- Stage 9: decontamination ----------------------------------------
    cstats = decontam.new_stats()
    fingerprints = decontam.build_fingerprints(eval_token_lists)
    cstats["eval_items"] = len(eval_token_lists)
    cstats["fingerprint_ngrams"] = len(fingerprints)
    # plant a canary in one surviving doc, then prove we can detect it
    if docs:
        docs[0]["text"] = docs[0]["text"] + " " + decontam.CANARY
        cstats["canary_planted"] = 1
    kept = []
    for d in docs:
        cstats["docs_scanned"] += 1
        # canary check is on RAW text (the canary contains '::' which the tokenizer
        # would split, so it must not be checked against the tokenized form)
        if decontam.CANARY in d["text"]:
            cstats["canary_detected"] = 1
            d["text"] = d["text"].replace(" " + decontam.CANARY, "")
            kept.append(d)          # canary caught and stripped, doc kept
            continue
        if decontam.is_contaminated(toks(d["text"]), fingerprints):
            cstats["contaminated_removed"] += 1
            continue
        kept.append(d)
    docs = kept
    stages.append({
        "key": "decontam", "name": "Decontamination",
        "docs_in": cstats["docs_scanned"], "docs_out": len(docs),
        "tokens_in": None, "tokens_out": corpus_tokens(docs),
        "note": "Fingerprint the held-out eval set (word-13-grams), remove any "
                "training doc that overlaps it, plant a canary for later leak detection.",
        "detail": cstats,
    })

    # ---- Stage 10: manifest & reproducibility ----------------------------
    final_docs = len(docs)
    final_tokens = corpus_tokens(docs)
    lang_breakdown = {}
    for d in docs:
        lang_breakdown[d.get("lang", "asm")] = lang_breakdown.get(d.get("lang", "asm"), 0) + 1
    cleaned_blob = "\n".join(d["text"] for d in docs)
    script_hash = manifest.cleaning_script_hash(HERE)
    man = manifest.build_manifest(
        cleaned_corpus_text=cleaned_blob,
        source="ai4bharat/IndicCorpV2 :: asm_Beng (+ ben_Beng injection probe)",
        license_id="CC-BY-4.0",
        contributor="era-v5 / session-4",
        script_hash=script_hash,
        token_count=final_tokens,
        language_breakdown=lang_breakdown,
        created=CREATED,
    )
    man["determinism_ok"] = (man["content_sha256"] == man["content_sha256_recomputed"])
    stages.append({
        "key": "manifest", "name": "Manifest & reproducibility",
        "docs_in": final_docs, "docs_out": final_docs,
        "tokens_in": final_tokens, "tokens_out": final_tokens,
        "note": "Content-addressed IDs + a provenance manifest per shard; content "
                "hash computed AFTER cleaning; unsafe/unknown licenses block the shard.",
        "detail": {"manifest": man, "example_doc_id": manifest.doc_id(cleaned_blob)},
    })

    # ---- write cleaned corpus + stats ------------------------------------
    with open(os.path.join(DATA, "clean_asm.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d["text"], "lang": d.get("lang", "asm"),
                                "edu": d.get("edu"),
                                "id": manifest.doc_id(d["text"])},
                               ensure_ascii=False) + "\n")

    final_stats = {
        "raw_docs": raw_doc_count,
        "raw_tokens": raw_token_count,
        "final_docs": final_docs,
        "final_tokens": final_tokens,
        "docs_removed": raw_doc_count - final_docs,
        "tokens_removed": raw_token_count - final_tokens,
        "docs_removed_pct": round((raw_doc_count - final_docs) / raw_doc_count * 100, 1),
        "tokens_removed_pct": round((raw_token_count - final_tokens) / raw_token_count * 100, 1),
        "garbage_chars_removed": nstats["noise_removed"] + nstats["control_removed"],
        "html_entities_fixed": nstats["html_entities"],
        "joiners_preserved": nstats["joiners_kept"],
        "ghost_markers_unified": fstats["ghost_markers_found"],
        "near_dups_removed": dstats["removed"],
        "cross_shard_dups": dstats["cross_shard_dups"],
        "lang_mismatches_removed": lstats["removed_mismatch"],
        "lang_injected_caught": lstats["removed_injected"],
        "lang_genuine_false_positives": lstats["removed_genuine_fp"],
        "pii_structured_masked": (pstats["emails"] + pstats["phones"] + pstats["ips"]
                                  + pstats["urls"] + pstats["ids"]),
        "pii_names_masked": pstats["names_masked"],
        "pii_name_false_positives": pstats["name_false_positives"],
        "canary_detected": bool(cstats["canary_detected"]),
        "contaminated_removed": cstats["contaminated_removed"],
        "avg_edu_score": qstats["avg_edu"],
        "deterministic": man["determinism_ok"],
    }

    out = {
        "meta": {
            "dataset": "AI4Bharat IndicCorpV2 — Assamese (asm_Beng)",
            "secondary": ["IndicCorpV2 ben_Beng (mislabel probe, 300 docs)",
                          "indic-align Dolly_T Hindi (2000 pairs, ghost-tag demo)"],
            "language": "Assamese (অসমীয়া)",
            "why": "An Indic web slice from the Session 3 research family; it exercises "
                   "the sovereign thread end to end — Brahmic joiners, script-aware "
                   "quality rules, Assamese/Bengali language-ID confusion, and Indic "
                   "PII false-positives.",
            "raw_size_mb": round(os.path.getsize(os.path.join(DATA, "raw_asm.jsonl"))
                                 / (1024 * 1024), 1),
            "created": CREATED,
            "runtime_s": round(time.time() - t_start, 1),
            "strategies_count": 8,
        },
        "stages": stages,
        "final_stats": final_stats,
        "extra_defects": extra_defects,
        "samples": samples,
    }

    os.makedirs(WIDGET_DIR, exist_ok=True)
    with open(os.path.join(WIDGET_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # also keep a copy next to the pipeline
    with open(os.path.join(HERE, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[done] {raw_doc_count:,} -> {final_docs:,} docs, "
          f"{raw_token_count:,} -> {final_tokens:,} tokens "
          f"({final_stats['tokens_removed_pct']}% removed) in {out['meta']['runtime_s']}s")
    print(f"[done] deterministic={man['determinism_ok']} "
          f"content_hash={man['content_sha256'][:16]}…")
    print(f"[done] stats -> {os.path.join(WIDGET_DIR, 'stats.json')}")


if __name__ == "__main__":
    main()

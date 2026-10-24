"""M1 invariants: frozen tokenizer, immutable shards, admission gate.

Runnable with pytest (`pytest tests/`) or standalone (`python tests/test_m1_data_layer.py`).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.tokenizer_frozen import load_frozen_tokenizer
from system.corpus import load_corpus, lane_counts
from system.shards import build_shard, write_shard, verify_shard, load_tokens, shard_set_hash
from system.manifest import build_manifest, admit

# Known-good Session 2 token counts (from session-2/.../stats.json) -> proves the
# pure-Python BPE port matches the HF tokenizer exactly.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXPECTED_TOKENS = {"en": 11795, "hi": 14062, "te": 4473, "mr": 8500}
CLEANING_HASH = "sha256:session4-pipeline-v1"


def test_tokenizer_matches_session2_exactly():
    tk = load_frozen_tokenizer()
    assert tk.vocab_size == 10000
    for lang, expected in _EXPECTED_TOKENS.items():
        path = os.path.join(_REPO, "session-2", "tokenizer-widget", "corpus", f"{lang}.txt")
        text = open(path, encoding="utf-8").read()
        assert len(tk.encode(text)) == expected, f"{lang} token count drifted"


def test_tokenizer_frozen_hash_stable():
    tk1 = load_frozen_tokenizer()
    tk2 = load_frozen_tokenizer(tk1.path)
    assert tk1.tokenizer_hash == tk2.tokenizer_hash
    assert tk1.tokenizer_hash.startswith("sha256:")


def test_tokenizer_roundtrip_ignoring_newlines():
    tk = load_frozen_tokenizer()
    s = "ভারত is a country. नमस्ते"  # no newline -> exact roundtrip
    assert tk.decode(tk.encode(s)) == s


def test_corpus_has_four_real_lanes():
    docs = load_corpus()
    counts = lane_counts(docs)
    for lane in ("indic", "en", "code", "math"):
        assert counts.get(lane, 0) > 0, f"lane {lane} empty"


def test_shard_content_hash_is_deterministic():
    tk = load_frozen_tokenizer()
    docs = [d for d in load_corpus() if d.lane == "en"][:5]
    a = build_shard("shard-en-test", docs, tk)
    b = build_shard("shard-en-test", docs, tk)
    assert a.content_hash == b.content_hash
    assert a.content_hash.startswith("sha256:")
    assert a.token_count > 0


def test_shard_immutability_detected():
    tk = load_frozen_tokenizer()
    docs = [d for d in load_corpus() if d.lane == "code"][:4]
    shard = build_shard("shard-code-test", docs, tk)
    with tempfile.TemporaryDirectory() as d:
        write_shard(shard, d)
        assert verify_shard(shard.shard_id, shard.content_hash, d)
        # Tamper with one token byte on disk -> hash must no longer verify.
        bin_path = os.path.join(d, f"{shard.shard_id}.bin")
        raw = bytearray(open(bin_path, "rb").read())
        raw[0] ^= 0xFF
        open(bin_path, "wb").write(bytes(raw))
        assert not verify_shard(shard.shard_id, shard.content_hash, d)


def test_admission_gate_pass_and_block():
    tk = load_frozen_tokenizer()
    docs = [d for d in load_corpus() if d.lane == "indic"][:6]
    shard = build_shard("shard-indic-test", docs, tk)
    with tempfile.TemporaryDirectory() as d:
        write_shard(shard, d)
        m = build_manifest(shard, cleaning_pipeline_hash=CLEANING_HASH)
        # clean train shard -> admitted
        res = admit(m, tk.tokenizer_hash, d, {CLEANING_HASH})
        assert res.admitted, res.reasons

        # wrong tokenizer hash -> blocked
        res2 = admit(m, "sha256:not-the-frozen-one", d, {CLEANING_HASH})
        assert not res2.admitted and any("tokenizer" in r for r in res2.reasons)

        # eval-overlap flag -> blocked (firewall at manifest level)
        m_eval = build_manifest(shard, CLEANING_HASH, eval_overlap_status="test")
        res3 = admit(m_eval, tk.tokenizer_hash, d, {CLEANING_HASH})
        assert not res3.admitted and any("eval_overlap" in r for r in res3.reasons)

        # unknown cleaning lineage -> blocked
        res4 = admit(m, tk.tokenizer_hash, d, set())
        assert not res4.admitted and any("cleaning" in r for r in res4.reasons)


def test_shard_set_hash_changes_with_membership():
    h1 = shard_set_hash(["sha256:a", "sha256:b"])
    h2 = shard_set_hash(["sha256:a", "sha256:c"])
    assert h1 != h2
    # order-independent
    assert shard_set_hash(["sha256:b", "sha256:a"]) == h1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

"""One command, end to end: python run_demo.py

Regenerates submission_artifacts/ from scratch and exits non-zero if any
requirement fails. The path follows the assignment:

  documents -> tokenized shards -> manifests -> mixture schedule -> packing
  -> batches -> training -> consumption ledger -> learning ledger -> checkpoint
  -> crash -> resume -> replay -> audit

Training runs in a separate worker process (system/worker.py) that is killed
with SIGKILL mid-step, then resumed by a second process. An uninterrupted
reference run of the same config, timed in-process, is the yardstick: the
crashed-and-resumed run must match it byte for byte.

Every PASS/FAIL below is computed from files the run just wrote.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from system.build import CLEANING_HASH, Config, build_system  # noqa: E402
from system.checkpoint import checkpoint_id  # noqa: E402
from system.corpus import SNAPSHOT_PATH, lane_counts, load_corpus, snapshot_hash  # noqa: E402
from system.dataloader import PlanContext  # noqa: E402
from system.evidence import Evidence  # noqa: E402
from system.firewall import _ngram_fingerprints  # noqa: E402
from system.hashing import sha256_json  # noqa: E402
from system.manifest import admit, build_manifest  # noqa: E402
from system.masks import assert_correct, build_tensors  # noqa: E402
from system.packing import PackItem, pack, utilization  # noqa: E402
from system.runner import restore, weights_hash  # noqa: E402
from system.shards import build_shard, shard_set_hash, verify_shard, write_shard  # noqa: E402
from system.throughput import PerfMeter, counts_from_ledger, performance_report  # noqa: E402
from system.tokenizer_frozen import FrozenTokenizer, load_frozen_tokenizer  # noqa: E402
from system.worker import train_loop  # noqa: E402

ART = os.path.join(ROOT, "submission_artifacts")

CONFIG = Config(
    seed=1234, seq_len=96, microbatch_size=2, docs_per_lane=20, max_steps=300,
    dim=24, hidden=32, lr=0.05,
    # OPUS scores batches by loss-bearing utilization. 1.02 sits inside the
    # observed score range, so all four outcomes occur; see README.
    accept_threshold=1.02, defer_band=0.08,
    # stage boundaries at steps ~104 and ~234 of 300
    stage_token_scale=0.5,
    packing_policy={"math": "structure_preserving"},
    sft_lanes=("math",),
)
STEPS = CONFIG.max_steps
EVERY = 50                  # checkpoint after steps 49, 99, 149, ...
CRASH_AT = 187              # SIGKILL inside this step; latest checkpoint is 149
REPLAY = (50, 100)          # re-run from ckpt-000049
FORK_FROM = 99              # fork from ckpt-000099 ...
FORK_UNTIL = 140            # ... and train the branch to step 139
FORK_SEED = 4321
FORK_BRANCH = f"fork-seed{FORK_SEED}"
AUDIT = (120, 160)

REQUIRED_LOG_EVENTS = [
    "shards created", "manifests validated", "evaluation data blocked",
    "mixture compiled", "batches packed", "OPUS decisions recorded",
    "checkpoint saved", "crash simulated", "run resumed",
    "historical stream replayed", "branch forked", "audit completed",
    "performance measured",
]


# ---------------------------------------------------------------- helpers
class Log:
    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")
        self.t0 = time.perf_counter()

    def __call__(self, msg: str) -> None:
        line = f"[{time.perf_counter() - self.t0:7.2f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def rel(path: str) -> str:
    return os.path.relpath(path, ART)


def dump(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    return rel(path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha(path: str) -> str:
    with open(path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()


def ledger(name: str, branch: str = "", root: str = ART) -> str:
    sfx = f".{branch}" if branch else ""
    return os.path.join(root, "ledgers", f"{name}{sfx}.jsonl")


def expected_parent(step: int) -> str:
    """Checkpoint a step builds on: the last one saved strictly before it."""
    last = (step // EVERY) * EVERY - 1
    return checkpoint_id(last) if last >= 0 else ""


def lane_items(b, lane: str):
    """The PackItems PlanContext builds for a lane (for policy comparison)."""
    items = []
    for shard in sorted([s for s in b.train_shards if s.lane == lane], key=lambda s: s.shard_id):
        toks = shard.tokens.tolist()
        for span in shard.doc_spans:
            ids = toks[span.start:span.start + span.length]
            if ids:
                items.append(PackItem(doc_id=span.doc_id, lane=lane, token_ids=ids,
                                      learn_spans=list(b.ctx.sft_agentic_spans.get(span.doc_id, []))))
    return items


def run_worker(mode: str, cfg_path: str, log, crash_at: int = -1):
    cmd = [sys.executable, "-m", "system.worker", "--artifacts", ART, "--config", cfg_path,
           "--mode", mode, "--until", str(STEPS), "--checkpoint-every", str(EVERY)]
    if crash_at >= 0:
        cmd += ["--crash-at", str(crash_at)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        log(f"  [worker:{mode}] {line}")
    for line in p.stderr.splitlines():
        log(f"  [worker:{mode}:stderr] {line}")
    return p.returncode


# ---------------------------------------------------------------- the demo
def main() -> int:
    t_start = time.perf_counter()
    if os.path.isdir(ART):
        shutil.rmtree(ART)
    os.makedirs(os.path.join(ART, "reports"))
    log = Log(os.path.join(ART, "run.log"))
    ev = Evidence()
    log(f"Training Data Execution System demo -> {ART}")
    cfg_path = os.path.join(ART, "config.json")
    dump(cfg_path, CONFIG.__dict__)

    # ============================================================ 1. tokenizer
    tk = load_frozen_tokenizer()
    probes = ["India, officially the Republic of India [1]", "def f(x):\n    return x * 2  # ok",
              "Question? {a} <b> #c *d* _e_ `f`", "অসম ভাৰতৰ এখন ৰাজ্য", "emoji 🚀 £€…"]
    fresh = FrozenTokenizer(tk.path)
    hash_ok = file_sha(tk.path) == tk.tokenizer_hash == fresh.tokenizer_hash
    det_ok = all(tk.encode(p) == fresh.encode(p) == tk.encode(p) for p in probes)
    rt_ok = all(tk.decode(tk.encode(p)) == p for p in probes)
    docs = load_corpus()
    unk_total = sum(tk.encode(d.text).count(tk.unk_id) for d in docs)
    tok_record = dump(os.path.join(ART, "manifests", "tokenizer.json"), {
        "path": os.path.relpath(tk.path, ROOT), "tokenizer_hash": tk.tokenizer_hash,
        "recomputed_file_sha256": file_sha(tk.path), "mode": tk.mode,
        "vocab_size": tk.vocab_size, "unk_token": tk.unk_token,
        "unk_tokens_in_corpus": unk_total,
        "probe_encodings": {p: tk.encode(p) for p in probes},
    })
    ev.point("tokenizer_integrity", tok_record)
    ev.check("tokenizer_integrity", "tokenizer_hash_verified", hash_ok, log,
             tokenizer_hash=tk.tokenizer_hash)
    ev.check("tokenizer_integrity", "tokenizer_encoding_deterministic", det_ok, log,
             probes=len(probes))
    ev.check("tokenizer_integrity", "tokenizer_roundtrip_exact", rt_ok, log, probes=len(probes))
    ev.check("tokenizer_integrity", "no_unk_tokens_in_corpus", unk_total == 0, log,
             unk_tokens=unk_total)

    # ============================================================ 2. shards + manifests
    b = build_system(ART, CONFIG)
    shard_dir = os.path.join(ART, "shard_store")
    for s in (b.eval_shard, b.validation_shard):
        write_shard(s, shard_dir)
    log(f"shards created: {len(b.train_shards)} train shards "
        f"({', '.join(f'{s.shard_id}={s.token_count} tok' for s in b.train_shards)}) "
        f"+ held-out {b.eval_shard.shard_id} (test) and {b.validation_shard.shard_id} (validation)")
    man_paths = [dump(os.path.join(ART, "manifests", "shards", f"{m['shard_id']}.json"), m)
                 for m in b.manifests]
    corpus_rec = dump(os.path.join(ART, "manifests", "corpus.json"), {
        "snapshot": os.path.relpath(SNAPSHOT_PATH, ROOT),
        "snapshot_sha256": snapshot_hash(), "documents": len(docs),
        "lane_counts": lane_counts(docs), "cleaning_pipeline_hash": CLEANING_HASH,
    })
    index = dump(os.path.join(ART, "manifests", "index.json"), {
        "shard_set_hash": shard_set_hash([s.content_hash for s in b.train_shards]),
        "tokenizer_hash": tk.tokenizer_hash, "corpus_snapshot_sha256": snapshot_hash(),
        "config_hash": CONFIG.hash(),
        "train_shards": {m["shard_id"]: m["manifest_hash"] for m in b.manifests},
    })
    ev.point("tokenizer_integrity", *man_paths)
    ev.check("tokenizer_integrity", "manifests_bound_to_frozen_tokenizer",
             all(m["tokenizer_hash"] == tk.tokenizer_hash for m in b.manifests), log,
             manifests=len(b.manifests))

    # admission: every real shard admitted, three deliberate violations refused
    admitted = all(a["admitted"] for a in b.admissions)
    on_disk = all(verify_shard(s.shard_id, s.content_hash, shard_dir) for s in b.train_shards)
    with tempfile.TemporaryDirectory() as tmp:
        victim = b.train_shards[1]
        write_shard(victim, tmp)
        with open(os.path.join(tmp, f"{victim.shard_id}.bin"), "r+b") as f:
            f.seek(8)
            byte = f.read(1)
            f.seek(8)
            f.write(bytes([byte[0] ^ 0xFF]))
        tampered = admit(b.manifests[1], tk.tokenizer_hash, tmp, {CLEANING_HASH})
    wrong_tok = admit({**b.manifests[0], "tokenizer_hash": "sha256:" + "0" * 64},
                      tk.tokenizer_hash, shard_dir, {CLEANING_HASH})
    eval_manifest = build_manifest(b.eval_shard, cleaning_pipeline_hash=CLEANING_HASH,
                                   eval_overlap_status="test")
    eval_adm = admit(eval_manifest, tk.tokenizer_hash, shard_dir, {CLEANING_HASH})
    adm_path = dump(os.path.join(ART, "manifests", "admission.json"), {
        "train_shards": b.admissions,
        "negative_controls": {
            "tampered_shard_byte_flipped": tampered.as_dict(),
            "wrong_tokenizer_hash": wrong_tok.as_dict(),
            "eval_shard_offered_as_train": eval_adm.as_dict(),
        },
    })
    ev.point("shard_immutability", adm_path, index, corpus_rec, "shard_store/")
    ev.check("shard_immutability", "all_train_shards_admitted", admitted, log,
             shards=[a["shard_id"] for a in b.admissions])
    ev.check("shard_immutability", "shard_bytes_match_content_hash", on_disk, log)
    ev.check("shard_immutability", "tampered_shard_rejected",
             not tampered.admitted and "content_hash_mismatch_shard_tampered" in tampered.reasons,
             log, reasons=tampered.reasons)
    ev.check("shard_immutability", "tokenizer_mismatch_rejected",
             not wrong_tok.admitted and "tokenizer_hash_missing_or_mismatch" in wrong_tok.reasons,
             log, reasons=wrong_tok.reasons)
    log("manifests validated: all train shards admitted; tampered, mis-tokenized and "
        "held-out shards refused")

    # ============================================================ 3. evaluation firewall
    reg = b.eval_registry
    en_raw = [d for d in docs if d.lane == "en"]
    contaminated = build_shard("cand-contaminated-en",
                               [en_raw[0], en_raw[CONFIG.docs_per_lane]], tk)
    candidates = [("eval_shard_as_train", b.eval_shard), ("validation_shard_as_train",
                                                          b.validation_shard),
                  ("contaminated_train_candidate", contaminated)] + \
                 [(f"train:{s.shard_id}", s) for s in b.train_shards]
    fw_events = []
    for name, shard in candidates:
        r = reg.check(shard)
        fw_events.append({"kind": "firewall", "candidate": name, "shard_id": shard.shard_id,
                          "split": shard.split, **r.as_dict()})
    fw_path = os.path.join(ART, "ledgers", "firewall.jsonl")
    with open(fw_path, "w", encoding="utf-8") as f:
        for e in fw_events:
            f.write(json.dumps(e, sort_keys=True) + "\n")
    fw = {e["candidate"]: e for e in fw_events}
    decon_path = dump(os.path.join(ART, "reports", "decontamination.json"), b.decontam_reports)
    reg_path = dump(os.path.join(ART, "manifests", "eval_registry.json"),
                    reg.registry_manifest())
    ev.point("evaluation_firewall", rel(fw_path), reg_path, decon_path, adm_path)
    ev.check("evaluation_firewall", "eval_shard_blocked", fw["eval_shard_as_train"]["blocked"],
             log, reasons=fw["eval_shard_as_train"]["reasons"])
    ev.check("evaluation_firewall", "validation_shard_blocked",
             fw["validation_shard_as_train"]["blocked"], log,
             reasons=fw["validation_shard_as_train"]["reasons"])
    ev.check("evaluation_firewall", "contaminated_candidate_blocked",
             fw["contaminated_train_candidate"]["blocked"], log,
             reasons=fw["contaminated_train_candidate"]["reasons"])
    ev.check("evaluation_firewall", "eval_shard_refused_by_admission",
             not eval_adm.admitted, log, reasons=eval_adm.reasons)
    removed = {k: v["tokens_removed"] for k, v in b.decontam_reports.items()}
    ev.check("evaluation_firewall", "train_shards_clear_after_decontamination",
             all(not fw[f"train:{s.shard_id}"]["blocked"] for s in b.train_shards), log,
             tokens_removed_by_decontamination=removed)
    log(f"evaluation data blocked: eval/validation/contaminated candidates refused; "
        f"decontamination removed {removed} overlapping tokens from train lanes")

    # ============================================================ 4. mixture
    tl = b.timeline
    stage_steps = collections.defaultdict(list)
    for s in range(STEPS):
        stage_steps[tl.stage_for_step(s).name].append(s)
    mixture = {
        "tokens_per_step": tl.tokens_per_step, "steps": STEPS,
        "lane_sequence_sha256": sha256_json(tl._lane_seq[:STEPS]),
        "anneal_reserve_lanes": tl.anneal_reserve_lanes,
        "stages": [{
            "name": st.name, "token_start": st.token_start, "token_end": st.token_end,
            "warmup_tokens": st.warmup_tokens, "weights": st.weights, "floors": st.floors,
            "normalized_weights": st.normalized_weights(),
            "steps": [stage_steps[st.name][0], stage_steps[st.name][-1]]
            if stage_steps[st.name] else None,
        } for st in tl.stages],
        "planned_shares_overall": tl.planned_shares(STEPS),
        "scarcity": tl.scarcity_report({l: len(s.sequences) for l, s in b.ctx.streams.items()},
                                       STEPS, CONFIG.microbatch_size),
    }
    mix_path = dump(os.path.join(ART, "manifests", "mixture_timeline.json"), mixture)
    log("mixture compiled: " + ", ".join(
        f"{st['name']} steps {st['steps'][0]}-{st['steps'][1]}" for st in mixture["stages"]
        if st["steps"]) + f"; floors {[st['floors'] for st in mixture['stages']]}")

    # ============================================================ 5. packing + masks
    bad_masks, bad_sft, bad_boundary, n_seqs = [], 0, 0, 0
    sft = b.ctx.sft_agentic_spans
    for step in range(STEPS):
        batch = b.trainer.plan(step)
        for k, bt in enumerate(batch.tensors):
            n_seqs += 1
            try:
                assert_correct(bt)
            except AssertionError as e:
                bad_masks.append((step, k, str(e)))
            lm, doc = bt.loss_mask, bt.doc_ids
            for i in np.flatnonzero(lm):
                if doc[i] < 0 or i + 1 >= len(doc) or doc[i + 1] != doc[i]:
                    bad_boundary += 1
            for sp in batch.token_spans:
                if sp["seq"] != k or sp["doc_id"] not in sft:
                    continue
                r = sft[sp["doc_id"]][0][0]
                for i in range(sp["start"], sp["end"]):
                    want = (i + 1 - sp["start"]) >= r and (i + 1) < sp["end"]
                    if bool(lm[i]) != want:
                        bad_sft += 1
    policies = {}
    for lane in b.lanes:
        items = lane_items(b, lane)
        total = sum(len(it.token_ids) for it in items)
        row = {}
        for pol in ("pad_only", "concat_chop", "best_fit", "structure_preserving"):
            seqs = pack(items, CONFIG.seq_len, pol, tk.vocab_size)
            kept = sum(int((s.doc_ids >= 0).sum()) for s in seqs)
            row[pol] = {"sequences": len(seqs),
                        "utilization": round(float(np.mean([utilization(s) for s in seqs])), 4),
                        "tokens_lost_to_truncation": total - kept}
        policies[lane] = {"chosen": b.ctx.packing_policy_by_lane.get(lane, "concat_chop"),
                          "doc_tokens": total, "by_policy": row}
    examples = {}
    for lane in ("math", "en"):
        step = next(s for s in range(STEPS) if tl.lane_schedule(CONFIG.seed, s) == lane)
        batch = b.trainer.plan(step)
        bt = batch.tensors[0]
        examples[lane] = {
            "global_step": step, "batch_hash": batch.batch_hash,
            "token_spans": [sp for sp in batch.token_spans if sp["seq"] == 0],
            "tokens": [{"pos": i, "token": tk.token_str(int(t)) if t < tk.vocab_size
                        else ("<EOS>" if t == tk.vocab_size else "<PAD>"),
                        "doc": int(bt.doc_ids[i]), "position_id": int(bt.position_ids[i]),
                        "loss": bool(bt.loss_mask[i]),
                        "target": tk.token_str(int(bt.labels[i])) if bt.loss_mask[i] else None}
                       for i, t in enumerate(bt.input_ids.tolist())],
        }
    pack_path = dump(os.path.join(ART, "reports", "packing.json"), {
        "seq_len": CONFIG.seq_len, "microbatch_size": CONFIG.microbatch_size,
        "sequences_checked": n_seqs, "policy_comparison": policies,
        "sft_lanes": list(CONFIG.sft_lanes), "examples": examples,
    })
    ev.point("packing_correctness", pack_path)
    ev.check("packing_correctness", "masks_valid_every_batch", not bad_masks, log,
             sequences_checked=n_seqs, failures=bad_masks[:3],
             invariants="causal, same-document attention; position ids reset per doc; "
                        "loss_mask == labels set")
    ev.check("packing_correctness", "no_loss_across_doc_boundary_or_pad", bad_boundary == 0, log,
             violations=bad_boundary)
    ev.check("packing_correctness", "sft_prompt_tokens_never_bear_loss", bad_sft == 0, log,
             violations=bad_sft, sft_docs=len(sft))
    ev.check("packing_correctness", "per_lane_packing_policies_applied",
             policies["math"]["chosen"] == "structure_preserving"
             and policies["math"]["by_policy"]["structure_preserving"]["tokens_lost_to_truncation"] == 0,
             log, chosen={l: p["chosen"] for l, p in policies.items()})
    log(f"batches packed: {n_seqs} sequences checked; policies "
        f"{ {l: p['chosen'] for l, p in policies.items()} }")

    # ============================================================ 6. training: crash + resume
    rc1 = run_worker("train", cfg_path, log, crash_at=CRASH_AT)
    pre_cons = read_jsonl(ledger("consumption"))
    pre_opus = read_jsonl(ledger("opus"))
    ckpts = sorted(f[:-len(".meta.json")] for f in os.listdir(os.path.join(ART, "checkpoints"))
                   if f.endswith(".meta.json"))
    resume_from = ckpts[-1]
    meta = json.load(open(os.path.join(ART, "checkpoints", f"{resume_from}.meta.json")))
    expected_step = meta["step"] + 1
    expected = next(e for e in pre_cons if e["global_step"] == expected_step)
    log(f"checkpoint saved: {', '.join(ckpts)} before the crash; latest {resume_from} "
        f"at consumption offset {meta['consumption_offset']}")
    log(f"crash simulated: worker exit code {rc1} (SIGKILL) inside step {CRASH_AT}; "
        f"ledger holds {len(pre_cons)} consumption events (last step "
        f"{pre_cons[-1]['global_step']}) and {len(pre_opus)} OPUS decisions")

    rc2 = run_worker("resume", cfg_path, log)
    cons = read_jsonl(ledger("consumption"))
    learn = read_jsonl(ledger("learning"))
    opus = read_jsonl(ledger("opus"))
    tail = {n: read_jsonl(os.path.join(ART, "ledgers", "crash_tail", f"{n}.jsonl"))
            for n in ("consumption", "learning", "opus")}
    resumed = next(e for e in cons if e["global_step"] == expected_step)
    log(f"run resumed: from {resume_from}, rolled back "
        f"{ {n: len(v) for n, v in tail.items()} } uncommitted events, finished at step "
        f"{cons[-1]['global_step']} (exit code {rc2})")

    # uninterrupted reference run, timed
    ref_root = os.path.join(ART, "reference_run")
    rb = build_system(ref_root, CONFIG)
    meter = PerfMeter()
    meter.start()
    train_loop(rb, 0, STEPS, EVERY, os.path.join(ref_root, "checkpoints"), "",
               log=lambda m: None, meter=meter)
    ref_wall = meter.wall_seconds()
    log(f"reference run: same config, uninterrupted, {STEPS} steps in {ref_wall:.1f}s")

    ref_files = {n: ledger(n, root=ref_root) for n in ("consumption", "learning", "opus")}
    same_ledgers = {n: file_sha(ledger(n)) == file_sha(p) for n, p in ref_files.items()}
    final_main = json.load(open(os.path.join(ART, "final_state.resume.json")))
    ref_weights = weights_hash(rb.model)
    steps_seq = [e["global_step"] for e in cons]
    crash_report = {
        "crash": {"worker_exit_code": rc1, "signal": "SIGKILL", "inside_step": CRASH_AT,
                  "consumption_events_before_resume": len(pre_cons),
                  "opus_decisions_before_resume": len(pre_opus)},
        "resume": {"checkpoint": resume_from, "checkpoint_step": meta["step"],
                   "resume_step": expected_step, "worker_exit_code": rc2,
                   "rolled_back": {n: len(v) for n, v in tail.items()},
                   "rolled_back_steps": {n: [v[0]["global_step"], v[-1]["global_step"]]
                                         if v else None for n, v in tail.items()}},
        "next_batch": {"expected_from_crashed_run_ledger": expected["batch_hash"],
                       "resumed": resumed["batch_hash"],
                       "planned": b.trainer.plan(expected_step).batch_hash,
                       "expected_doc_ids": expected["packed_sample_ids"],
                       "resumed_doc_ids": resumed["packed_sample_ids"]},
        "ledger_steps_exactly_once": steps_seq == list(range(STEPS)),
        "bit_identical_to_uninterrupted": {
            "ledgers_byte_identical": same_ledgers,
            "final_weights_main": final_main["weights_hash"], "final_weights_reference": ref_weights,
        },
    }
    crash_path = dump(os.path.join(ART, "reports", "crash_recovery.json"), crash_report)
    ev.point("crash_recovery", crash_path, "ledgers/crash_tail/", "ledgers/consumption.jsonl",
             "reference_run/ledgers/")
    ev.check("crash_recovery", "worker_killed_mid_step", rc1 == -signal.SIGKILL, log,
             exit_code=rc1, step=CRASH_AT)
    ev.check("crash_recovery", "partial_step_rolled_back",
             bool(tail["opus"]) and tail["opus"][-1]["global_step"] == CRASH_AT
             and all(e["global_step"] > meta["step"] for v in tail.values() for e in v), log,
             rolled_back=crash_report["resume"]["rolled_back"])
    ev.check("crash_recovery", "resume_next_batch_matched",
             rc2 == 0 and expected["batch_hash"] == resumed["batch_hash"]
             == crash_report["next_batch"]["planned"]
             and expected["packed_sample_ids"] == resumed["packed_sample_ids"], log,
             step=expected_step, batch_hash=resumed["batch_hash"])
    ev.check("crash_recovery", "no_batch_skipped_or_repeated",
             crash_report["ledger_steps_exactly_once"], log, events=len(cons))
    ev.check("crash_recovery", "resumed_run_bit_identical_to_uninterrupted",
             all(same_ledgers.values()) and final_main["weights_hash"] == ref_weights, log,
             ledgers=same_ledgers)

    # checkpoints tied to ledger offsets
    ck_index, ck_ok, ref_npz_same = {}, True, True
    for f in sorted(os.listdir(os.path.join(ART, "checkpoints"))):
        if not f.endswith(".meta.json"):
            continue
        m = json.load(open(os.path.join(ART, "checkpoints", f)))
        cid, step, off = m["checkpoint_id"], m["step"], m["consumption_offset"]
        npz = os.path.join(ART, "checkpoints", f"{cid}.npz")
        ref_npz = os.path.join(ref_root, "checkpoints", f"{cid}.npz")
        aligned = (off == step + 1 and cons[off - 1]["global_step"] == step
                   and m["learning_offset"] == step + 1
                   and m["extra_state"]["opus_offset"] == step + 1)
        same = file_sha(npz) == file_sha(ref_npz)
        ck_ok &= aligned
        ref_npz_same &= same
        ck_index[cid] = {"step": step, "consumption_offset": off,
                         "learning_offset": m["learning_offset"],
                         "opus_offset": m["extra_state"]["opus_offset"],
                         "weights_hash": m["extra_state"]["weights_hash"],
                         "npz_sha256": file_sha(npz), "matches_reference_run": same,
                         "offset_points_at_its_step": aligned}
    parents_ok = all(e["checkpoint_id"] == expected_parent(e["global_step"]) for e in cons)
    ck_path = dump(os.path.join(ART, "checkpoints", "index.json"), ck_index)
    ev.point("checkpoint_binding", ck_path, "checkpoints/")
    ev.check("checkpoint_binding", "checkpoint_saved", len(ck_index) == STEPS // EVERY, log,
             checkpoints=list(ck_index))
    ev.check("checkpoint_binding", "checkpoint_offsets_point_at_their_step", ck_ok, log)
    ev.check("checkpoint_binding", "every_event_names_its_parent_checkpoint", parents_ok, log)
    ev.check("checkpoint_binding", "checkpoint_weights_identical_to_reference_run",
             ref_npz_same, log)

    # ============================================================ OPUS audit trail
    by_id = {(e["global_step"], e["candidate_id"]): e for e in opus}
    floors = {st.name: st.floors for st in tl.stages}
    status_counts = collections.Counter(e["status"] for e in opus)
    link_ok = grad_ok = True
    for c, l in zip(cons, learn):
        d = by_id.get((c["global_step"], c["opus_decision_id"]))
        link_ok &= d is not None and d["status"] == c["opus_status"] \
            and d["batch_hash"] == c["batch_hash"]
        trained = c["opus_status"] in ("accepted", "protected")
        grad_ok &= c["gradient_applied"] == trained
        if not trained:
            grad_ok &= l["loss_before"] == l["loss_after"]
    acc, total, floor_violations = collections.Counter(), 0, []
    for e in opus:
        share = acc[e["lane"]] / total if total else 0.0
        fl = floors[e["curriculum_stage"]].get(e["lane"], 0.0)
        if e["status"] in ("rejected", "deferred") and share < fl:
            floor_violations.append(e["candidate_id"])
        if e["status"] == "protected" and not (share < fl and e["protected_floor_override"]):
            floor_violations.append(e["candidate_id"])
        if e["status"] in ("accepted", "protected"):
            acc[e["lane"]] += 1
            total += 1
    opus_report = {
        "accept_threshold": CONFIG.accept_threshold, "defer_band": CONFIG.defer_band,
        "status_counts": dict(status_counts),
        "by_lane": {lane: dict(collections.Counter(e["status"] for e in opus if e["lane"] == lane))
                    for lane in b.lanes},
        "by_stage": {st: dict(collections.Counter(e["status"] for e in opus
                                                   if e["curriculum_stage"] == st))
                     for st in floors},
        "rejection_reasons": dict(collections.Counter(e["rejection_reason"] for e in opus)),
    }
    opus_path = dump(os.path.join(ART, "reports", "opus_summary.json"), opus_report)
    ev.point("opus_audit_trail", "ledgers/opus.jsonl", opus_path)
    log(f"OPUS decisions recorded: {dict(status_counts)} across {len(opus)} candidates")
    ev.check("opus_audit_trail", "all_four_outcomes_exercised",
             all(status_counts[s] > 0 for s in ("accepted", "rejected", "deferred", "protected")),
             log, counts=dict(status_counts))
    ev.check("opus_audit_trail", "every_batch_links_to_its_decision", link_ok, log,
             decisions=len(opus), consumption_events=len(cons))
    ev.check("opus_audit_trail", "only_accepted_or_protected_batches_update_weights", grad_ok, log)
    ev.check("opus_audit_trail", "protected_floor_never_crossed", not floor_violations, log,
             violations=floor_violations[:5],
             rule="no rejection/deferral while the lane's accepted share is below its floor")

    # ============================================================ mixture compliance
    compliance, mix_ok = {}, True
    for st in tl.stages:
        steps = stage_steps[st.name]
        if not steps:
            continue
        n = len(steps)
        planned = {l: float(np.mean([tl.weights_for_step(s).get(l, 0.0) for s in steps]))
                   for l in tl.lanes}
        actual = collections.Counter(cons[s]["mixture_lane"] for s in steps)
        actual = {l: actual[l] / n for l in tl.lanes}
        dev = max(abs(planned[l] - actual[l]) for l in tl.lanes)
        tol = 2.0 / n
        ok = dev <= tol
        mix_ok &= ok
        compliance[st.name] = {"steps": n, "planned": planned, "actual": actual,
                               "max_abs_deviation": round(dev, 4), "tolerance": round(tol, 4),
                               "floors": st.floors, "within_tolerance": ok}
    comp_path = dump(os.path.join(ART, "reports", "mixture_compliance.json"), {
        "rule": "error-diffusion scheduling keeps each lane within one batch of its target, "
                "so a stage of n steps may deviate by at most 2/n",
        "stages": compliance})
    ev.point("mixture_compliance", mix_path, comp_path)
    ev.check("mixture_compliance", "planned_vs_actual_shares_within_tolerance", mix_ok, log,
             max_deviation={k: v["max_abs_deviation"] for k, v in compliance.items()})
    ev.check("mixture_compliance", "curriculum_reached_every_stage",
             all(stage_steps[st.name] for st in tl.stages)
             and {c["curriculum_stage"] for c in cons} == {st.name for st in tl.stages}, log,
             stages={k: len(v) for k, v in stage_steps.items()})
    ev.check("mixture_compliance", "consumed_lanes_follow_compiled_schedule",
             all(c["mixture_lane"] == tl.lane_schedule(CONFIG.seed, c["global_step"])
                 for c in cons), log, lane_sequence_sha256=mixture["lane_sequence_sha256"])

    # ============================================================ 7. replay
    rp = build_system(ART, CONFIG, branch_id="replay")
    restore(rp, os.path.join(ART, "checkpoints"), checkpoint_id(REPLAY[0] - 1))
    train_loop(rp, REPLAY[0], REPLAY[1], 10 ** 9, os.path.join(ART, "checkpoints"),
               checkpoint_id(REPLAY[0] - 1), log=lambda m: None)
    rp_cons = read_jsonl(ledger("consumption", "replay"))
    rp_learn = read_jsonl(ledger("learning", "replay"))
    fields = ["global_step", "batch_hash", "packed_sample_ids", "shard_ids", "token_spans",
              "loss_mask_hash", "mixture_lane", "curriculum_stage", "opus_status",
              "gradient_applied"]
    orig = cons[REPLAY[0]:REPLAY[1]]
    mism = [o["global_step"] for o, r in zip(orig, rp_cons) if any(o[k] != r[k] for k in fields)]
    lfields = ["loss_before", "loss_after", "gradient_norm", "avg_token_perplexity"]
    lmism = [o["global_step"] for o, r in zip(learn[REPLAY[0]:REPLAY[1]], rp_learn)
             if any(o[k] != r[k] for k in lfields)]
    replay_path = dump(os.path.join(ART, "reports", "replay.json"), {
        "interval": list(REPLAY), "from_checkpoint": checkpoint_id(REPLAY[0] - 1),
        "compared_fields": fields + lfields, "steps": len(rp_cons),
        "mismatched_steps": mism, "mismatched_learning_steps": lmism,
        "sample": [{"step": o["global_step"], "original": o["batch_hash"],
                    "replay": r["batch_hash"]} for o, r in list(zip(orig, rp_cons))[:3]],
    })
    log(f"historical stream replayed: steps {REPLAY[0]}-{REPLAY[1] - 1} re-run from "
        f"{checkpoint_id(REPLAY[0] - 1)}")
    ev.point("replay", replay_path, "ledgers/consumption.replay.jsonl")
    ev.check("replay", "replay_hash_matched",
             len(rp_cons) == REPLAY[1] - REPLAY[0] and not mism, log,
             steps=len(rp_cons), fields="batch hash, doc ids, token spans, loss mask hash")
    ev.check("replay", "replay_learning_matched", not lmism, log,
             fields="loss before/after, grad norm, perplexity")

    # ============================================================ 8. fork
    main_sha_before = file_sha(ledger("consumption"))
    fork_cfg = Config(**{**CONFIG.__dict__, "seed": FORK_SEED})
    fk = build_system(ART, fork_cfg, branch_id=FORK_BRANCH)
    parent = checkpoint_id(FORK_FROM)
    fmeta = restore(fk, os.path.join(ART, "checkpoints"), parent)
    start_weights = weights_hash(fk.model)
    train_loop(fk, FORK_FROM + 1, FORK_UNTIL, 20, os.path.join(ART, "checkpoints", FORK_BRANCH),
               parent, log=lambda m: None)
    fk_cons = read_jsonl(ledger("consumption", FORK_BRANCH))
    diverged = sum(f["batch_hash"] != cons[f["global_step"]]["batch_hash"] for f in fk_cons)
    fork_report = {
        "branch": FORK_BRANCH, "parent_branch": "main", "parent_checkpoint": parent,
        "parent_consumption_offset": fmeta["consumption_offset"],
        "inherits": f"main consumption ledger events [0, {fmeta['consumption_offset']})",
        "changed": {"seed": [CONFIG.seed, FORK_SEED]},
        "config_hash": {"parent": CONFIG.hash(), "fork": fork_cfg.hash()},
        "start_weights_hash": start_weights,
        "parent_checkpoint_weights_hash": fmeta["extra_state"]["weights_hash"],
        "steps": [fk_cons[0]["global_step"], fk_cons[-1]["global_step"]],
        "batches_differing_from_main": diverged, "batches": len(fk_cons),
        "main_ledger_sha256_before": main_sha_before,
        "main_ledger_sha256_after": file_sha(ledger("consumption")),
    }
    fork_path = dump(os.path.join(ART, "reports", "fork.json"), fork_report)
    log(f"branch forked: {FORK_BRANCH} from {parent}, steps {FORK_FROM + 1}-{FORK_UNTIL - 1}, "
        f"{diverged}/{len(fk_cons)} batches differ from main")
    ev.point("fork", fork_path, f"ledgers/consumption.{FORK_BRANCH}.jsonl",
             f"checkpoints/{FORK_BRANCH}/")
    ev.check("fork", "fork_starts_from_parent_checkpoint_weights",
             start_weights == fmeta["extra_state"]["weights_hash"], log, parent=parent)
    ev.check("fork", "fork_continues_at_next_step",
             fk_cons[0]["global_step"] == FORK_FROM + 1
             and all(e["branch_id"] == FORK_BRANCH for e in fk_cons), log)
    ev.check("fork", "main_branch_untouched",
             fork_report["main_ledger_sha256_before"] == fork_report["main_ledger_sha256_after"],
             log)
    ev.check("fork", "fork_stream_diverges", diverged > 0, log,
             differing=diverged, of=len(fk_cons))

    # ============================================================ 9. learning trace + validation
    doc_shard = {sp.doc_id: s.shard_id for s in b.train_shards for sp in s.doc_spans}
    restore(b, os.path.join(ART, "checkpoints"), checkpoint_id(STEPS - 1))
    trace = []
    for lane in ("math", "indic", "code"):
        step = max(s for s in range(STEPS) if cons[s]["mixture_lane"] == lane
                   and cons[s]["gradient_applied"])
        batch = b.trainer.plan(step)
        for row in b.trainer.token_trace(step, tk, max_tokens=40):
            sp = next(x for x in batch.token_spans
                      if x["seq"] == row["seq"] and x["start"] <= row["position"] < x["end"])
            bt = batch.tensors[row["seq"]]
            trace.append({**row, "global_step": step, "batch_hash": batch.batch_hash,
                          "source_doc_id": sp["doc_id"], "shard_id": doc_shard[sp["doc_id"]],
                          "target_preview": tk.token_str(int(bt.labels[row["position"]]))})
    trace_path = os.path.join(ART, "ledgers", "token_trace.jsonl")
    with open(trace_path, "w", encoding="utf-8") as f:
        for r in trace:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    linked = all(l["global_step"] == c["global_step"] and l["shard_ids"] == c["shard_ids"]
                 and l["lane"] == c["mixture_lane"] for c, l in zip(cons, learn))
    trace_linked = all(r["shard_id"] in cons[r["global_step"]]["shard_ids"] for r in trace)
    trained = [l["avg_token_loss"] for l, c in zip(learn, cons) if c["gradient_applied"]]
    first, last = float(np.mean(trained[:25])), float(np.mean(trained[-25:]))

    val_ctx = PlanContext([b.validation_shard], CONFIG.seq_len, CONFIG.microbatch_size,
                          tk.vocab_size)
    val_tensors = [build_tensors(s, val_ctx.pad_id) for s in val_ctx.streams["en"].sequences]
    val_rows = []
    for cid in sorted(ck_index):
        restore(b, os.path.join(ART, "checkpoints"), cid)
        before = weights_hash(b.model)
        loss, _, _ = b.model.loss_and_grad(val_tensors)
        val_rows.append({"kind": "validation", "checkpoint_id": cid,
                         "step": ck_index[cid]["step"], "split": "validation",
                         "shard_id": b.validation_shard.shard_id, "val_loss": round(loss, 6),
                         "loss_tokens": int(sum(t.n_loss_tokens() for t in val_tensors)),
                         "weights_unchanged": weights_hash(b.model) == before})
    val_path = os.path.join(ART, "ledgers", "validation.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    lane_loss = {}
    for lane in b.lanes:
        ls = [l["avg_token_loss"] for l, c in zip(learn, cons)
              if c["mixture_lane"] == lane and c["gradient_applied"]]
        lane_loss[lane] = {"first5": round(float(np.mean(ls[:5])), 4),
                           "last5": round(float(np.mean(ls[-5:])), 4), "trained_steps": len(ls)}
    learn_path = dump(os.path.join(ART, "reports", "learning_trace.json"), {
        "mean_loss_first25_trained_steps": round(first, 4),
        "mean_loss_last25_trained_steps": round(last, 4),
        "loss_by_lane": lane_loss,
        "usefulness": dict(collections.Counter(l["usefulness"] for l in learn)),
        "validation_loss_by_checkpoint": {r["checkpoint_id"]: r["val_loss"] for r in val_rows},
    })
    ev.point("learning_trace", learn_path, "ledgers/learning.jsonl", rel(trace_path),
             rel(val_path))
    ev.check("learning_trace", "learning_events_linked_to_consumption", linked, log,
             events=len(learn))
    ev.check("learning_trace", "token_losses_traced_to_source_docs",
             bool(trace) and trace_linked, log, tokens=len(trace))
    ev.check("learning_trace", "loss_decreases_on_trained_batches", last < first, log,
             first25=round(first, 4), last25=round(last, 4))
    ev.check("learning_trace", "validation_read_only",
             all(r["weights_unchanged"] for r in val_rows), log,
             val_loss={r["checkpoint_id"]: r["val_loss"] for r in val_rows})

    # post-run firewall scan over everything that actually trained
    held_ids = set(reg.entries)
    consumed_ids = {sid for c in cons + fk_cons + rp_cons for sid in c["shard_ids"]}
    hits = 0
    for c in cons:
        batch = b.trainer.plan(c["global_step"])
        for sp in batch.token_spans:
            ids = batch.tensors[sp["seq"]].input_ids[sp["start"]:sp["end"]].tolist()
            hits += len(_ngram_fingerprints(ids) & reg._ngrams)
    ev.check("evaluation_firewall", "no_heldout_shard_or_ngram_in_any_trained_batch",
             not (held_ids & consumed_ids) and hits == 0, log,
             heldout_shards=sorted(held_ids), ngram_hits=hits,
             branches_scanned=["main", "replay", FORK_BRANCH])

    # ============================================================ 10. audit
    lo, hi = AUDIT
    rows = cons[lo:hi]
    docs_trained = collections.Counter()
    replanned_ok = True
    for c in rows:
        p = b.trainer.plan(c["global_step"])
        replanned_ok &= p.batch_hash == c["batch_hash"] and p.token_spans == c["token_spans"]
        if c["gradient_applied"]:
            for sp in c["token_spans"]:
                docs_trained[sp["doc_id"]] += sp["loss_tokens"]
    audit_path = dump(os.path.join(ART, "reports", "audit.json"), {
        "question": f"Which data trained steps [{lo}, {hi})?",
        "steps": hi - lo,
        "lanes": dict(collections.Counter(c["mixture_lane"] for c in rows)),
        "stages": dict(collections.Counter(c["curriculum_stage"] for c in rows)),
        "opus": dict(collections.Counter(c["opus_status"] for c in rows)),
        "shards": sorted({s for c in rows for s in c["shard_ids"]}),
        "loss_tokens_trained_by_doc": dict(docs_trained.most_common()),
        "every_event_reproduced_by_planner": replanned_ok,
    })
    log(f"audit completed: steps {lo}-{hi - 1} trained {len(docs_trained)} documents "
        f"from {len({s for c in rows for s in c['shard_ids']})} shards")
    ev.point("learning_trace", audit_path)

    # ============================================================ 11. performance
    perf = performance_report(meter, counts_from_ledger(read_jsonl(ref_files["consumption"])),
                              rb.ctx)
    perf_path = dump(os.path.join(ART, "performance.json"), perf)
    log(f"performance measured: {perf['tokens_per_second']['raw']} raw tok/s, "
        f"{perf['tokens_per_second']['loss_bearing']} loss-bearing tok/s, "
        f"{perf['tokens_per_second']['trained']} trained tok/s; utilization "
        f"{perf['utilization']['loss_bearing_over_raw']}")
    ev.point("throughput", perf_path)
    ev.check("throughput", "performance_counts_reconstructed_from_ledger",
             perf["reconstructable"], log, counts=perf["counts"]["loss_bearing_tokens"])
    ev.check("throughput", "main_run_counts_match_reference",
             counts_from_ledger(cons) == perf["counts"], log)
    ev.check("throughput", "packing_utilization_reported",
             0 < perf["utilization"]["loss_bearing_over_raw"] <= 1
             and set(perf["packing"]) == set(b.lanes), log,
             utilization=perf["utilization"])

    # ============================================================ evidence
    with open(os.path.join(ART, "run.log"), encoding="utf-8") as f:
        text = f.read()
    missing = [e for e in REQUIRED_LOG_EVENTS if e not in text]
    run_info = {
        "command": "python run_demo.py",
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(),
        "config_hash": CONFIG.hash(), "tokenizer_hash": tk.tokenizer_hash,
        "corpus_snapshot_sha256": snapshot_hash(),
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        "run_log_missing_events": missing,
    }
    ev.write(os.path.join(ART, "evidence.json"), os.path.join(ART, "evidence.md"), run_info)
    n_pass = sum(ev.result(r) == "PASS" for r in ev.checks)
    log(f"evidence written: {n_pass}/{len(ev.checks)} requirements PASS"
        + (f"; run.log missing {missing}" if missing else ""))
    return 0 if ev.all_passed() and not missing else 1


if __name__ == "__main__":
    sys.exit(main())

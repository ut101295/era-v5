# Dossier 03 — Immutable Shards, Content-Addressed Manifests, and Lineage

> **The question this dossier answers:** how do we store tokenized training data as
> **sealed, self-verifying objects** whose identity *is* their content, carrying enough
> metadata (tokenizer hash, provenance, dedup/contamination status, lineage) that the
> training system can later **audit** every token — and so that "modify a shard ⇒ new hash
> ⇒ new lineage" is enforced by construction? This is Session 6 §6 (tokenized shards and
> manifests). Graded block: *"Shards, manifests and tokenizer integrity — 100 pts."*

All claims below are grounded in primary docs fetched during research; URLs are inline.

---

## 1. Content addressing = immutability + dedup + self-verification

The reusable primitive across IPFS, DVC, Git, and every lakehouse is **content addressing**:
name an object by the hash of its bytes.

- **Merkle DAG (IPFS):** "each node has an identifier, and this is the result of hashing the
  node's contents" (payload + children ids). Nodes are **immutable** — "any change in a node
  would alter its identifier and thus affect all the ascendants," "essentially creating a
  different DAG." They are **self-verifying**, and "two nodes with the same CID univocally
  represent exactly the same DAG," which enables **de-duplication**. DAGs "can only be
  constructed from the leaves" (children hashes must exist before a parent can link them).
  <https://docs.ipfs.tech/concepts/merkle-dag/>
- **DVC cache** is content-addressable storage: the path is derived from the file's MD5, with
  "the first two characters … [as] the directory … and the rest … the file name" (e.g.
  `.dvc/cache/files/md5/ec/1d2935…`). Hashes are "calculated from file contents only," so
  identical content is stored once (dedup) and any content change yields a new hash. A
  directory is tracked by a `.dir` entry — "the mapping of files inside (as a JSON array),
  identified by their hash values." <https://doc.dvc.org/user-guide/project-structure/internal-files>

> **What we adopt:** a shard's identity is `content_hash = sha256(token_bytes)`. A shard set
> is a directory-manifest that lists member shards by hash (a mini `.dir`). Editing any shard
> changes its hash, which changes the set manifest's hash — immutability and lineage for free.

---

## 2. WebDataset — the sharded-tar format for structured samples

For SFT/agentic samples where one example is several related files, the assignment's §17
points at WebDataset:

- Files are "standard POSIX tar archives" readable by standard tools; **files that belong to
  one sample share the same basename** once extensions are stripped (`sample.json` +
  `sample.png` = one example; labels/metadata live in `.json`/`.txt`/`.cls`).
- Shards are "numbered like `something-000000.tar` to `something-012345.tar`," addressed with
  brace notation `something-{000000..012345}.tar`; each shard is "often ~1GB" while the full
  set "can be multiple terabytes."
- Streaming contiguous tar chunks "can be orders of magnitude faster than reading separate
  data files one by one," and WebDataset "is an implementation of PyTorch `IterableDataset`."
  <https://github.com/webdataset/webdataset> · <https://huggingface.co/docs/hub/en/datasets-webdataset>

> **What we adopt:** pretraining shards = indexed binary token arrays (Megatron-style);
> SFT/agentic shards = WebDataset-style records keyed by a common basename. The *manifest
> discipline is identical regardless of storage format* (assignment §6).

---

## 3. The shard manifest (fields + why each exists)

The assignment §6 enumerates the manifest fields; research grounds *why* each is required:

| Field | Why (grounded) |
| --- | --- |
| **content_hash** | the shard's identity; SHA-256 on every training snapshot so teams "can prove which file trained which model checkpoint after rollback" (atlan) |
| **tokenizer_hash** | token counts "depend entirely on the tokenizer" — store IDs against the exact tokenizer/version (asoasis); this hash "gives meaning to the token IDs" (assignment) |
| **source_ids / document_ids** | provenance must be **field-level**: "aggregate 'licensed' claims fail audits without source IDs and refresh timestamps" (atlan) |
| **token_count / language / script / capability_lane** | routes the shard in the mixture scheduler (Dossier 04) |
| **license / provenance_tier** | admission gate blocks unsafe licenses |
| **cleaning_pipeline_hash** | "explains how raw text became admitted training data" (assignment); Session 4 contract |
| **dedup_status** | dedup should be documented across **exact (hash) + fuzzy (MinHash/LSH) + semantic (embedding)** granularities, "across components, not only within" (toloka) |
| **contamination_status / eval_overlap_status** | the firewall gate (Dossier 05) |
| **parent_shard_ids** | lineage — a re-tokenized/re-cleaned shard points back to its ancestor |

Provenance modeling has real standards we can borrow vocabulary from:

- **W3C PROV** — provenance is "information about entities, activities, and people (agents)
  involved in producing" data. <https://www.w3.org/TR/prov-dm/>
- **OpenLineage** — Dataset / Job / **Run** (Run id "must be a UUID"); emit at least START and
  COMPLETE/FAIL/ABORT per run; metadata is additive via **facets** (a new facet with the same
  name replaces the old). <https://github.com/OpenLineage/OpenLineage/blob/main/spec/OpenLineage.md>
- **ML Metadata (TFX)** — Artifacts / Executions / Contexts, with **Events** recording which
  artifacts an execution *used* and *produced*, so "MLMD can … recurse back from any artifact
  to all of its upstream inputs." <https://www.tensorflow.org/tfx/guide/mlmd>
- **Datasheets for Datasets** (Gebru et al.) — every dataset ships a datasheet of motivation /
  composition / collection / recommended uses. <https://arxiv.org/abs/1803.09010>

A record-/token-level provenance research direction (arXiv 2607.13037) notes that most
systems "operate at file or dataset level, forcing catastrophic over-deletion," and that
attribution "is lost during tokenization and packing" unless an **independent token-index
layer** records source attribution — "what is not recorded does not exist." This is precisely
the token-span-id idea we carry into the ledger (Dossier 04).

---

## 4. Immutability discipline (the rule we enforce)

Grounded best practice: "push shards plus manifest to an immutable bucket, tag with a
semantic version, and register the snapshot in a data-versioning system such as DVC or
LakeFS"; **"never mutate released shards; append new versions instead."** Every training
snapshot cited in a model card must carry a SHA-256, or "teams cannot prove which file
trained which model checkpoint after rollback." (asoasis / atlan — industry/vendor guides.)

> **What we adopt (assignment §6):** "Once a shard is admitted, it should behave like a
> sealed object. If we modify it, it becomes a new shard with a new hash and a new lineage."
> Our admission gate **blocks** shards with unsafe license, missing tokenizer hash, unknown
> cleaning lineage, or eval overlap — and writes a rejection event either way.

---

## 5. Minimal manifest schema (what we will emit)

```json
{
  "shard_id": "shard-000007",
  "content_hash": "sha256:9f3a…",           // identity: sha256 of the token byte array
  "tokenizer_hash": "sha256:c1d2…",         // frozen Session-2 tokenizer.json hash
  "storage_format": "indexed_bin | webdataset_tar",
  "source_ids": ["s3-asm-wiki", "…"],
  "document_ids": ["doc-00123", "…"],
  "token_count": 262144,
  "language": "asm", "script": "Beng",
  "capability_lane": "indic",
  "license": "CC-BY-SA-4.0", "provenance_tier": "verified-native",
  "cleaning_pipeline_hash": "sha256:aa11…",  // Session-4 pipeline hash
  "dedup_status": {"exact": true, "minhash_lsh": true, "semantic": false},
  "contamination_status": "clean",           // clean | flagged | blocked
  "eval_overlap_status": "none",             // none | val | test  (Dossier 05)
  "split": "train | validation | test",
  "parent_shard_ids": [],                    // lineage; [] for a freshly built shard
  "created_by_run": "uuid-…", "created_at_step": null
}
```

**Admission gate (pure function → PASS/FAIL + reason):**
```
admit(manifest):
    assert manifest.tokenizer_hash == FROZEN_TOKENIZER_HASH      # else FAIL missing/wrong tokenizer
    assert manifest.content_hash == sha256(load(shard))          # else FAIL corrupted/mutated
    assert manifest.license in ALLOWED_LICENSES                  # else FAIL unsafe license
    assert manifest.cleaning_pipeline_hash in KNOWN_LINEAGE      # else FAIL unknown lineage
    assert manifest.eval_overlap_status == "none"               # else BLOCK (firewall)
    -> [PASS] shard admitted  |  [BLOCK] + rejection event
```

---

## 6. Minimal implementation recipe

1. **`build_shard(docs, tokenizer)`** → tokenize with the frozen tokenizer, write immutable
   token bytes, compute `content_hash`, emit the manifest JSON above.
2. **`verify_shard(manifest)`** → recompute `content_hash` and `tokenizer_hash`; any mismatch
   fails `[PASS] tokenizer_hash_verified`.
3. **`admit(manifest)`** → the gate above; log `shards created` / `manifests validated` /
   `evaluation data blocked` to `run.log`.
4. **Shard-set manifest** = a `.dir`-style JSON listing member shards by hash → its own hash
   is the *dataset version*. Editing one shard changes the set hash (lineage propagation).
5. All manifests written under `submission_artifacts/manifests/`, machine-readable, so the
   grader can reconstruct every claim (assignment: "Hardcoded evidence will not be accepted").

---

## 7. Sources

- IPFS Merkle DAG — <https://docs.ipfs.tech/concepts/merkle-dag/>
- DVC internal files (CAS, `.dir`) — <https://doc.dvc.org/user-guide/project-structure/internal-files>
- WebDataset — <https://github.com/webdataset/webdataset> · <https://huggingface.co/docs/hub/en/datasets-webdataset>
- W3C PROV — <https://www.w3.org/TR/prov-dm/>
- OpenLineage spec — <https://github.com/OpenLineage/OpenLineage/blob/main/spec/OpenLineage.md>
- ML Metadata (TFX) — <https://www.tensorflow.org/tfx/guide/mlmd>
- Datasheets for Datasets — <https://arxiv.org/abs/1803.09010>
- Record-/token-level provenance (preprint) — <https://arxiv.org/html/2607.13037v1>
- (industry/vendor, secondary) manifest & lineage practice — asoasis.tech, atlan.com, toloka.ai

"""
Stages 5 & 6 — Deduplication: the mechanism (MinHash + LSH) and at scale (global).

The naive picture — drop exact duplicates — misses almost all real duplication,
which is *near* duplicates: the same article with a different header, the same page
crawled twice with different boilerplate. We catch those in three moves:

  1. Shingling   — each doc becomes a set of overlapping word 5-grams.
  2. MinHash     — replace each set with a fixed-length signature of minimum hash
                   values; the chance two signatures agree in a slot equals the true
                   Jaccard overlap of the sets.
  3. LSH banding — split each signature into bands and only compare docs that match
                   in at least one band, so we never compare all-pairs.

Parameters follow the FineWeb recipe cited in Session 3: 112 hashes, 14 bands × 8
rows (implied similarity threshold ≈ (1/14)^(1/8) ≈ 0.72).

Global (Stage 6): duplication is GLOBAL. We assign each doc a pretend "student
shard", dedup each shard locally, then run one global pass and report how many
duplicates were cross-shard — the ones every local pass necessarily missed.
"""
import random
import zlib

import numpy as np

NUM_PERM = 112
BANDS = 14
ROWS = 8               # BANDS * ROWS == NUM_PERM
SHINGLE_N = 5
MERSENNE = (1 << 61) - 1
N_SHARDS = 8           # pretend student shards, for the global-dedup story

assert BANDS * ROWS == NUM_PERM


def _permutations(seed=42):
    rng = random.Random(seed)
    a = np.array([rng.randrange(1, MERSENNE) for _ in range(NUM_PERM)], dtype=np.uint64)
    b = np.array([rng.randrange(0, MERSENNE) for _ in range(NUM_PERM)], dtype=np.uint64)
    return a, b


_A, _B = _permutations()


def _shingle_hashes(tokens):
    """32-bit base hashes of the doc's word 5-grams (fallback: the token set)."""
    if len(tokens) >= SHINGLE_N:
        shingles = {" ".join(tokens[i:i + SHINGLE_N])
                    for i in range(len(tokens) - SHINGLE_N + 1)}
    else:
        shingles = set(tokens) or {""}
    # zlib.crc32 is a STABLE hash (Python's built-in hash() is per-process
    # randomized, which would make dedup — and therefore the corpus — irreproducible).
    return np.array([zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF for s in shingles],
                    dtype=np.uint64)


def signature(tokens):
    """MinHash signature (NUM_PERM,) for a tokenized document."""
    h = _shingle_hashes(tokens)
    # (NUM_PERM, n_shingles): (a*h + b) mod Mersenne prime, then min over shingles
    m = (np.outer(_A, h) + _B[:, None]) % MERSENNE
    return m.min(axis=1).astype(np.uint64)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)  # keep the lowest index as survivor


def dedup(sigs, shard_ids):
    """
    Cluster near-duplicate docs by LSH over their signatures.

    Returns:
      keep_mask   — bool list, True for the survivor of each cluster
      n_clusters_dup — number of clusters with >1 member
      cross_shard_dups — duplicate docs whose cluster spans >1 shard
      local_dups       — duplicate docs whose cluster is within a single shard
    """
    n = len(sigs)
    uf = UnionFind(n)
    sig_arr = np.array(sigs, dtype=np.uint64)  # (n, NUM_PERM)

    for band in range(BANDS):
        lo, hi = band * ROWS, (band + 1) * ROWS
        buckets = {}
        for i in range(n):
            key = sig_arr[i, lo:hi].tobytes()
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            if len(members) > 1:
                first = members[0]
                for other in members[1:]:
                    uf.union(first, other)

    # roll up clusters
    clusters = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    keep_mask = [False] * n
    n_clusters_dup = 0
    removed = 0
    cross_shard_dups = 0
    local_dups = 0
    for root, members in clusters.items():
        survivor = min(members)
        keep_mask[survivor] = True
        if len(members) > 1:
            n_clusters_dup += 1
            dup_members = [m for m in members if m != survivor]
            removed += len(dup_members)
            shards = {shard_ids[m] for m in members}
            if len(shards) > 1:
                cross_shard_dups += len(dup_members)
            else:
                local_dups += len(dup_members)

    return {
        "keep_mask": keep_mask,
        "clusters_with_dups": n_clusters_dup,
        "removed": removed,
        "cross_shard_dups": cross_shard_dups,
        "local_dups": local_dups,
        "shards": N_SHARDS,
        "index_bytes_per_doc": NUM_PERM * 8,
    }

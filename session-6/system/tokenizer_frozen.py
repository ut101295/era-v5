"""Frozen byte-level BPE tokenizer (pure Python, stdlib only).

This is a faithful Python port of Session 2's browser widget BPE
(`session-2/tokenizer-widget/index.html`), which is itself a verified port of
HuggingFace's byte-level BPE merge order. Porting it here keeps the whole
Session 6 system dependency-free (numpy + stdlib only) while honoring the
**Session 2 tokenizer contract** (research/02, research/03):

  * same raw text -> same token IDs, always;
  * NFC normalization (Indic-safe), byte->visible-Unicode mapping;
  * `use_regex=False`: BPE runs over the whole text, so Indic matras/joiners are
    not split before merges can form (the key Session 2 finding).

The tokenizer is **frozen**: we load `tokenizer.json`, hash its bytes, and the
`tokenizer_hash` becomes the thing that "gives meaning to the token IDs" in
every shard manifest and ledger event. If the file changes, the hash changes,
and every downstream object's lineage changes with it.

Algorithm details ported verbatim from the widget:
  * GPT-2 byte->unicode table;
  * leftmost-wins tie-break at equal merge rank, via a versioned linked-list +
    heap (identical to HF's `merge` loop ordering).
"""
from __future__ import annotations

import heapq
import json
import os
import unicodedata
from typing import Dict, List, Tuple

from .hashing import sha256_bytes

# Default location of the frozen Session 2 tokenizer.
_DEFAULT_TOKENIZER_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "session-2",
        "multilingual-tokenizer",
        "output",
        "tokenizer.json",
    )
)


def _byte_to_unicode() -> Dict[int, str]:
    """GPT-2 / HF byte-level reversible byte->unicode map (256 entries).

    Printable ASCII-ish ranges map to themselves; the rest map to code points
    starting at 256, so every byte becomes a single visible character with no
    whitespace/control ambiguity.
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class FrozenTokenizer:
    """Immutable byte-level BPE tokenizer loaded from a frozen tokenizer.json."""

    def __init__(self, path: str | None = None):
        self.path = path or _DEFAULT_TOKENIZER_PATH
        with open(self.path, "rb") as f:
            raw = f.read()
        # The tokenizer hash is over the exact on-disk bytes -> this is the
        # "frozen tokenizer" identity used everywhere downstream.
        self.tokenizer_hash = sha256_bytes(raw)
        tk = json.loads(raw.decode("utf-8"))

        model = tk["model"]
        self.vocab: Dict[str, int] = model["vocab"]
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # merges: list of pairs -> rank (lower rank = earlier/preferred merge).
        self.rank: Dict[Tuple[str, str], int] = {}
        for i, m in enumerate(model["merges"]):
            pair = tuple(m) if isinstance(m, list) else tuple(m.split(" "))
            self.rank[(pair[0], pair[1])] = i

        self.unk_token = model.get("unk_token", "<unk>")
        self.unk_id = self.vocab.get(self.unk_token, 0)

        self._b2u = _byte_to_unicode()
        self._u2b = {u: b for b, u in self._b2u.items()}

        # Special-token ids (from added_tokens), used for boundary/EOS handling.
        self.special_ids = {t["id"] for t in tk.get("added_tokens", [])}
        # There is no dedicated EOS in the Session 2 tokenizer; the packer uses a
        # reserved logical EOS marker (see packing.py). We expose unk for masks.

    # ------------------------------------------------------------------ encode
    def _text_to_symbols(self, text: str) -> List[str]:
        """NFC-normalize, UTF-8 encode, map each byte to its visible-Unicode char."""
        data = unicodedata.normalize("NFC", text).encode("utf-8")
        return [self._b2u[b] for b in data]

    def _bpe(self, symbols: List[str]) -> List[str]:
        """Rank-ordered merges with leftmost tie-break (verbatim widget port).

        Uses a doubly-linked list over the symbol array plus a heap keyed on
        (rank, version, index). The `version` counter invalidates stale heap
        entries; ties at equal rank are broken by the lower index, i.e.
        leftmost-first -- exactly HF's behavior.
        """
        n = len(symbols)
        if n <= 1:
            return symbols
        val = symbols[:]
        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 if i + 1 < n else -1 for i in range(n)]
        ver = [0] * n
        alive = [True] * n
        heap: List[Tuple[int, int, int]] = []

        def add(i: int) -> None:
            j = nxt[i]
            if j == -1:
                return
            r = self.rank.get((val[i], val[j]))
            if r is not None:
                heapq.heappush(heap, (r, ver[i], i))

        for i in range(n):
            add(i)

        while heap:
            r, seq, i = heapq.heappop(heap)
            if not alive[i] or ver[i] != seq:
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            if self.rank.get((val[i], val[j])) != r:
                continue
            # merge j into i
            val[i] = val[i] + val[j]
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k != -1:
                prev[k] = i
            ver[i] += 1
            if prev[i] != -1:
                ver[prev[i]] += 1
                add(prev[i])
            add(i)

        out = []
        i = 0
        while i != -1 and i < n:
            if alive[i]:
                out.append(val[i])
            i = nxt[i]
        return out

    def encode(self, text: str) -> List[int]:
        """Text -> list of token IDs."""
        symbols = self._bpe(self._text_to_symbols(text))
        return [self.vocab.get(s, self.unk_id) for s in symbols]

    def encode_tokens(self, text: str) -> List[str]:
        """Text -> list of token strings (visible-Unicode form). For debugging."""
        return self._bpe(self._text_to_symbols(text))

    # ------------------------------------------------------------------ decode
    def decode(self, ids: List[int]) -> str:
        """Token IDs -> text. Roundtrip is exact except newlines are dropped on
        decode (a known Session 2 property that does not affect correctness of
        IDs or scores)."""
        pieces = "".join(self.id_to_token.get(i, "") for i in ids)
        data = bytes(self._u2b[ch] for ch in pieces if ch in self._u2b)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ misc
    def token_str(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, self.unk_token)


_SINGLETON: FrozenTokenizer | None = None


def load_frozen_tokenizer(path: str | None = None) -> FrozenTokenizer:
    """Load (and cache) the frozen tokenizer singleton."""
    global _SINGLETON
    if _SINGLETON is None or path is not None:
        _SINGLETON = FrozenTokenizer(path)
    return _SINGLETON

"""
Byte-position embeddings for Session 7, Problem #3 — dynamic word length.

Two codecs live here. Both turn a token's raw UTF-8 bytes into a fixed-length
"code" vector; a single shared trainable projection then maps that code to the
model width. Neither codec stores a row per token, which is the whole point of
Kronecker embeddings (lecture notes sec. 7).

----------------------------------------------------------------------------
BASELINE (the course construction, notes sec. 7-8):

    kappa(b) = (1/sqrt(L)) * vec( sum_p  onehot_256(byte_p) (x) onehot_32(pos_p) )

    * a 256 x 32 grid: 256 possible byte values, 32 byte POSITIONS.
    * byte at position p sets cell (value, p).
    * flatten -> 8192-dim code -> shared projection (8192 x d_model).

    THE WALL: 32 position columns => only the first 32 BYTES of a token are
    ever seen. Indic scripts cost ~3 bytes/char (conjuncts up to 9 bytes), so
    long words overflow the window and COLLIDE SILENTLY: two different words
    that agree on their first 32 bytes get the identical code forever.

----------------------------------------------------------------------------
OURS — FUNCTIONAL-POSITION KRONECKER (Problem #3 solution):

    kappa(b) = (1/sqrt(L)) * vec( sum_p  onehot_256(byte_p) (x) phi(p) )

    Replace the discrete onehot_32(p) position factor with a CONTINUOUS
    position function phi(p) in R^K (sinusoidal / Fourier features) that is
    defined for EVERY p. Consequences:

      * No wall     - phi(p) exists for all p, so a 200-byte word is fully
                      encoded. Nothing is cropped.
      * Less waste  - K can be smaller than 32 (default 16) because continuous
                      codes pack position densely => FEWER projection params
                      than the baseline, not more.
      * Course-aligned - this is literally the lecture's own prescription
                      (notes sec. 11: "stop storing position and start
                      computing it"), applied to the byte-position factor.

The code row for byte value v becomes  sum over positions p where byte_p == v
of phi(p).  Distinct byte sequences therefore map to distinct codes at any
length (verified empirically in collision_test.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

BYTE_DIM = 256  # one row per possible byte value, for both codecs


def utf8_bytes(token: str) -> list[int]:
    """The token's raw UTF-8 bytes — the only input either codec ever reads."""
    return list(token.encode("utf-8"))


class BaselineKroneckerCodec:
    """Fixed (non-trainable) byte->code codec, exactly as the course defines it.

    Reproduces the 32-byte wall on purpose so we have an honest baseline to beat.
    """

    def __init__(self, pos_dim: int = 32):
        self.pos_dim = pos_dim
        self.code_dim = BYTE_DIM * pos_dim  # 256 * 32 = 8192

    def encode_one(self, token: str) -> torch.Tensor:
        b = utf8_bytes(token)
        grid = torch.zeros(BYTE_DIM, self.pos_dim)
        placed = 0
        for p, val in enumerate(b):
            if p >= self.pos_dim:  # <-- THE WALL: bytes past column 32 are dropped
                break
            grid[val, p] = 1.0
            placed += 1
        # L = number of bytes actually placed. Two tokens that share their first
        # pos_dim bytes thus share L and share the grid => identical code.
        grid = grid / math.sqrt(max(placed, 1))
        return grid.reshape(-1)

    def encode(self, tokens: list[str]) -> torch.Tensor:
        return torch.stack([self.encode_one(t) for t in tokens])


class FunctionalPositionCodec:
    """Our dynamic-length codec: continuous sinusoidal position factor phi(p).

    phi(p) is defined for every byte position p, so there is no maximum word
    length and nothing is ever cropped. Still fixed (non-trainable) and still
    vocabulary-free — only the shared projection on top carries parameters.
    """

    def __init__(self, pos_dim: int = 16, base: float = 10000.0):
        assert pos_dim % 2 == 0, "pos_dim must be even for sin/cos pairs"
        self.pos_dim = pos_dim
        self.base = base
        self.code_dim = BYTE_DIM * pos_dim  # 256 * 16 = 4096  (half the baseline)

    def phi(self, p: int) -> torch.Tensor:
        """Sinusoidal position features, the Transformer's original encoding."""
        k = torch.arange(0, self.pos_dim, 2, dtype=torch.float32)
        div = torch.exp(-math.log(self.base) * k / self.pos_dim)
        out = torch.zeros(self.pos_dim)
        out[0::2] = torch.sin(p * div)
        out[1::2] = torch.cos(p * div)
        return out

    def encode_one(self, token: str) -> torch.Tensor:
        b = utf8_bytes(token)
        grid = torch.zeros(BYTE_DIM, self.pos_dim)
        for p, val in enumerate(b):  # no break: EVERY byte is encoded
            grid[val] += self.phi(p)  # row for this byte value accumulates phi(p)
        grid = grid / math.sqrt(max(len(b), 1))
        return grid.reshape(-1)

    def encode(self, tokens: list[str]) -> torch.Tensor:
        return torch.stack([self.encode_one(t) for t in tokens])


class KroneckerEmbedding(nn.Module):
    """A codec (fixed) + one shared trainable projection to the model width.

    The codec holds zero parameters; all learning happens in `proj`. Precomputes
    and caches codes per unique token so training does not re-encode every step.
    """

    def __init__(self, codec, d_model: int):
        super().__init__()
        self.codec = codec
        self.proj = nn.Linear(codec.code_dim, d_model, bias=False)
        self._cache: dict[str, torch.Tensor] = {}

    @property
    def projection_params(self) -> int:
        return self.codec.code_dim * self.proj.out_features

    def code_for(self, tokens: list[str]) -> torch.Tensor:
        missing = [t for t in tokens if t not in self._cache]
        if missing:
            codes = self.codec.encode(missing)
            for t, c in zip(missing, codes):
                self._cache[t] = c
        return torch.stack([self._cache[t] for t in tokens])

    def forward(self, tokens: list[str]) -> torch.Tensor:
        codes = self.code_for(tokens).to(self.proj.weight.device)
        return self.proj(codes)

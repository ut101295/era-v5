"""
A deliberately small causal Transformer, used only to PROVE the embedding works.

The model never sees token ids. It is handed per-token vectors produced by a
KroneckerEmbedding (baseline or functional), adds a learned sequence-position
embedding, runs a few pre-norm causal self-attention blocks, and predicts a
class at the final position. Everything except the embedding is identical across
the two runs, so any difference in accuracy is attributable to the input codec.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):  # x: [B, T, d]
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyCausalTransformer(nn.Module):
    """Input: token vectors [B, T, d]. Output: class logits at the last position."""

    def __init__(self, d_model: int, n_classes: int, max_T: int = 8,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.pos = nn.Embedding(max_T, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, token_vecs):  # [B, T, d]
        B, T, d = token_vecs.shape
        x = token_vecs + self.pos(torch.arange(T, device=token_vecs.device))[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x[:, -1, :])  # predict from the last (causal) position

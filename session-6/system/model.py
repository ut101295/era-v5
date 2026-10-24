"""A tiny, real, gradient-trained token LM in pure numpy.

The goal is NOT model quality -- it is to produce *genuine* per-token
cross-entropy, perplexity, loss deltas, and gradient norms so the learning ledger
(research/04 §5) links real learning signal back to source data. "Hardcoded
evidence will not be accepted", so every number the ledger reports is measured
here.

Architecture -- a masked causal *mean-context* LM. For each position i:

    emb_i   = TokenEmb[input_ids_i] + PosEmb[position_ids_i]
    ctx_i   = mean of emb_j over j allowed by the attention mask (causal AND
              same-document) -- so the block-diagonal mask is *functionally*
              consumed: context never crosses a document boundary.
    h_i     = tanh([emb_i ; ctx_i] @ W1 + b1)
    logits_i= h_i @ Wout + bout

This deliberately uses `attention_mask` and `position_ids` (not just token ids),
so the masks from M2 actually shape learning, while keeping backprop exact and
robust (the context averaging matrix is constant w.r.t. parameters). Loss is
cross-entropy over the `loss_mask` positions only.

Includes an Adam optimizer with `state_dict`/`load_state_dict` so a checkpoint
can restore the *exact* optimizer state (research/01 §3) -- required for true
resume.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .masks import BatchTensors, IGNORE_INDEX


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TinyLM:
    def __init__(self, vocab_size: int, dim: int = 24, hidden: int = 32,
                 max_pos: int = 256, seed: int = 0):
        self.V = vocab_size
        self.d = dim
        self.h = hidden
        self.max_pos = max_pos
        rng = np.random.default_rng(seed)
        s = 0.02
        self.params: Dict[str, np.ndarray] = {
            "E": rng.normal(0, s, (vocab_size, dim)).astype(np.float64),
            "P": rng.normal(0, s, (max_pos, dim)).astype(np.float64),
            "W1": rng.normal(0, s, (2 * dim, hidden)).astype(np.float64),
            "b1": np.zeros(hidden),
            "Wout": rng.normal(0, s, (hidden, vocab_size)).astype(np.float64),
            "bout": np.zeros(vocab_size),
        }

    # ------------------------------------------------------------- forward
    def _forward_seq(self, input_ids, position_ids, attn_mask):
        p = self.params
        pos = np.clip(position_ids, 0, self.max_pos - 1)
        emb = p["E"][input_ids] + p["P"][pos]              # [L,d]
        # row-normalized allowed-context averaging matrix A (constant wrt params)
        A = attn_mask.astype(np.float64)
        rowsum = A.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        A = A / rowsum
        ctx = A @ emb                                       # [L,d]
        X = np.concatenate([emb, ctx], axis=1)             # [L,2d]
        z = X @ p["W1"] + p["b1"]                          # [L,h]
        a = np.tanh(z)
        logits = a @ p["Wout"] + p["bout"]                 # [L,V]
        cache = (input_ids, pos, emb, A, X, a, logits)
        return logits, cache

    # ------------------------------------------------- loss + grad (microbatch)
    def loss_and_grad(self, batch: List[BatchTensors]) -> Tuple[float, List[np.ndarray], Dict[str, np.ndarray]]:
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        total_loss = 0.0
        total_tok = 0
        per_seq_token_losses: List[np.ndarray] = []

        for bt in batch:
            logits, cache = self._forward_seq(bt.input_ids, bt.position_ids, bt.attention_mask)
            input_ids, pos, emb, A, X, a, _ = cache
            L = logits.shape[0]
            probs = _softmax_rows(logits)                  # [L,V]

            mask = bt.loss_mask
            labels = bt.labels
            idx = np.where(mask)[0]
            tok_loss = np.zeros(L)
            if idx.size:
                p_true = probs[idx, labels[idx]]
                p_true = np.clip(p_true, 1e-12, 1.0)
                ce = -np.log(p_true)
                tok_loss[idx] = ce
                total_loss += float(ce.sum())
                total_tok += int(idx.size)

            per_seq_token_losses.append(tok_loss)

            # ---- backward (only masked positions contribute) ----
            dlogits = np.zeros_like(logits)
            if idx.size:
                dlogits[idx] = probs[idx]
                dlogits[idx, labels[idx]] -= 1.0
                # average over all masked tokens in the whole microbatch later;
                # here accumulate raw, divide at the end.
            grads["Wout"] += a.T @ dlogits
            grads["bout"] += dlogits.sum(axis=0)
            da = dlogits @ self.params["Wout"].T           # [L,h]
            dz = da * (1.0 - a * a)
            grads["W1"] += X.T @ dz
            grads["b1"] += dz.sum(axis=0)
            dX = dz @ self.params["W1"].T                  # [L,2d]
            demb = dX[:, : self.d].copy()
            dctx = dX[:, self.d:]
            demb += A.T @ dctx                              # ctx = A @ emb
            # scatter into E and P
            np.add.at(grads["E"], input_ids, demb)
            np.add.at(grads["P"], pos, demb)

        denom = max(1, total_tok)
        for k in grads:
            grads[k] /= denom
        mean_loss = total_loss / denom
        return mean_loss, per_seq_token_losses, grads

    def token_perplexities(self, bt: BatchTensors) -> np.ndarray:
        """Per-token perplexity over a single sequence (0 where no loss)."""
        logits, _ = self._forward_seq(bt.input_ids, bt.position_ids, bt.attention_mask)
        probs = _softmax_rows(logits)
        L = logits.shape[0]
        ppl = np.zeros(L)
        idx = np.where(bt.loss_mask)[0]
        if idx.size:
            p_true = np.clip(probs[idx, bt.labels[idx]], 1e-12, 1.0)
            ppl[idx] = np.exp(-np.log(p_true))
        return ppl

    # ------------------------------------------------------------- state
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        for k, v in sd.items():
            self.params[k] = np.asarray(v).copy()

    def grad_global_norm(self, grads: Dict[str, np.ndarray]) -> float:
        return float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))


class Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr: float = 0.05,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def state_dict(self) -> Dict:
        return {"t": self.t,
                "m": {k: v.copy() for k, v in self.m.items()},
                "v": {k: v.copy() for k, v in self.v.items()}}

    def load_state_dict(self, sd: Dict) -> None:
        self.t = int(sd["t"])
        self.m = {k: np.asarray(v).copy() for k, v in sd["m"].items()}
        self.v = {k: np.asarray(v).copy() for k, v in sd["v"].items()}

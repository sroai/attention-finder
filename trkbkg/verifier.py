"""Learned window verifier (phase 6b): is THIS candidate window the burst?

Replaces the 11 hand-pooled features of trkbkg/reranker.py with learned
pooling over the window's raw hits. Why this should beat the hand features
(expectation recorded per protocol): the selector plateaued at the feature
set, not at selector capacity (measured, scratchpad selector_iter*.py), and
it trains on candidate windows — i.e. on HARD NEGATIVES (background bursts
the finder ranked high), which watt never saw in its training.

Inputs per hit: x, y, z, dt (t - candidate window start), r (derived), the
watt_r2 sigmoid score (the transferring hit-level knowledge, precomputed at
dataset build), and a system-id embedding. Windows are subsampled to at
most MAX_HITS hits; the true hit count enters the window-level head as
log1p(n_hits) together with the finder's logit and candidate rank.

Architecture mirrors WindowNet (normalization baked in, manual QKV -> ONNX
friendly), with masked attention + masked mean-pool for batched
variable-length windows. n_blocks=0 is the DeepSets baseline — train it
alongside to measure what attention itself buys here.
"""

import numpy as np
import torch
from torch import nn

MAX_HITS = 512
N_SCALARS = 3  # log1p(n_hits_total), finder logit, candidate rank


class VerifierBlock(nn.Module):
    """One self-attention + FFN block with key-padding mask."""

    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.d, self.h = d, n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                nn.Linear(2 * d, d))

    def forward(self, x, pad):  # x [B,N,d], pad [B,N] True=padding
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        hs = d // self.h

        def split(t):
            return t.view(B, N, self.h, hs).transpose(1, 2)  # [B,h,N,hs]

        att = (split(q) @ split(k).transpose(-2, -1)) / hs ** 0.5
        att = att.masked_fill(pad[:, None, None, :], -1e9)
        out = (att.softmax(-1) @ split(v)).transpose(1, 2).reshape(B, N, d)
        x = x + self.proj(out)
        return x + self.ff(self.n2(x))


class WindowVerifier(nn.Module):
    def __init__(self, mean, std, n_systems: int, d: int = 64,
                 n_blocks: int = 2, sys_dim: int = 8):
        super().__init__()
        # mean/std over (x, y, z, dt, r, watt_score)
        self.register_buffer("mu", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(std, dtype=torch.float32))
        self.sys_emb = nn.Embedding(n_systems, sys_dim)
        self.enc = nn.Sequential(nn.Linear(6 + sys_dim, d), nn.GELU(),
                                 nn.Linear(d, d))
        self.blocks = nn.ModuleList(VerifierBlock(d) for _ in range(n_blocks))
        self.head = nn.Sequential(nn.Linear(d + N_SCALARS, 64), nn.GELU(),
                                  nn.Linear(64, 1))

    def forward(self, X, sysi, pad, scalars):
        """X [B,N,6] (x,y,z,dt,r,watt), sysi [B,N], pad [B,N] True=pad,
        scalars [B,3] -> window logits [B]."""
        h = self.enc(torch.cat([(X - self.mu) / self.sd,
                                self.sys_emb(sysi)], dim=-1))
        for blk in self.blocks:
            h = blk(h, pad)
        w = (~pad).float().unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        return self.head(torch.cat([pooled, scalars], dim=1)).squeeze(1)


def pack_windows(hits_list):
    """Pad a list of [n_i, 7] arrays (6 feats + sys idx) into batch tensors."""
    B = len(hits_list)
    N = max(1, max(len(h) for h in hits_list))
    X = np.zeros((B, N, 6), np.float32)
    S = np.zeros((B, N), np.int64)
    pad = np.ones((B, N), bool)
    for i, h in enumerate(hits_list):
        n = len(h)
        X[i, :n] = h[:, :6]
        S[i, :n] = h[:, 6]
        pad[i, :n] = False
    return (torch.from_numpy(X), torch.from_numpy(S), torch.from_numpy(pad))

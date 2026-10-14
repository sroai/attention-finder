"""Phase 4: per-hit classification INSIDE the found window, with attention.

Pipeline position: the phase-3 window finder gives t0_est per frame. Hits
outside [t0_est - front, t0_est + back] are dropped as background by the
window cut itself; this module classifies the hits INSIDE, where the real
contest is signal tracks vs the synrad haze + shower fragments that fell
into the same time slice.

Why attention here (and why it was hopeless in phase 1): a signal hit is not
special on its own — it is special because ~30 OTHER hits form consistent
tracks with it (same burst, radially ordered times, common origin). Attention
lets every hit exchange information with every other hit in the window, so
"do my neighbours vouch for me?" becomes a learnable feature. Inside a
~350 ns window that is ~1000 hits -> a 1000x1000 attention matrix, cheap on
CPU; over the whole 2000 ns frame it would be 7000x7000 and mostly
background-on-background — that is what the window buys us.

Inputs per hit (deployment convention, mirrors HitMLP):
    hits   float32 [N, 4]: x, y, z [mm], dt [ns]  (dt = t - t0_est,
           computed by the caller — one subtraction in C++)
    system int64 [N]: dense system index
Output: background logit per hit (higher = filter), same as phase 1.

Two models share the encoder/head so the comparison is honest:
  * WindowMLP  — per-hit only (the "is dt alone enough?" baseline)
  * WindowAttention — + `n_blocks` self-attention blocks between encoder
    and head. Manual single-head QKV (Linear + matmul + softmax) rather than
    nn.MultiheadAttention: exports to clean ONNX with dynamic N and is easy
    to read.
"""

import torch
from torch import nn


class _Encoder(nn.Module):
    """Raw hit -> d-dim embedding (normalization baked in as buffers)."""

    def __init__(self, norm_mean, norm_std, n_systems, emb_dim, d):
        super().__init__()
        self.register_buffer("norm_mean", torch.tensor(norm_mean, dtype=torch.float32))
        self.register_buffer("norm_std", torch.tensor(norm_std, dtype=torch.float32))
        self.sys_emb = nn.Embedding(n_systems, emb_dim)
        self.proj = nn.Sequential(nn.Linear(5 + emb_dim, d), nn.ReLU(),
                                  nn.Linear(d, d))

    def forward(self, x, sys_idx):
        r = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2).unsqueeze(-1)
        f = (torch.cat([x, r], dim=-1) - self.norm_mean) / self.norm_std
        return self.proj(torch.cat([f, self.sys_emb(sys_idx)], dim=-1))


class _AttentionBlock(nn.Module):
    """Pre-norm single-head self-attention + FFN. Shapes [..., N, d]."""

    def __init__(self, d):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(),
                                 nn.Linear(2 * d, d))
        self.scale = d ** -0.5

    def forward(self, h, pad_mask=None):
        """pad_mask: [..., N] True where PADDING (ignored as keys)."""
        x = self.norm1(h)
        att = torch.matmul(self.q(x), self.k(x).transpose(-2, -1)) * self.scale
        if pad_mask is not None:
            att = att.masked_fill(pad_mask.unsqueeze(-2), float("-inf"))
        h = h + self.out(torch.matmul(att.softmax(dim=-1), self.v(x)))
        return h + self.ffn(self.norm2(h))


class WindowNet(nn.Module):
    """Encoder [+ attention blocks] + per-hit head. n_blocks=0 => WindowMLP."""

    def __init__(self, norm_mean, norm_std, n_systems, emb_dim=8, d=64,
                 n_blocks=2):
        super().__init__()
        self.encoder = _Encoder(norm_mean, norm_std, n_systems, emb_dim, d)
        self.blocks = nn.ModuleList(_AttentionBlock(d) for _ in range(n_blocks))
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, x, sys_idx):
        """Single window: x [N,4], sys [N] -> logits [N]. ONNX export path."""
        h = self.encoder(x, sys_idx)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)

    @torch.no_grad()
    def forward_with_attention(self, x, sys_idx):
        """Single window, but keep the attention matrices (analysis only).

        Same maths as forward(), except each block's softmax matrix is saved
        instead of being consumed. Returns (logits [N], atts list of [N, N]),
        where atts[b][i, j] = how much hit i listened to hit j in block b
        (rows sum to 1). Not used at inference; ONNX export never sees it.
        """
        h = self.encoder(x, sys_idx)
        atts = []
        for blk in self.blocks:
            xn = blk.norm1(h)
            a = torch.matmul(blk.q(xn), blk.k(xn).transpose(-2, -1)) * blk.scale
            a = a.softmax(dim=-1)
            atts.append(a)
            h = h + blk.out(torch.matmul(a, blk.v(xn)))
            h = h + blk.ffn(blk.norm2(h))
        return self.head(h).squeeze(-1), atts

    def forward_padded(self, x, sys_idx, pad_mask):
        """Training path: x [B,N,4], sys [B,N], pad_mask [B,N] (True=pad).
        Returns logits [B,N]; caller masks the loss with ~pad_mask."""
        h = self.encoder(x, sys_idx)
        for blk in self.blocks:
            h = blk(h, pad_mask)
        return self.head(h).squeeze(-1)

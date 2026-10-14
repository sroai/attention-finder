"""Candidate-window reranking: features + selector (phase 5/6b).

The window finder's argmax misses ~12% of frames, but the true bin is in its
top-10 for ~99% — the burst is visible, just outranked. Reranking works ONLY
with information the finder's occupancy image does not carry (measured: an
image crop-verifier CNN gained nothing, cropver_v1); what does carry it is
the per-hit attention model watt — its scores separate signal from
background hits at AUC 0.94-0.99 on every preset (watt_transfer_check).

Per candidate window we extract FEAT_NAMES (watt-score counts, geometry of
confident hits, the finder's own logit/rank) and train a tiny MLP with
per-candidate BCE. Listwise training and deeper selectors were tried and do
NOT help (selector_iter.py) — the features and the amount of training data
are the binding constraints, hence the all-preset builder in script 31.
"""

import numpy as np
import torch

FEAT_NAMES = ["logit", "rank", "log_n_hits", "log_n_conf05", "log_n_conf02",
              "coherent02", "coherent05", "log_conf02_outer",
              "log_conf02_farfwd", "log_conf02_vertex", "log_sum_sig"]

FRONT, BACK = 50.0, 300.0  # candidate window extent around the bin start


def cand_features(logit, rank, Xw, sc, sys_of_hits):
    """11 features for one candidate window (same definition as script 23).

    Xw [n,4] hits (x, y, z, dt); sc watt sigmoid scores (LOW = signal-like);
    sys_of_hits raw system indices.
    """
    r = np.hypot(Xw[:, 0], Xw[:, 1])
    z = Xw[:, 2]
    c05, c02 = sc < 0.5, sc < 0.2
    return [
        float(logit), float(rank), np.log1p(len(sc)),
        np.log1p(c05.sum()), np.log1p(c02.sum()),
        float(len(np.unique(sys_of_hits[c02]))),
        float(len(np.unique(sys_of_hits[c05]))),
        np.log1p((c02 & (r > 550) & (np.abs(z) < 1600)).sum()),
        np.log1p((c02 & (z > 4000)).sum()),
        np.log1p((c02 & (r < 150)).sum()),
        np.log1p(float((1 - sc).sum())),
    ]


def within_frame_std(F, ok):
    """Append within-frame standardized copies of every feature.

    Selection is a comparison WITHIN one frame's k candidates; raw counts
    are not comparable across processes (windows differ 10x in hit count
    between DVMP and DIS-NC), standardized ones are.
    """
    m = ok[..., None].astype(np.float32)
    cnt = m.sum(1, keepdims=True).clip(1)
    mu = (F * m).sum(1, keepdims=True) / cnt
    sd = np.sqrt(((F - mu) ** 2 * m).sum(1, keepdims=True) / cnt) + 1e-6
    return np.concatenate([F, (F - mu) / sd], axis=2).astype(np.float32)


def train_selector(F, ok, label, epochs=300, hidden=16, seed=0):
    """Tiny BCE MLP over valid candidates. Returns (net, (mu, sd))."""
    torch.manual_seed(seed)
    nf = F.shape[-1]
    m = ok.reshape(-1)
    Xf = F.reshape(-1, nf)[m]
    yf = label.reshape(-1)[m].astype(np.float32)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(nf, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    Xt = torch.from_numpy((Xf - mu) / sd)
    yt = torch.from_numpy(yf)
    w = torch.where(yt == 1, (len(yf) - yt.sum()) / yt.sum(), 1.0)
    for _ in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            net(Xt).squeeze(1), yt, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    return net, (mu, sd)


def apply_selector(net, norm, F, ok, cand_t0, margin=0.0):
    """Refined t0 per frame: candidate with the highest selector score.

    margin > 0 gates the override: the winner must beat the finder's own
    argmax candidate (rank 0) by that much in selector logit, else rank 0
    is kept. Measured: the selector cannot beat argmax on DVMP even with
    preset-specific training (dim few-track signal -> uninformative hit
    features), so "defer when unsure" is the correct global policy.
    """
    mu, sd = norm
    with torch.no_grad():
        p = net(torch.from_numpy(
            ((F.reshape(-1, F.shape[-1]) - mu) / sd).astype(np.float32))
        ).squeeze(1).reshape(F.shape[0], F.shape[1]).numpy()
    p[~ok] = -1e9
    sel = p.argmax(axis=1)
    if margin > 0:
        keep = p[np.arange(len(sel)), sel] - p[:, 0] <= margin
        sel[keep] = 0
    return cand_t0[np.arange(len(sel)), sel], sel

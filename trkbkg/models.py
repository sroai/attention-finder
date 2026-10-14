"""Model architectures for per-hit background classification.

Baseline: HitMLP — a small multilayer perceptron applied to each hit
independently. Why start here:
  * Fastest to train (~minutes for 50M hits on GPU), trivially ONNX-exportable,
    tiny at inference (<1 MB weights, a few kFLOPs per hit) — far below the
    C++/CPU/2GB deployment budget.
  * The data analysis showed strong per-hit separation already (detector
    system and r/z position), so a per-hit model is a sound baseline.
  * Its score is a clean reference point for any future event-context model
    (e.g. a point-cloud / transformer over all hits in an event).

Deployment convention: the network input is RAW detector values
(x,y,z [mm], t [ns], dense system index). Feature engineering (r = hypot(x,y))
and standardization live INSIDE forward() as constant buffers, so the C++
ONNX runtime needs no preprocessing code and cannot go out of sync with
training-time normalization.
"""

import torch
from torch import nn


class HitMLP(nn.Module):
    """Per-hit classifier: [x,y,z,t] + system embedding -> background logit.

    Parameters
    ----------
    norm_mean / norm_std : per-feature stats for (x, y, z, t, r), from
        the training fold (dataset meta.json).
    n_systems : number of distinct detector systems (embedding rows).
    emb_dim : system embedding width. The embedding lets the model learn a
        per-detector "prior" (some systems are 96% signal, others 0.1%) plus
        how to interpret position/time differently per detector.
    hidden : widths of the fully-connected trunk.
    """

    def __init__(self, norm_mean, norm_std, n_systems: int,
                 emb_dim: int = 8, hidden=(64, 64, 32)):
        super().__init__()
        self.register_buffer("norm_mean", torch.tensor(norm_mean, dtype=torch.float32))
        self.register_buffer("norm_std", torch.tensor(norm_std, dtype=torch.float32))
        self.sys_emb = nn.Embedding(n_systems, emb_dim)

        layers, prev = [], 5 + emb_dim  # 5 = x,y,z,t,r
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sys_idx: torch.Tensor) -> torch.Tensor:
        """x: [N,4] raw (x,y,z,t); sys_idx: [N] int64. Returns [N] logits.

        Higher logit = more background-like (y=1 is 'filter out').
        """
        # sqrt(x^2+y^2) instead of torch.hypot: hypot has no ONNX mapping
        r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2).unsqueeze(1)
        feats = torch.cat([x, r], dim=1)
        feats = (feats - self.norm_mean) / self.norm_std
        h = torch.cat([feats, self.sys_emb(sys_idx)], dim=1)
        return self.trunk(h).squeeze(1)


class EventContextNet(nn.Module):
    """DeepSets-style classifier: per-hit encoding + per-event context.

    Motivation: HitMLP judges every hit in isolation, but background is
    correlated within an event (beam-gas showers, synchrotron fans, and the
    overall occupancy varies 6.5k-11k hits/event). This model gives each hit
    a summary of its whole event:

        h_i = phi(features_i)                    per-hit encoding (small MLP)
        c   = mean_i h_i                         event context (order-free pool)
        y_i = head(features_i, h_i, c)           per-hit decision

    Mean-pooling makes the context permutation-invariant and works for any
    number of hits — and it is a single ReduceMean in ONNX, so CPU inference
    stays one cheap pass over the event.

    forward() treats its whole input as ONE event (the deployment case:
    score a frame per call). forward_batched() trains on many events at once
    using a per-hit segment id to pool within each event only.
    """

    def __init__(self, norm_mean, norm_std, n_systems: int,
                 emb_dim: int = 8, enc_dim: int = 64, hidden=(64, 64, 32)):
        super().__init__()
        self.register_buffer("norm_mean", torch.tensor(norm_mean, dtype=torch.float32))
        self.register_buffer("norm_std", torch.tensor(norm_std, dtype=torch.float32))
        self.sys_emb = nn.Embedding(n_systems, emb_dim)

        in_dim = 5 + emb_dim  # x,y,z,t,r + system embedding
        self.phi = nn.Sequential(
            nn.Linear(in_dim, enc_dim), nn.ReLU(),
            nn.Linear(enc_dim, enc_dim), nn.ReLU(),
        )
        layers, prev = [], in_dim + 2 * enc_dim  # feats + h_i + context
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def _features(self, x, sys_idx):
        r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2).unsqueeze(1)
        feats = (torch.cat([x, r], dim=1) - self.norm_mean) / self.norm_std
        return torch.cat([feats, self.sys_emb(sys_idx)], dim=1)

    def forward(self, x, sys_idx):
        """Single event: all N input hits share one context. Used for ONNX."""
        f = self._features(x, sys_idx)
        h = self.phi(f)
        # broadcast instead of .expand(h.shape[0], ...): a traced shape[0]
        # would be baked in as a constant and break dynamic N in ONNX
        c = torch.zeros_like(h) + h.mean(dim=0, keepdim=True)
        return self.head(torch.cat([f, h, c], dim=1)).squeeze(1)

    def forward_batched(self, x, sys_idx, seg_id, n_events):
        """Many events at once; seg_id[i] gives hit i's event index (0..n_events-1)."""
        f = self._features(x, sys_idx)
        h = self.phi(f)
        sums = torch.zeros(n_events, h.shape[1], device=h.device, dtype=h.dtype)
        sums.index_add_(0, seg_id, h)
        counts = torch.zeros(n_events, device=h.device, dtype=h.dtype)
        counts.index_add_(0, seg_id, torch.ones_like(seg_id, dtype=h.dtype))
        c = (sums / counts.clamp(min=1).unsqueeze(1))[seg_id]
        return self.head(torch.cat([f, h, c], dim=1)).squeeze(1)


def build_model(name: str, meta: dict, **kwargs) -> nn.Module:
    """Factory keyed by config name; meta is the dataset meta.json dict."""
    common = dict(norm_mean=meta["norm"]["mean"], norm_std=meta["norm"]["std"],
                  n_systems=len(meta["system_ids"]))
    if name == "hit_mlp":
        return HitMLP(**common, **kwargs)
    if name == "event_context":
        return EventContextNet(**common, **kwargs)
    raise ValueError(f"unknown model '{name}'")

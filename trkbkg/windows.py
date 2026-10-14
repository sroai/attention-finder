"""Window finder (phase 3): locate the physics interaction time in a frame.

Problem recap (reports/03): each 2000 ns frame holds ONE ep interaction at a
random t0. Aligned to t0 the signal is a tight burst, so knowing t0 is worth
more than any per-hit feature — but naive occupancy peaks fail (21% success)
because beam-gas produces its own bursts. What distinguishes the ep burst is
collective GEOMETRY: hits across the whole barrel radius, far-forward hits a
few 10s of ns later, all radiating from the origin.

Approach — "scan and classify time slices":
 1. Chop the frame into fixed 25 ns bins (88 bins covering -100..2100 ns).
 2. For each bin compute a handful of aggregate geometry features (counts in
    physics-motivated regions, r/z spread, occupancy contrast). This turns a
    variable-size hit cloud into a fixed [88, F] image of the frame.
 3. A small 1D CNN slides over the bin axis (so each bin's score sees its
    temporal neighbourhood — the burst is 2-8 bins wide including the
    far-forward tail) and outputs one logit per bin.
 4. Training: softmax + cross-entropy over the 88 bins against the bin that
    contains the true t0 — i.e. "which time slice holds the interaction?"
    as an 88-way classification. Inference: argmax bin -> t0_est.

Deployment: the CNN is a tiny fixed-shape ONNX graph ([1, F, 88] input). The
per-bin counting (step 2) is ~20 lines of C++ documented in deploy_info.

NOTE for phase 4: the finder outputs t0_est only. Which hits to keep around
t0_est (front margin / back extent) is a separate physics decision made from
the containment curves in reports/04 — nothing is filtered here.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

# ---- time grid ------------------------------------------------------------
BIN_NS = 25.0
T_MIN, T_MAX = -100.0, 2100.0
N_BINS = int((T_MAX - T_MIN) / BIN_NS)  # 88

FEATURE_NAMES = [
    "log1p_n_hits",        # overall activity in the slice
    "log1p_n_outer",       # r > 550 mm, |z| < 1600 (outer barrel + TOF): the
                           #   ep burst reaches here, synrad/beam-gas rarely do
    "log1p_n_farfwd",      # z > 4000 mm (B0/RomanPots/OffM): signal secondaries
    "log1p_n_backward",    # z < -1200 mm (backward endcap disks)
    "log1p_n_vertex",      # r < 150 mm (vertex barrel: burst core)
    "r_std",               # radial spread: burst spans many layers coherently
    "z_std",               # longitudinal spread
    "occupancy_contrast",  # bin count / frame-average bin count: burst-ness
                           #   independent of the frame's background level
    "n_distinct_systems",  # how many detector systems fire in this slice —
                           #   the ep burst lights up many at once, a beam-gas
                           #   shower only a few
    "n_barrel_layers",     # distinct systems among the barrel radial stack
                           #   (VertexBarrel + TrackerBarrels + BarrelTOF):
                           #   coherent radial traversal = tracks from origin
    "log1p_n_tof",         # BarrelTOF (92) hits: outermost barrel layer
]
N_FEATURES = len(FEATURE_NAMES)

# dense indices (position in dataset meta system_ids list) are not available
# here, so coherence features use RAW system ids
BARREL_STACK = (31, 59, 60, 61, 64, 92)


def bin_index(t: np.ndarray) -> np.ndarray:
    return np.clip(((t - T_MIN) / BIN_NS).astype(np.int64), 0, N_BINS - 1)


def frame_features(X: np.ndarray, evt_index: np.ndarray, n_events: int,
                   system_id: np.ndarray = None):
    """Fixed-size [n_events, N_FEATURES, N_BINS] image from per-hit arrays.

    X: [N,4] raw (x, y, z, t); evt_index: [N] 0..n_events-1 (hits of one
    event contiguous or not — bincount doesn't care); system_id: [N] RAW
    detector system ids (for the coherence features).
    All features are computed with bincount over a flat (event, bin) key, so
    the whole fold is featurized in one pass without Python loops.
    """
    x, y, z, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    r = np.hypot(x, y)
    flat = evt_index * N_BINS + bin_index(t)
    size = n_events * N_BINS

    def count(mask=None):
        w = None if mask is None else mask.astype(np.float64)
        return np.bincount(flat, weights=w, minlength=size).reshape(
            n_events, N_BINS)

    n = count()
    n_safe = np.maximum(n, 1.0)

    def spread(v):
        s1 = np.bincount(flat, weights=v, minlength=size).reshape(n_events, N_BINS)
        s2 = np.bincount(flat, weights=v * v, minlength=size).reshape(n_events, N_BINS)
        var = s2 / n_safe - (s1 / n_safe) ** 2
        return np.sqrt(np.maximum(var, 0.0))

    def distinct_systems(restrict=None):
        """Per (event, bin): number of DISTINCT systems with >=1 hit."""
        sel = np.ones(len(flat), bool) if restrict is None else \
            np.isin(system_id, restrict)
        pairs = np.unique(np.stack([flat[sel], system_id[sel]]), axis=1)
        return np.bincount(pairs[0], minlength=size).reshape(n_events, N_BINS)

    feats = np.stack([
        np.log1p(n),
        np.log1p(count((r > 550) & (np.abs(z) < 1600))),
        np.log1p(count(z > 4000)),
        np.log1p(count(z < -1200)),
        np.log1p(count(r < 150)),
        spread(r) / 100.0,     # ~unit scale (mm -> 100 mm units)
        spread(z) / 1000.0,    # mm -> m units
        n / np.maximum(n.sum(axis=1, keepdims=True) / N_BINS, 1.0),
        distinct_systems() / 5.0,
        distinct_systems(BARREL_STACK) / 3.0,
        np.log1p(count(system_id == 92)),
    ], axis=1)
    return feats.astype(np.float32)


def targets_from_t0(t0: np.ndarray) -> np.ndarray:
    """Target bin per frame (int64); t0 must be finite."""
    return bin_index(t0)


class WindowFinderCNN(nn.Module):
    """Per-bin logits from the [F, N_BINS] frame image.

    Three dilated 1D convolutions give each bin a ~±8-bin (±200 ns) receptive
    field — enough to see the whole burst shape including the delayed
    far-forward tail — followed by a 1x1 conv producing one logit per bin.
    ~17k parameters; inference is microseconds on CPU.
    """

    def __init__(self, n_features: int = N_FEATURES, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, width, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(width, 1, kernel_size=1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, F, N_BINS] -> logits [B, N_BINS]."""
        return self.net(frames).squeeze(1)


# ---- containment analysis (for phase-4 window choice) ----------------------
def containment_scan(t, is_signal, evt_index, t0_est, fronts, backs):
    """For each (front, back) pair: fraction of signal hits inside
    [t0_est - front, t0_est + back] and fraction of background hits inside.

    This is THE table phase 4 uses to pick the cut: the physics requirement
    is that front/back are generous enough to keep essentially all signal.
    """
    est_per_hit = t0_est[evt_index]
    dt = t - est_per_hit
    out = {}
    for f in fronts:
        for b in backs:
            inside = (dt >= -f) & (dt <= b)
            out[f"front{int(f)}_back{int(b)}"] = {
                "signal_containment": float(inside[is_signal].mean()),
                "background_inside": float(inside[~is_signal].mean()),
            }
    return out

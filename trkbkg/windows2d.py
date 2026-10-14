"""Phase 6: 2D-image window finder.

The phase-3 finder (windows.py) compresses each 25 ns slice into 11
hand-picked numbers, so it cannot see SHAPE: a track ladder climbing the
barrel layers, a shower's wedge, the far-forward streak arriving a couple of
slices late. Here each frame becomes a small multi-channel IMAGE and a 2D
CNN learns those shapes directly:

    channel 0: (z, t) hit-count map    64 z-bins x 88 time-bins, log1p
    channel 1: (r, t) hit-count map    64 r-bins x 88 time-bins, log1p

Hits beyond the axis ranges clip into the edge rows, which turns the far
detectors (RomanPots at z=32 m) into a dedicated "far-forward" top row.
Future detector information — e.g. calorimeter energy maps — is simply more
channels; nothing else changes.

Model: VGG-style 3x3 conv blocks (the pattern, not the 138M-param network)
that pool ONLY along the spatial axis, so all 88 time columns survive to a
per-column logit head. Training remains "which 25 ns slice starts the
interaction" — softmax + cross-entropy over 88, identical protocol and
downstream (candidates + verifier) to phase 3.

Data leverage: because t0 is uniform by construction, a CYCLIC ROLL of the
image along time (label shifted with it) is a physically exact augmentation
— it multiplies the effective number of training frames ~90-fold and is the
antidote to the finder's data hunger.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .windows import BIN_NS, N_BINS, T_MIN, bin_index  # same time grid

N_SPACE = 64
Z_RANGE = (-2500.0, 8000.0)   # clip: backward disks .. far-forward edge row
R_RANGE = (0.0, 800.0)        # clip: beamline .. outer barrel/TOF edge row
N_CHANNELS = 2


def frame_images(X: np.ndarray, evt_index: np.ndarray, n_events: int):
    """[n_events, 2, N_SPACE, N_BINS] float32 images from per-hit arrays."""
    t_bin = bin_index(X[:, 3])
    z_bin = np.clip(((X[:, 2] - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0])
                     * N_SPACE).astype(np.int64), 0, N_SPACE - 1)
    r = np.hypot(X[:, 0], X[:, 1])
    r_bin = np.clip(((r - R_RANGE[0]) / (R_RANGE[1] - R_RANGE[0])
                     * N_SPACE).astype(np.int64), 0, N_SPACE - 1)

    size = n_events * N_SPACE * N_BINS
    img = np.zeros((2, size), np.float32)
    for ch, sbin in ((0, z_bin), (1, r_bin)):
        flat = (evt_index * N_SPACE + sbin) * N_BINS + t_bin
        img[ch] = np.bincount(flat, minlength=size).astype(np.float32)
    img = np.log1p(img).reshape(2, n_events, N_SPACE, N_BINS)
    return np.ascontiguousarray(img.transpose(1, 0, 2, 3))


# ---- phase 10: calorimeter channels ---------------------------------------
# Digitized calorimeter rec hits as two more channels, same time grid. The
# calorimeters live at larger radius and reach the ZDC at z = 36 m, so they
# get their own spatial ranges; the ZDC clips into the edge row exactly like
# the RomanPots do in the tracker channels. Energy-weighted (a burst deposits
# GeV where albedo deposits MeV — hit counting would hide that, see
# cal_hit_bkg_plan.md §6): each cell contributes its energy in 10 MeV units,
# log1p-compressed like the count channels.
CAL_Z_RANGE = (-4500.0, 8000.0)   # backward Hcal .. B0/ZDC edge row
CAL_R_RANGE = (0.0, 2800.0)       # beamline .. outer HcalBarrel
CAL_ENERGY_UNIT = 0.01            # [GeV] -> 10 MeV units before log1p


def frame_images_cal(X: np.ndarray, evt_index: np.ndarray, n_events: int):
    """[n_events, 2, N_SPACE, N_BINS] energy maps from per-hit cal arrays.

    X columns: x, y, z, time, energy (the 132 builder's CAL_COLS order).
    """
    t_bin = bin_index(X[:, 3])
    z_bin = np.clip(((X[:, 2] - CAL_Z_RANGE[0]) / (CAL_Z_RANGE[1] - CAL_Z_RANGE[0])
                     * N_SPACE).astype(np.int64), 0, N_SPACE - 1)
    r = np.hypot(X[:, 0], X[:, 1])
    r_bin = np.clip(((r - CAL_R_RANGE[0]) / (CAL_R_RANGE[1] - CAL_R_RANGE[0])
                     * N_SPACE).astype(np.int64), 0, N_SPACE - 1)
    weights = X[:, 4].astype(np.float64) / CAL_ENERGY_UNIT

    size = n_events * N_SPACE * N_BINS
    img = np.zeros((2, size), np.float32)
    for ch, sbin in ((0, z_bin), (1, r_bin)):
        flat = (evt_index * N_SPACE + sbin) * N_BINS + t_bin
        img[ch] = np.bincount(flat, weights=weights, minlength=size
                              ).astype(np.float32)
    img = np.log1p(img).reshape(2, n_events, N_SPACE, N_BINS)
    return np.ascontiguousarray(img.transpose(1, 0, 2, 3))


class WindowFinder2D(nn.Module):
    """VGG-style blocks, y-only pooling, per-time-column logits.

    Tunables (phase-6 CNN sweep):
      width     channels of the first block (doubles once after block 1)
      n_blocks  conv blocks; each halves the spatial axis (64 -> 64/2^n)
      batchnorm BatchNorm2d after every conv (faster convergence, mild reg.)
    """

    def __init__(self, n_channels: int = N_CHANNELS, width: int = 32,
                 n_blocks: int = 3, batchnorm: bool = False):
        super().__init__()
        w = width

        def block(cin, cout):
            layers = []
            for c_in, c_out in ((cin, cout), (cout, cout)):
                layers.append(nn.Conv2d(c_in, c_out, 3, padding=1))
                if batchnorm:
                    layers.append(nn.BatchNorm2d(c_out))
                layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d((2, 1)))  # halve space, keep time
            return layers

        widths = [n_channels] + [w] + [2 * w] * (n_blocks - 1)
        blocks = []
        for i in range(n_blocks):
            blocks += block(widths[i], widths[i + 1])
        self.backbone = nn.Sequential(*blocks)
        wout = widths[-1]
        self.head = nn.Sequential(      # 1D over time after y-collapse
            nn.Conv1d(wout, wout, 5, padding=2), nn.ReLU(),
            nn.Conv1d(wout, wout, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(wout, 1, 1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img [B, C, N_SPACE, N_BINS] -> logits [B, N_BINS]."""
        h = self.backbone(img)          # [B, 2w, 8, 88]
        h = h.mean(dim=2)               # collapse space -> [B, 2w, 88]
        return self.head(h).squeeze(1)


class WindowTypeNet(WindowFinder2D):
    """Phase 12: two heads on the finder backbone — WHEN and WHAT.

    The window head is inherited unchanged (88 per-column logits). The class
    head mean-pools the backbone's time-collapsed features over a band of
    columns around the window — the TRUE column during training (teacher
    forcing), the predicted one at inference — and maps them to class
    logits. Band pooling keeps classification focused on the signal region:
    the two classes come from different campaigns, so global pooling would
    invite the network to key on background differences instead of the event.

    The band is asymmetric like the verifier crops: the event's hits trail
    t0 (burst extends late, not early). Gathering is cyclic to match the
    cyclic roll augmentation.
    """

    def __init__(self, n_channels: int = N_CHANNELS, width: int = 32,
                 n_blocks: int = 3, batchnorm: bool = False,
                 n_classes: int = 2, band_before: int = 1, band_after: int = 6):
        super().__init__(n_channels, width, n_blocks, batchnorm)
        self.band_before, self.band_after = band_before, band_after
        wout = width if n_blocks == 1 else 2 * width
        self.cls_head = nn.Sequential(nn.Linear(wout, 64), nn.ReLU(),
                                      nn.Linear(64, n_classes))

    def forward(self, img: torch.Tensor, col: torch.Tensor | None = None):
        """img [B, C, N_SPACE, N_BINS] -> (window logits [B, N_BINS],
        class logits [B, n_classes]). `col` [B] picks the pooling band
        (training: true column); None uses the window head's argmax."""
        h = self.backbone(img).mean(dim=2)      # [B, 2w, 88]
        logits_t = self.head(h).squeeze(1)      # [B, 88]
        if col is None:
            col = logits_t.argmax(1)
        offsets = torch.arange(-self.band_before, self.band_after + 1,
                               device=img.device)
        cols = (col[:, None] + offsets[None, :]) % N_BINS
        band = torch.gather(h, 2, cols[:, None, :].expand(
            h.shape[0], h.shape[1], cols.shape[1]))
        return logits_t, self.cls_head(band.mean(dim=2))


CROP_BEFORE = 4    # columns kept in front of a candidate bin
CROP_AFTER = 12    # columns kept behind (burst extends late, not early)
CROP_W = CROP_BEFORE + CROP_AFTER


def candidate_crops(img: np.ndarray, cand_bins: np.ndarray) -> np.ndarray:
    """[n, k, C, N_SPACE, CROP_W] image crops around candidate time bins.

    img [n, C, N_SPACE, N_BINS]; cand_bins [n, k]. Frames are zero-padded at
    the borders (physical: nothing arrives before/after the frame).
    """
    n, C, H, _ = img.shape
    k = cand_bins.shape[1]
    padded = np.pad(img, ((0, 0), (0, 0), (0, 0), (CROP_BEFORE, CROP_AFTER)))
    out = np.empty((n, k, C, H, CROP_W), np.float32)
    for ci in range(k):
        starts = cand_bins[:, ci]          # + CROP_BEFORE - CROP_BEFORE
        for i in range(n):
            s = starts[i]
            out[i, ci] = padded[i, :, :, s:s + CROP_W]
    return out


class CandidateVerifier2D(nn.Module):
    """Second look at each candidate window: a small CNN on the image CROP
    around the candidate bin + the finder's own scalars (logit, rank,
    within-frame softmax prob), one score per candidate.

    Why this beats the 11-feature logistic selector: it sees SHAPE (the
    vertical comb, the radial ladder) instead of hand-picked counts, and it
    trains on every preset's frames (the hit-based selector was limited to
    18x275 where per-hit folds exist).
    """

    N_SCALARS = 3  # finder logit, candidate rank, within-frame softmax prob

    def __init__(self, n_channels: int = N_CHANNELS, width: int = 32):
        super().__init__()
        w = width

        def block(cin, cout, pool):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(pool)]

        self.backbone = nn.Sequential(          # 64 x 16
            *block(n_channels, w, (2, 2)),      # 32 x 8
            *block(w, 2 * w, (2, 2)),           # 16 x 4
            *block(2 * w, 2 * w, (2, 2)),       # 8 x 2
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * w + self.N_SCALARS, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, crop: torch.Tensor, scalars: torch.Tensor):
        """crop [B, C, N_SPACE, CROP_W], scalars [B, 3] -> logits [B]."""
        h = self.backbone(crop).flatten(1)
        return self.head(torch.cat([h, scalars], dim=1)).squeeze(1)


def cyclic_roll(img: torch.Tensor, target: torch.Tensor, rng: torch.Generator):
    """Random cyclic time-shift (exact augmentation: t0 is uniform)."""
    shift = int(torch.randint(0, N_BINS, (1,), generator=rng).item())
    return torch.roll(img, shifts=shift, dims=-1), (target + shift) % N_BINS

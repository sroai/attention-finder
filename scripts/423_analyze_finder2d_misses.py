#!/usr/bin/env python
"""Phase 6: anatomy of the 2D finder's MISSED frames (and optional TTA).

Where script 27 says HOW OFTEN the finder is right, this script says WHY it
is wrong, so the next fix is chosen from evidence (the phase-5 lesson):

  * rank of the true bin among the sorted logits of missed frames — if the
    burst is "visible but outranked" a verifier/ensemble helps; if the rank
    is deep, the input representation is blind and needs new channels;
  * occupancy trap — does the wrong argmax sit on a busier time column than
    the truth (background shower-burst outshining the ep burst)?
  * signal brightness — recall vs how much the true column pokes above the
    frame's typical column occupancy (dim DVMP few-track signal vs bright
    DIS); computed from the image itself, no parquet join needed;
  * a gallery of the worst misses with true/predicted columns marked.

--tta N additionally evaluates TEST-TIME AUGMENTATION: average the logits
over N evenly spaced cyclic time-shifts (each rolled back). The shift is
exact physics (t0 uniform), so TTA is a free ensemble — if it helps here we
bake the trick into the eval/export path.

Usage (from the repo root):
    python scripts/423_analyze_finder2d_misses.py \
        finder2d_v1 [--tta 8] [--cpu]

Outputs: plots/phase6/<run>_miss_rank.png, <run>_miss_anatomy.png,
<run>_miss_gallery.png, reports/phase6_<run>_misses.json.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.windows import BIN_NS, N_BINS, T_MIN
from trkbkg.windows2d import WindowFinder2D

DATA = (CFG.dataset_dir / "images2d_v1")
RUNS = CFG.runs_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase6"
PLOTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--tta", type=int, default=0,
                    help="also evaluate logits averaged over N cyclic shifts")
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

run_dir = CFG.runs_local_dir / args.run_name
if not run_dir.exists():
    run_dir = RUNS / args.run_name
ckpt = torch.load(run_dir / "best.pt", weights_only=False)
dev = torch.device("cpu" if args.cpu else
                   ("cuda" if torch.cuda.is_available() else "cpu"))
cfg = ckpt["config"]
model = WindowFinder2D(width=cfg["width"], n_blocks=cfg.get("blocks", 3),
                       batchnorm=cfg.get("batchnorm", False)).to(dev).eval()
model.load_state_dict(ckpt["model_state"])

meta = json.loads((DATA / "meta.json").read_text())
te = np.load(DATA / "test.npz")
X, target, t0 = torch.from_numpy(te["img"]), te["target"], te["t0"]


def infer(shifts):
    """Mean logits over cyclic time-shifts (rolled back). shifts=[0] = plain."""
    out = np.zeros((len(X), N_BINS), np.float32)
    with torch.no_grad():
        for s in shifts:
            for i in range(0, len(X), 512):
                xb = torch.roll(X[i:i + 512], shifts=s, dims=-1).to(dev)
                out[i:i + 512] += np.roll(model(xb).cpu().numpy(), -s, axis=1)
    return out / len(shifts)


def recall50(logits):
    err = logits.argmax(1) * BIN_NS + T_MIN + BIN_NS / 2 - t0
    return np.abs(err) <= 50


logits = infer([0])
pred = logits.argmax(1)
hit = recall50(logits)
miss = ~hit

# rank of the true bin (±1 tolerance, as in script 27) among sorted logits
order = np.argsort(-logits, axis=1)
rank = np.argmax(np.abs(order - target[:, None]) <= 1, axis=1)

# per-column occupancy proxy from the image itself: sum over space+channels
# of log1p counts. brightness = true-column occupancy minus frame median.
col_occ = te["img"].sum(axis=(1, 2))                      # [frames, N_BINS]
brightness = col_occ[np.arange(len(X)), target] - np.median(col_occ, axis=1)
occ_true = col_occ[np.arange(len(X)), target]
occ_pred = col_occ[np.arange(len(X)), pred]

stats = {"run": args.run_name, "test_frames": int(len(X)),
         "recall_within_50ns": float(hit.mean()),
         "missed_frames": int(miss.sum()),
         "miss_true_rank": {
             "median": float(np.median(rank[miss])),
             "top3": float((rank[miss] < 3).mean()),
             "top5": float((rank[miss] < 5).mean()),
             "top10": float((rank[miss] < 10).mean()),
             "deep(>=20)": float((rank[miss] >= 20).mean())},
         # occupancy trap: fraction of misses where the chosen wrong column
         # is busier than the true one
         "miss_pred_busier_than_true": float(
             (occ_pred[miss] > occ_true[miss]).mean()),
         "per_preset": {}}
for pi, name in enumerate(meta["presets"]):
    m = te["preset"] == pi
    if m.any():
        stats["per_preset"][name] = {
            "recall": float(hit[m].mean()),
            "median_brightness": float(np.median(brightness[m])),
            "miss_median_brightness": float(np.median(brightness[m & miss]))
            if (m & miss).any() else None}

if args.tta:
    shifts = np.linspace(0, N_BINS, args.tta, endpoint=False).astype(int)
    hit_tta = recall50(infer(list(shifts)))
    stats["tta"] = {"n_shifts": args.tta,
                    "recall_within_50ns": float(hit_tta.mean()),
                    "gain": float(hit_tta.mean() - hit.mean())}

out_json = Path(__file__).resolve().parents[1] / "reports" / \
    f"phase6_{args.run_name}_misses.json"
out_json.write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))

# ---- plot 1: true-bin rank among misses ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(rank[miss], bins=np.arange(0, 45) - 0.5, color="#2c5aa0")
axes[0].set_xlabel("rank of true bin (±1) in sorted logits")
axes[0].set_ylabel("missed frames")
axes[0].set_title(f"misses: {miss.sum()} of {len(X)} "
                  f"(recall {hit.mean():.1%})")
axes[0].grid(alpha=0.3)
cum = np.cumsum(np.bincount(rank[miss], minlength=N_BINS)) / miss.sum()
axes[1].plot(np.arange(1, N_BINS + 1), cum, color="#1a7f37")
axes[1].set_xlim(1, 20); axes[1].set_ylim(0, 1.02)
axes[1].set_xlabel("k"); axes[1].set_ylabel("misses with true bin in top-k")
axes[1].set_title("verifier ceiling on missed frames")
axes[1].grid(alpha=0.3)
fig.suptitle(f"{args.run_name}: where does the truth rank when argmax misses?")
fig.tight_layout()
fig.savefig(PLOTS / f"{args.run_name}_miss_rank.png", dpi=130)

# ---- plot 2: brightness + occupancy anatomy --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
q = np.quantile(brightness, np.linspace(0, 1, 11))
centers, recalls = [], []
for lo, hi in zip(q[:-1], q[1:]):
    m = (brightness >= lo) & (brightness < hi)
    if m.sum() > 20:
        centers.append((lo + hi) / 2); recalls.append(hit[m].mean())
axes[0].plot(centers, recalls, "o-", color="#2c5aa0")
axes[0].set_xlabel("signal brightness (true-column occupancy - frame median)")
axes[0].set_ylabel("recall within 50 ns")
axes[0].set_title("dim signal is the hard case")
axes[0].grid(alpha=0.3)
axes[1].scatter(occ_true[miss], occ_pred[miss], s=4, alpha=0.4,
                color="#d62728")
lim = max(occ_true[miss].max(), occ_pred[miss].max()) * 1.05
axes[1].plot([0, lim], [0, lim], "k--", lw=0.8)
axes[1].set_xlabel("true-column occupancy")
axes[1].set_ylabel("chosen-column occupancy")
axes[1].set_title("misses above the diagonal = occupancy trap")
axes[1].grid(alpha=0.3)
fig.suptitle(f"{args.run_name}: miss anatomy")
fig.tight_layout()
fig.savefig(PLOTS / f"{args.run_name}_miss_anatomy.png", dpi=130)

# ---- plot 3: gallery of the worst misses (deepest true rank) ---------------
worst = np.where(miss)[0][np.argsort(-rank[miss])][:12]
fig, axes = plt.subplots(3, 4, figsize=(16, 8))
for ax, fi in zip(axes.ravel(), worst):
    ax.imshow(te["img"][fi, 0], aspect="auto", origin="lower",
              cmap="Greys", interpolation="nearest")
    ax.axvline(target[fi], color="#1a7f37", lw=1.2, label="true")
    ax.axvline(pred[fi], color="#d62728", lw=1.2, ls="--", label="pred")
    ax.set_title(f"preset {te['preset'][fi]}  rank {rank[fi]}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].legend(fontsize=7, loc="upper right")
fig.suptitle(f"{args.run_name}: worst misses — (z,t) channel, "
             "green=true / red=argmax")
fig.tight_layout()
fig.savefig(PLOTS / f"{args.run_name}_miss_gallery.png", dpi=130)
print(f"wrote 3 plots to {PLOTS}")

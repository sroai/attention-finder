#!/usr/bin/env python
"""STEP 422 — phase 12: evaluate the two-head window + event-type network.

Runs the deployment path on the test fold (class predicted at the PREDICTED
window column) and reports: window recall@25/50/100 ns, class accuracy,
ROC-AUC, accuracy per sample slice (k-Λ, DIS minQ2 1/10/100), and DIS
accuracy vs truth Q² — the curve that says whether the classifier works
across the kinematic range or only where DIS looks least k-Λ-like.

Usage:
    python scripts/422_evaluate_event_type.py p12_cal \
        [--dataset images2d_p12] [--channels 0]

Outputs: <runs>/<run>/metrics.json, plots/phase12/<run>_{acc_vs_q2,scores}.png
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
from trkbkg.windows import BIN_NS, T_MIN
from trkbkg.windows2d import WindowTypeNet

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--dataset", default="images2d_p12")
parser.add_argument("--channels", type=int, default=0)
args = parser.parse_args()

DATA = CFG.dataset_dir / args.dataset
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase12"
PLOTS.mkdir(parents=True, exist_ok=True)

run_dir = CFG.runs_dir / args.run_name
ckpt = torch.load(run_dir / "best.pt", weights_only=False, map_location="cpu")
cfg = ckpt["config"]
n_ch = args.channels or cfg.get("channels", 4)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WindowTypeNet(n_channels=n_ch, width=cfg["width"],
                      n_blocks=cfg.get("blocks", 3),
                      batchnorm=cfg.get("batchnorm", False),
                      band_before=cfg.get("band_before", 1),
                      band_after=cfg.get("band_after", 6)).to(dev).eval()
model.load_state_dict(ckpt["model_state"])

te = np.load(DATA / "test.npz")
meta = json.loads((DATA / "meta.json").read_text())
X = torch.from_numpy(te["img"][:, :n_ch])
cls = te["cls"].astype(np.int64)
q2 = te["q2"]
sample_id = te["sample_id"]

preds_t, scores_c = [], []
with torch.no_grad():
    for i in range(0, len(X), 1024):
        lt, lc = model(X[i:i + 1024].to(dev))
        preds_t.append(lt.argmax(1).cpu().numpy())
        scores_c.append(torch.softmax(lc, 1)[:, 1].cpu().numpy())
pred_t = np.concatenate(preds_t)
score = np.concatenate(scores_c)            # P(DIS)
pred_c = (score > 0.5).astype(np.int64)
err = pred_t * BIN_NS + T_MIN + BIN_NS / 2 - te["t0"]


def roc_auc(y, s):
    """Rank-based AUC (Mann-Whitney); no sklearn dependency."""
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


metrics = {
    "run": args.run_name, "channels": n_ch, "frames": len(cls),
    "window": {f"within_{w}ns": float((np.abs(err) <= w).mean())
               for w in (25, 50, 100)},
    "cls_accuracy": float((pred_c == cls).mean()),
    "cls_auc": roc_auc(cls, score),
    "acc_klambda": float((pred_c == cls)[cls == 0].mean()),
    "acc_dis": float((pred_c == cls)[cls == 1].mean()),
    "per_sample": {},
}
for si, s in enumerate(meta["samples"]):
    m = sample_id == si
    metrics["per_sample"][s["name"]] = {
        "frames": int(m.sum()), "accuracy": float((pred_c == cls)[m].mean())}

# DIS accuracy vs truth Q² (log bins over the covered range)
def wilson(k, n, z=1.0):
    """Wilson score interval — honest near accuracy 1 with small n."""
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


dis = (cls == 1) & np.isfinite(q2) & (q2 > 0)
edges = np.geomspace(1, 1000, 13)
centers, accs, counts, err_lo, err_hi = [], [], [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = dis & (q2 >= lo) & (q2 < hi)
    if m.sum() >= 20:
        k, n_bin = int((pred_c == cls)[m].sum()), int(m.sum())
        w_lo, w_hi = wilson(k, n_bin)
        centers.append(float(np.sqrt(lo * hi)))
        accs.append(k / n_bin)
        counts.append(n_bin)
        err_lo.append(w_lo)
        err_hi.append(w_hi)
metrics["acc_vs_q2"] = {"q2_center": centers, "accuracy": accs,
                        "frames": counts, "wilson68_lo": err_lo,
                        "wilson68_hi": err_hi}

# per-frame record for miss autopsies (score = P(DIS))
np.savez(run_dir / "eval_frames.npz", gevt=te["gevt"], cls=cls, q2=q2,
         x=te["x"], sample_id=sample_id, score=score, pred_cls=pred_c,
         window_err_ns=err)

(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

fig, ax = plt.subplots(figsize=(7, 4.4))
yerr = np.maximum(0, [np.array(accs) - np.array(err_lo),
                      np.array(err_hi) - np.array(accs)])
ax.errorbar(centers, accs, yerr=yerr, fmt="o-", color="#2c5aa0",
            capsize=3, lw=1.3)
for cx, a, n_bin in zip(centers, accs, counts):
    ax.annotate(f"{n_bin}", (cx, a), xytext=(0, -14),
                textcoords="offset points", ha="center",
                fontsize=7, color="#888888")
ax.set_xscale("log")
ax.set_xlabel("truth Q² [GeV²]")
ax.set_ylabel("DIS classification accuracy")
ax.set_title(f"{args.run_name}: DIS accuracy vs Q² "
             f"(overall {metrics['cls_accuracy']:.4f}, AUC {metrics['cls_auc']:.4f})")
ax.grid(alpha=0.3); ax.set_ylim(min(0.9, min(accs) - 0.02) if accs else 0.5, 1.005)
fig.tight_layout(); fig.savefig(PLOTS / f"{args.run_name}_acc_vs_q2.png", dpi=120)

fig, ax = plt.subplots(figsize=(7, 4.4))
ax.hist(score[cls == 0], bins=60, histtype="step", lw=1.6, label="k-Λ")
ax.hist(score[cls == 1], bins=60, histtype="step", lw=1.6, label="DIS/NC")
ax.set_yscale("log"); ax.set_xlabel("P(DIS)"); ax.set_ylabel("frames")
ax.set_title(f"{args.run_name}: class score distributions")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(PLOTS / f"{args.run_name}_scores.png", dpi=120)
print("wrote", PLOTS)

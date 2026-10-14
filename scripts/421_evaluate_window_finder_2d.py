#!/usr/bin/env python
"""Phase 6: evaluate the 2D finder — per-process, vs the phase-3 finder.

Usage (from the repo root):
    python scripts/421_evaluate_window_finder_2d.py <run>

Reports test recall within 25/50/100 ns overall and PER PRESET, and — on
the frozen 18x275 holdout — the direct comparison against wf_v2_r3 (78.5%)
and the phase-5 verifier (88.1%). Also top-k coverage (the ceiling a
verifier stage on top of this finder would inherit).

Outputs: plots/phase6/finder2d_comparison.png, runs/<run>/metrics.json,
reports/phase6 numbers merged into reports/phase5_stats.json sibling file
reports/phase6_stats.json.
"""

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
from trkbkg.windows2d import WindowFinder2D

RUNS = CFG.runs_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase6"
PLOTS.mkdir(parents=True, exist_ok=True)

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("run_name")
_ap.add_argument("--dataset", default="images2d_v1",
                 help="dataset dir under <dataset_dir> (phase 10: images2d_p10)")
_ap.add_argument("--channels", type=int, default=0,
                 help="evaluate on the first N image channels (0 = all); must "
                      "match the training slice")
_args = _ap.parse_args()
DATA = CFG.dataset_dir / _args.dataset
if _args.dataset != "images2d_v1":   # phase-10 plots stay out of phase6/
    PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase10"
    PLOTS.mkdir(parents=True, exist_ok=True)
run_name = _args.run_name
run_dir = CFG.runs_local_dir / run_name
if not run_dir.exists():
    run_dir = RUNS / run_name
ckpt = torch.load(run_dir / "best.pt", weights_only=False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = ckpt["config"]
model = WindowFinder2D(n_channels=cfg.get("channels", 2), width=cfg["width"],
                       n_blocks=cfg.get("blocks", 3),
                       batchnorm=cfg.get("batchnorm", False)).to(dev).eval()
model.load_state_dict(ckpt["model_state"])

meta = json.loads((DATA / "meta.json").read_text())
te = np.load(DATA / "test.npz")
logits = []
with torch.no_grad():
    X = torch.from_numpy(te["img"])
    if _args.channels:
        X = X[:, :_args.channels].contiguous()
    for i in range(0, len(X), 1024):
        logits.append(model(X[i:i + 1024].to(dev)).cpu().numpy())
logits = np.concatenate(logits)
pred = logits.argmax(1)
err = pred * BIN_NS + T_MIN + BIN_NS / 2 - te["t0"]

order = np.argsort(-logits, axis=1)
rank_hit = np.argmax(np.abs(order - te["target"][:, None]) <= 1, axis=1)


def block(mask):
    return {
        "frames": int(mask.sum()),
        "within_25ns": float((np.abs(err[mask]) <= 25).mean()),
        "within_50ns": float((np.abs(err[mask]) <= 50).mean()),
        "within_100ns": float((np.abs(err[mask]) <= 100).mean()),
        "top5_coverage": float((rank_hit[mask] < 5).mean()),
        "top10_coverage": float((rank_hit[mask] < 10).mean()),
    }


metrics = {"run": run_name, "epoch": ckpt["epoch"],
           "overall": block(np.ones(len(err), bool))}
# phase-10 single-sample datasets carry no preset list; "overall" is the story
presets = meta.get("presets", [])
for pi, name in enumerate(presets):
    m = te["preset"] == pi
    if m.any():
        metrics[name] = block(m)
(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

# ---- comparison plot on the frozen 18x275 holdout (multi-preset runs only;
# phase-10 datasets stop at metrics.json — the A/B report plots its own) ----
if "18x275_v3" not in metrics:
    print(json.dumps(metrics["overall"], indent=2))
    print("single-sample dataset: skipped the phase-6 comparison plot")
    raise SystemExit(0)
m18 = te["preset"] == 0
baselines = {"occupancy peak\n(phase 2)": 0.16,
             "wf_v2_r3 argmax\n(phase 3)": 0.785,
             "+ verifier k=10\n(phase 5)": 0.881,
             f"{run_name} argmax\n(phase 6)": metrics["18x275_v3"]["within_50ns"],
             f"{run_name} top-10 ceiling": metrics["18x275_v3"]["top10_coverage"]}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
names = list(baselines)
colors = ["#c9c9c9", "#999999", "#2c5aa0", "#1a7f37", "#7fc97f"]
axes[0].bar(range(len(names)), [baselines[n] for n in names], color=colors)
axes[0].set_xticks(range(len(names)), names, fontsize=7)
axes[0].set_ylabel("18x275 test frames within 50 ns")
axes[0].set_ylim(0, 1.02); axes[0].grid(alpha=0.3, axis="y")
for i, n in enumerate(names):
    axes[0].text(i, baselines[n] + 0.01, f"{baselines[n]:.1%}",
                 ha="center", fontsize=8)

pn = [n for n in presets if n in metrics]
axes[1].bar(range(len(pn)), [metrics[n]["within_50ns"] for n in pn],
            color="#1a7f37")
axes[1].set_xticks(range(len(pn)), [n.replace("background", "bg")
                                    for n in pn], fontsize=7, rotation=15)
axes[1].set_ylabel("test frames within 50 ns")
axes[1].set_ylim(0, 1.02); axes[1].grid(alpha=0.3, axis="y")
axes[1].set_title("per-process generalization")
fig.suptitle(f"phase 6: 2D-image finder ({run_name})")
fig.tight_layout()
fig.savefig(PLOTS / f"{run_name}_comparison.png", dpi=130)

stats_f = Path(__file__).resolve().parents[1] / "reports" / f"phase6_{run_name}.json"
stats_f.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
print(f"wrote {PLOTS / 'finder2d_comparison.png'}")

#!/usr/bin/env python
"""Phase 8: per-preset in-window AUC audit of watt models.

Replaces the lost scratchpad watt_transfer_check.py with a repo script.
Scores TRUE windows (val/test of window_hits_all_v1) with each given watt
run and reports the in-window signal/background AUC per preset — the
number that decides whether retraining watt on all presets (watt_all_v1)
actually closed the DVMP/DIS gap (watt_r2 baseline: 0.99/0.99/0.94/0.95).

Usage (from the repo root):
    python scripts/426_watt_preset_audit.py \
        watt_r2 watt_all_v1 [--fold test] [--cpu]

Outputs: reports/phase8_watt_audit.json, plots/phase8/watt_preset_auc.png.
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
from trkbkg.attention import WindowNet
from trkbkg.dataset import load_meta
from trkbkg.evaluate import roc_auc

DATA = (CFG.dataset_dir / "window_hits_all_v1")
HIT_META = (CFG.dataset_dir / "18x275_v3")
RUNS = CFG.runs_dir
RUNS_LOCAL = CFG.runs_local_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase8"
PLOTS.mkdir(parents=True, exist_ok=True)
REPORTS = Path(__file__).resolve().parents[1] / "reports"

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+")
parser.add_argument("--fold", default="test")
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
dev = torch.device("cpu" if args.cpu else
                   ("cuda" if torch.cuda.is_available() else "cpu"))

meta = json.loads((DATA / "meta.json").read_text())
PRESETS = meta["presets"]
SHORT = {"18x275_v3": "18x275", "10x130_v3": "10x130",
         "background_dvmp-10x275-25-10_v3": "DVMP",
         "background-dis-nc-10x275_v3": "DIS-NC"}
n_systems = len(load_meta(HIT_META)["system_ids"])

d = np.load(DATA / f"{args.fold}.npz")
X, sysi, y, evt, wpre = (d["X"], d["sys"].astype(np.int64),
                         d["y"], d["evt"], d["wpreset"])
change = np.flatnonzero(np.diff(evt.astype(np.int64))) + 1
starts = np.concatenate([[0], change])
ends = np.concatenate([change, [len(evt)]])

results = {}
for run in args.runs:
    run_dir = RUNS_LOCAL / run
    if not (run_dir / "best.pt").exists():
        run_dir = RUNS / run
    ck = torch.load(run_dir / "best.pt", weights_only=False)
    net = WindowNet([0] * 5, [1] * 5, n_systems, d=ck["config"]["dim"],
                    n_blocks=ck["config"]["blocks"]).to(dev).eval()
    net.load_state_dict(ck["model_state"])   # norm buffers come from ckpt

    scores = np.empty(len(y), np.float32)
    with torch.no_grad():
        for s, e in zip(starts, ends):
            scores[s:e] = torch.sigmoid(net(
                torch.from_numpy(X[s:e]).to(dev),
                torch.from_numpy(sysi[s:e]).to(dev))).cpu().numpy()

    per = {}
    for pi, pname in enumerate(PRESETS):
        m = wpre == pi
        per[SHORT[pname]] = roc_auc(y[m], scores[m])
    results[run] = per
    print(run, {k: round(v, 4) for k, v in per.items()}, flush=True)

out = {"fold": args.fold, "auc_per_preset": results}
REPORTS.joinpath("phase8_watt_audit.json").write_text(
    json.dumps(out, indent=2))

labels = [SHORT[p] for p in PRESETS]
x = np.arange(len(labels))
w = 0.8 / len(args.runs)
fig, ax = plt.subplots(figsize=(7, 4))
for j, run in enumerate(args.runs):
    vals = [results[run][l] for l in labels]
    ax.bar(x + j * w - 0.4 + w / 2, vals, w, label=run)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.3f}", (xi + j * w - 0.4 + w / 2, v), size=7,
                    xytext=(0, 2), textcoords="offset points", ha="center")
ax.set_xticks(x, labels)
ax.set_ylim(0.9, 1.005)
ax.set_ylabel(f"in-window signal/background AUC ({args.fold})")
ax.set_title("Hit-score quality per preset: the verifier's 6th feature")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS / "watt_preset_auc.png", dpi=150)
print(f"wrote {PLOTS / 'watt_preset_auc.png'}")

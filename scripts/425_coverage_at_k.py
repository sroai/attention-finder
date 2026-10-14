#!/usr/bin/env python
"""Phase 8: candidate-quality diagnostic — coverage@k for the single finder
vs the 4-variant logit ensemble.

The whole reranking pipeline can never beat its candidate list: t0_final_v1
reached 91.9% test recall against a 99.0% top-10 ceiling. Two free knobs can
raise that ceiling BEFORE any retraining:
  * ENSEMBLE the four sweep checkpoints (prob-mean was +0.8 argmax in the
    phase-6 scratchpad check) — better logits should also rank the true bin
    higher in missed frames;
  * larger k — top-20 instead of top-10.
This script measures both on val+test so the rebuild decision (which finder
source, which k) is made from evidence, not hope.

Coverage@k = fraction of frames where at least one of the top-k bins has
|bin_center - t0| <= 50 ns — i.e. the ceiling any selector can reach.

Usage (from the repo root):
    python scripts/425_coverage_at_k.py [--cpu]

Outputs: plots/phase8/coverage_at_k.png, reports/phase8_coverage.json.
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
RUNS_LOCAL = CFG.runs_local_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase8"
PLOTS.mkdir(parents=True, exist_ok=True)
REPORTS = Path(__file__).resolve().parents[1] / "reports"

VARIANTS = ["finder2d_v1", "finder2d_v2_deep", "finder2d_v3_wide",
            "finder2d_v4_bn"]
KS = list(range(1, 21))

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
dev = torch.device("cpu" if args.cpu else
                   ("cuda" if torch.cuda.is_available() else "cpu"))

meta = json.loads((DATA / "meta.json").read_text())
PRESETS = meta["presets"]


def load_model(name):
    ck = torch.load(RUNS_LOCAL / name / "best.pt", weights_only=False)
    cfg = ck["config"]
    m = WindowFinder2D(width=cfg["width"], n_blocks=cfg.get("blocks", 3),
                       batchnorm=cfg.get("batchnorm", False)).to(dev).eval()
    m.load_state_dict(ck["model_state"])
    return m


def logits_of(model, X):
    out = np.zeros((len(X), N_BINS), np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 512):
            out[i:i + 512] = model(X[i:i + 512].to(dev)).cpu().numpy()
    return out


def coverage_curve(logits, t0):
    """coverage@k for every k in KS. Candidate time = bin center."""
    order = np.argsort(-logits, axis=1)                     # [frames, bins]
    cand_t = order * BIN_NS + T_MIN + BIN_NS / 2            # candidate centers
    ok = np.abs(cand_t - t0[:, None]) <= 50                 # [frames, bins]
    cum = np.cumsum(ok, axis=1) > 0                         # covered by top-k
    return {k: float(cum[:, k - 1].mean()) for k in KS}, cum


models = {v: load_model(v) for v in VARIANTS}
results = {}
for fold in ("val", "test"):
    d = np.load(DATA / f"{fold}.npz")
    X = torch.from_numpy(d["img"])
    t0, preset = d["t0"], d["preset"]

    per_model = {v: logits_of(m, X) for v, m in models.items()}
    # prob-mean ensemble: softmax each model over bins, then average
    probs = np.mean([np.exp(l - l.max(1, keepdims=True)) /
                     np.exp(l - l.max(1, keepdims=True)).sum(1, keepdims=True)
                     for l in per_model.values()], axis=0)

    fold_res = {}
    for tag, lg in [("finder2d_v1", per_model["finder2d_v1"]),
                    ("ensemble4", np.log(probs + 1e-12))]:
        curve, cum = coverage_curve(lg, t0)
        per_preset = {}
        for pi, pname in enumerate(PRESETS):
            m = preset == pi
            if m.sum() == 0:
                continue
            per_preset[pname] = {
                "argmax": float(cum[m, 0].mean()),
                "cov@10": float(cum[m, 9].mean()),
                "cov@15": float(cum[m, 14].mean()),
                "cov@20": float(cum[m, 19].mean()),
            }
        fold_res[tag] = {"curve": curve, "per_preset": per_preset,
                         "frames": int(len(t0))}
        print(f"{fold:5s} {tag:12s} argmax {curve[1]:.4f}  cov@10 "
              f"{curve[10]:.4f}  cov@15 {curve[15]:.4f}  cov@20 "
              f"{curve[20]:.4f}", flush=True)
    results[fold] = fold_res

REPORTS.joinpath("phase8_coverage.json").write_text(
    json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, fold in zip(axes, ("val", "test")):
    for tag, color in [("finder2d_v1", "#2c5aa0"), ("ensemble4", "#c0392b")]:
        curve = results[fold][tag]["curve"]
        ax.plot(KS, [curve[k] for k in KS], "o-", ms=3, color=color,
                label=tag)
    ax.axhline(results[fold]["finder2d_v1"]["curve"][10], color="#2c5aa0",
               ls=":", lw=0.8)
    ax.set_title(f"{fold} ({results[fold]['finder2d_v1']['frames']} frames)")
    ax.set_xlabel("k (candidates kept)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
axes[0].set_ylabel("coverage@k (selector ceiling)")
fig.suptitle("Window-candidate coverage: single finder vs 4-model ensemble")
fig.tight_layout()
fig.savefig(PLOTS / "coverage_at_k.png", dpi=150)
print(f"wrote {PLOTS / 'coverage_at_k.png'}")

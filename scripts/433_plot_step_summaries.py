#!/usr/bin/env python
"""Backfill summary plots for every phase-3/4 step that lacked one.

Usage (from the repo root):
    python scripts/433_plot_step_summaries.py

Produces (plots/window/):
    wf_training_comparison.png    wf_v1 vs wf_v2 learning curves (feature value)
    watt_training_comparison.png  watt_v1 vs wmlp_v1 (attention value)
    watt_v1_e2e_budget.png        background/signal budget bars, real vs
                                  perfect finder (from script 18's numbers)
All read data already on disk (history.jsonl / metrics_end_to_end.json) —
safe to re-run anytime.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

RUNS = CFG.runs_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "window"


def history(run):
    return [json.loads(l) for l in
            (RUNS / run / "history.jsonl").read_text().splitlines()]


# --- 1. window finder: wf_v1 vs wf_v2 --------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for run, color, lab in [("wf_v1", "#999999", "wf_v1 (8 geometry features, v1 data)"),
                        ("wf_v2_r2", "#2c5aa0", "wf_v2_r2 (+coherence, v2 albedo-aware data)")]:
    h = history(run)
    ax.plot([r["epoch"] for r in h], [r["val_within_50ns"] for r in h],
            color=color, label=lab)
ax.axhline(0.16, color="#c0392b", ls="--", lw=1,
           label="phase-2 occupancy-peak baseline")
ax.set_xlabel("epoch"); ax.set_ylabel("val frames with |t0 err| <= 50 ns")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title("window finder: what the burst-coherence features bought")
fig.tight_layout(); fig.savefig(PLOTS / "wf_training_comparison.png", dpi=130)
plt.close(fig)

# --- 2. in-window model: attention vs no-attention -------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
for run, color, lab in [("wmlp_r2", "#999999", "wmlp_r2 (0 blocks: dt only)"),
                        ("watt_r2", "#1a7f37", "watt_r2 (2 attention blocks)")]:
    h = history(run)
    rej_key = next(k for k in h[0] if k.startswith("val_rej"))
    axes[0].plot([r["epoch"] for r in h], [r["val_auc"] for r in h],
                 color=color, label=lab)
    axes[1].plot([r["epoch"] for r in h], [r[rej_key] for r in h],
                 color=color, label=lab)
axes[0].axhline(0.865, color="#c0392b", ls="--", lw=1,
                label="phase-1 ceiling (no dt)")
axes[0].set_ylabel("val AUC (in-window)")
axes[1].set_ylabel("val in-window rejection @ 99.9% in-window eff")
for ax in axes:
    ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("in-window classifier: value of attention at equal everything else")
fig.tight_layout(); fig.savefig(PLOTS / "watt_training_comparison.png", dpi=130)
plt.close(fig)

# --- 3. end-to-end budget: real finder vs perfect finder -------------------
dec = json.loads((RUNS / "watt_r2" / "metrics_end_to_end.json").read_text())[
    "decomposition"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, key, name in [(axes[0], "finder_est", "REAL finder"),
                      (axes[1], "finder_true", "PERFECT finder (true t0)")]:
    title = (f"{name}: "
             f"{100*dec[key]['background_rejection']:.1f}% rejection @ "
             f"{100*dec[key]['signal_efficiency']:.1f}% eff")
    d = dec[key]["background_budget"]
    shares = [d["inside_share_of_bkg"], d["outside_share_of_bkg"]]
    rejected = [shares[0] * d["inside_rejected"], shares[1] * d["outside_rejected"]]
    kept = [s - r for s, r in zip(shares, rejected)]
    x = ["inside window", "outside window"]
    ax.bar(x, rejected, color="#1a7f37", label="rejected")
    ax.bar(x, kept, bottom=rejected, color="#c0392b", label="kept (survives)")
    for i, s in enumerate(shares):
        ax.text(i, s + 0.02, f"{100*s:.0f}% of bkg\n"
                f"({100*rejected[i]/s:.0f}% rejected)", ha="center", fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("fraction of ALL background")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
fig.suptitle("where background survives: the loose outside threshold is the "
             "price of signal stranded by finder misses")
fig.tight_layout(); fig.savefig(PLOTS / "watt_r2_e2e_budget.png", dpi=130)
plt.close(fig)

print("wrote wf_training_comparison, watt_training_comparison, "
      "watt_v1_e2e_budget to", PLOTS)

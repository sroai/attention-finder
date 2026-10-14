#!/usr/bin/env python
"""Phase 6 CNN tuning summary: all finder2d variants side by side.

Usage (from the repo root):
    python scripts/422_finder2d_tuning_report.py

Aggregates every <runs_local_dir>/finder2d_* run: learning curves (val recall@50ns
vs epoch), final test recall per variant with parameter counts, and the best
variant's per-process bars + error histogram.
Outputs: plots/phase6/tuning_curves.png, tuning_summary.png; prints a table.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

LOCAL = CFG.runs_local_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase6"
COLORS = ["#999999", "#2c5aa0", "#1a7f37", "#c0392b", "#8e5aa0", "#e6a03c"]

runs = sorted(d for d in LOCAL.glob("finder2d_*") if (d / "metrics.json").exists())
if not runs:
    sys.exit("no evaluated finder2d runs found")

fig, ax = plt.subplots(figsize=(8.5, 5))
rows = []
for d, color in zip(runs, COLORS):
    cfg = json.loads((d / "config.json").read_text())
    hist = [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines()]
    met = json.loads((d / "metrics.json").read_text())
    label = (f"{d.name}: w{cfg['width']} b{cfg.get('blocks', 3)}"
             f"{' bn' if cfg.get('batchnorm') else ''}"
             f" ({cfg.get('parameters', 0)/1e3:.0f}k par)")
    ax.plot([h["epoch"] for h in hist], [h["val_within_50ns"] for h in hist],
            color=color, label=label)
    rows.append({"run": d.name, "width": cfg["width"],
                 "blocks": cfg.get("blocks", 3),
                 "bn": cfg.get("batchnorm", False),
                 "params_k": round(cfg.get("parameters", 0) / 1e3),
                 "test_overall_50ns": met["overall"]["within_50ns"],
                 "test_18x275_50ns": met["18x275_v3"]["within_50ns"],
                 "top10_18x275": met["18x275_v3"]["top10_coverage"]})
ax.axhline(0.785, color="k", ls=":", lw=1, label="wf_v2_r3 argmax (phase 3)")
ax.axhline(0.881, color="k", ls="--", lw=1, label="phase-5 verifier")
ax.set_xlabel("epoch"); ax.set_ylabel("val recall within 50 ns")
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
ax.set_title("phase 6: window-finder CNN tuning")
fig.tight_layout(); fig.savefig(PLOTS / "tuning_curves.png", dpi=130)
plt.close(fig)

rows.sort(key=lambda r: -r["test_18x275_50ns"])
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(rows))
ax.bar(x - 0.2, [r["test_18x275_50ns"] for r in rows], 0.4,
       color="#1a7f37", label="18x275 test recall@50ns")
ax.bar(x + 0.2, [r["top10_18x275"] for r in rows], 0.4,
       color="#7fc97f", label="top-10 ceiling")
ax.set_xticks(x, [f"{r['run']}\nw{r['width']} b{r['blocks']}"
                  f"{' bn' if r['bn'] else ''}\n{r['params_k']}k" for r in rows],
              fontsize=7)
ax.axhline(0.785, color="k", ls=":", lw=1)
ax.axhline(0.881, color="k", ls="--", lw=1)
ax.set_ylim(0.7, 1.01); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
ax.set_title("finder2d variants (frozen 18x275 holdout)")
fig.tight_layout(); fig.savefig(PLOTS / "tuning_summary.png", dpi=130)

print(f"{'run':>16} {'w':>3} {'b':>2} {'bn':>3} {'par(k)':>7} "
      f"{'test@50':>8} {'18x275@50':>10} {'top10':>6}")
for r in rows:
    print(f"{r['run']:>16} {r['width']:>3} {r['blocks']:>2} "
          f"{str(r['bn']):>3} {r['params_k']:>7} "
          f"{r['test_overall_50ns']:>8.4f} {r['test_18x275_50ns']:>10.4f} "
          f"{r['top10_18x275']:>6.3f}")
print("wrote tuning_curves.png, tuning_summary.png")

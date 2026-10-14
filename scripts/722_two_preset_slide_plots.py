#!/usr/bin/env python
"""Phase 8 (slides): energy-preset-only (18x275 + 10x130) plot variants.

The audience-facing deck quotes the two ENERGY presets only for now — the
process presets (DVMP, DIS-NC) have preliminary statistics (682 / 184 test
frames) and are kept to the internal reports until they firm up. All
numbers are read from the stored report JSONs, nothing is retyped.

Outputs (plots/phase8/):
  finder_2presets.png   — 1D finder vs 2D finder argmax vs top-20 ceiling
  final_2presets.png    — finder argmax vs combined selector vs ceiling
  negatives_2presets.png— crop-verifier honest negative, energy presets only

Usage (from the repo root):
    python scripts/722_two_preset_slide_plots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path(__file__).resolve().parents[1] / "reports"
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase8"
PLOTS.mkdir(parents=True, exist_ok=True)

P18, P10 = "18x275_v3", "10x130_v3"
WF_V2_18X275 = 0.785          # phase-3 1D finder, 18x275 test (report 04)

final = json.loads((REPORTS / "phase6_t0_final_v2.json").read_text())
rerank = json.loads((REPORTS / "phase6_t0_rerank_all_v2.json").read_text())
cropver = json.loads((REPORTS / "phase6_cropver_v1.json").read_text())

argmax = {p: final["test"]["per_preset"][p]["argmax"] for p in (P18, P10)}
combined = {p: final["test"]["per_preset"][p]["combined"] for p in (P18, P10)}
ceiling = {p: rerank["test"]["per_preset"][p]["ceiling"] for p in (P18, P10)}

BLUE, RED, GREY, GREEN = "#2c5aa0", "#c0392b", "#9aa5b1", "#27ae60"
LABELS = {P18: "18x275", P10: "10x130"}


def bar_panel(ax, series, colors, presets=(P18, P10)):
    """series: list of (name, {preset: value})."""
    x = np.arange(len(presets))
    w = 0.8 / len(series)
    for j, ((name, vals), c) in enumerate(zip(series, colors)):
        xs = x + j * w - 0.4 + w / 2
        ys = [vals.get(p, np.nan) for p in presets]
        ax.bar(xs, ys, w, label=name, color=c)
        for xi, v in zip(xs, ys):
            if np.isfinite(v):
                ax.annotate(f"{100 * v:.1f}", (xi, v), size=9, xytext=(0, 2),
                            textcoords="offset points", ha="center")
    ax.set_xticks(x, [LABELS[p] for p in presets])
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("test frames with |t0 err| ≤ 50 ns")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)


# 1 — the finder itself (slide: Seeing shape)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
bar_panel(ax, [
    ("1D feature finder", {P18: WF_V2_18X275}),
    ("2D image finder (argmax)", argmax),
    ("top-20 candidate ceiling", ceiling),
], [GREY, BLUE, GREEN])
ax.set_title("2D images beat hand-picked features — and the candidate\n"
             "list almost always contains the right window", size=10)
fig.tight_layout()
fig.savefig(PLOTS / "finder_2presets.png", dpi=150)

# 2 — the full selection pipeline (slide: Current pipeline)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
bar_panel(ax, [
    ("finder argmax", argmax),
    ("combined selector (prelim.)", combined),
    ("top-20 candidate ceiling", ceiling),
], [GREY, RED, GREEN])
ax.set_title("Candidates + retrained judges: the selector closes most\n"
             "of the gap to the ceiling", size=10)
fig.tight_layout()
fig.savefig(PLOTS / "final_2presets.png", dpi=150)

# 3 — honest negative: crop-verifier (slide: What didn't work)
fig, ax = plt.subplots(figsize=(6.0, 4.0))
cv_arg = {p: cropver["test"]["per_preset"][p]["argmax"] for p in (P18, P10)}
cv_ver = {p: cropver["test"]["per_preset"][p]["verifier"] for p in (P18, P10)}
bar_panel(ax, [
    ("finder argmax", cv_arg),
    ("+ crop verifier", cv_ver),
], [GREY, BLUE])
ax.set_title("Crop-verifier CNN: re-reading the same pixels adds nothing",
             size=10)
fig.tight_layout()
fig.savefig(PLOTS / "negatives_2presets.png", dpi=150)

for f in ("finder_2presets", "final_2presets", "negatives_2presets"):
    print(f"wrote {PLOTS / f}.png")

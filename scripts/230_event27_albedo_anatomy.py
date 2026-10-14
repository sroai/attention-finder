#!/usr/bin/env python
"""Anatomy plot of test frame gevt=27 — the "late central hits" mystery.

Usage (from the repo root):
    python scripts/230_event27_albedo_anatomy.py

Findings documented in reports/06_event27_anatomy.md: the two central hits
at ~1650 ns (780 ns after the interaction) are neutron-albedo hits from the
K+ hadronic stop behind BarrelTOF, attributed to the generator K+ because
the intermediate slow neutrons were below the MCParticle persistence cut.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

OUT = Path(__file__).resolve().parents[1] / "plots" / "window" / "event27_anatomy.png"

df = (pl.scan_parquet(str(CFG.parquet_dir / "18x275_v3/0022.trk_hits.parquet"))
      .filter(pl.col("evt") == 27).collect()
      .with_columns(r=(pl.col("trk_hit_pos_x")**2 + pl.col("trk_hit_pos_y")**2).sqrt()))

t = df["trk_hit_time"].to_numpy()
r = df["r"].to_numpy()
z = df["trk_hit_pos_z"].to_numpy()
org = df["prt_origin"].to_numpy()
pdg = df["prt_pdg"].to_numpy()

VTX_T = 878.3   # interaction time from mc_particles
K_END = (898.0, 1106.0, 884.0)  # K+ endpoint r, z, ~time

groups = [
    ("background", (org >= 3), "#cccccc", 3, 1),
    ("e' 12.8 GeV (pdg 11, prompt)", (org == 1) & (pdg == 11), "#2c5aa0", 26, 3),
    ("K+ 10.5 GeV prompt pass", (org == 1) & (pdg == 321) & (t < 900), "#1a7f37", 26, 3),
    ("K+ ALBEDO (origin 5, slow-n, +760 ns)", (org == 5), "#c0392b", 60, 5),
    ("Lambda-decay secondaries @ZDC/RP", (org == 2) & (np.abs(z) > 4000), "#7fc97f", 14, 2),
    ("other signal secondaries", (org == 2) & (np.abs(z) <= 4000), "#e6a03c", 20, 2),
]

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for ax, yv, ylab, ylim in [(axes[0], r, "r [mm]", (0, 1400)),
                           (axes[1], z, "z [mm]", (-2200, 2200))]:
    for lab, m, c, s, zo in groups:
        if m.any():
            ax.scatter(t[m], yv[m], s=s, c=c, zorder=zo, label=f"{lab} ({m.sum()})",
                       linewidths=0)
    ax.axvline(VTX_T, color="k", lw=1, ls="-")
    ax.set_ylabel(ylab)
    ax.set_ylim(*ylim)
axes[0].scatter([K_END[2]], [K_END[0]], marker="x", s=90, c="#c0392b", zorder=6,
                label="K+ endpoint (nuclear stop)")
axes[1].scatter([K_END[2]], [K_END[1]], marker="x", s=90, c="#c0392b", zorder=6)
axes[0].annotate("slow-neutron transport\n~604 mm in ~750 ns = 0.0027c (~3 keV)",
                 xy=(1633, 557), xytext=(1150, 1050), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="#c0392b"))
axes[0].legend(loc="upper left", fontsize=7, markerscale=1.3)
axes[1].set_xlabel("hit time [ns]")
axes[0].set_title("frame 0022/evt27 anatomy: e p -> e' K+ Lambda at t=878 ns; "
                  "K+ stops behind BarrelTOF; its albedo neutrons hit the "
                  "trackers 760 ns later (attributed to the K+)")
axes[1].set_title("note: Lambda(240 GeV) daughters reach z=33 m (off scale) at ~980 ns")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")

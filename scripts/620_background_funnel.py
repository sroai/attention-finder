#!/usr/bin/env python
"""Phase 7: funnel plot — where the 55k background particles per frame go.

Recomputes the particle -> sim hit -> raw -> rec -> seed -> track reduction
chain on one 26.07.0 DIS+background file (unfiltered reconstruction) and
draws it as a horizontal log-scale funnel with per-stage reduction factors.
Companion note: notes/stock-eicrecon-has-no-time-window.md — the big drop
happens in the ACTS seed finder (helix-pointing combinatorics), NOT in any
time cut (stock EICrecon has none) and NOT at the digitization threshold.

Output: plots/phase7_dis/background_funnel.png
"""

import json
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

IN = "/data/background-2026-07/pythia8NCDIS_10x275_minQ2=1000_beamEffects_xAngle=-0.025_hiDiv_1.0531.edm4hep.root"
OUT = str(CFG.results_root / "eicrecon-dis/unfilt_10531.root")
PLOT = str(CFG.plots_dir / "phase7_dis/background_funnel.png")
JSON = (CFG.reports_dir / "phase7_funnel.json")

# central-tracker sim collections and their digi thresholds [keV]
SIM = {"VertexBarrelHits": 0.54, "SiBarrelHits": 0.54, "TrackerEndcapHits": 0.54,
       "MPGDBarrelHits": 0.10, "OuterMPGDBarrelHits": 0.10,
       "BackwardMPGDEndcapHits": 0.10, "ForwardMPGDEndcapHits": 0.10}
REC = ["SiBarrelVertexRecHits", "SiBarrelTrackerRecHits", "SiEndcapTrackerRecHits",
       "MPGDBarrelRecHits", "OuterMPGDBarrelRecHits", "BackwardMPGDEndcapRecHits",
       "ForwardMPGDEndcapRecHits"]
# every tracker the AI filter sees (the filter's 12 collections): the
# 7 central ones above + TOF + the far-forward trackers
SIM_ALL = list(SIM) + ["TOFBarrelHits", "TOFEndcapHits", "B0TrackerHits",
                       "ForwardRomanPotHits", "ForwardOffMTrackerHits"]

TEXT = "#0b0b0b"


def main():
    t = uproot.open(IN)["events"]
    n_evt = t.num_entries
    tot_sim = tot_pass = tot_cells = 0
    pidx = [set() for _ in range(n_evt)]
    for c, thr in SIM.items():
        edep = t[f"{c}/{c}.eDep"].array()
        cid = t[f"{c}/{c}.cellID"].array()
        e = ak.flatten(edep).to_numpy() * 1e6  # keV
        tot_sim += len(e)
        tot_pass += (e >= thr).sum()
        for k in range(n_evt):
            ee = edep[k].to_numpy() * 1e6
            tot_cells += len(np.unique(cid[k].to_numpy()[ee >= thr]))
        pi = t[f"_{c}_particle/_{c}_particle.index"].array()
        for k in range(n_evt):
            pidx[k].update(pi[k].to_numpy().tolist())
    n_mc = float(ak.mean(ak.num(t["MCParticles/MCParticles.generatorStatus"].array())))
    # all tracker sim hits of the frame — what the AI filter is shown (12 colls)
    tot_sim_all = sum(len(ak.flatten(t[f"{c}/{c}.eDep"].array()))
                      for c in SIM_ALL)

    o = uproot.open(OUT)["events"]
    rec = seedable = 0
    for c in REC:
        x = ak.flatten(o[f"{c}/{c}.position.x"].array()).to_numpy()
        y = ak.flatten(o[f"{c}/{c}.position.y"].array()).to_numpy()
        z = ak.flatten(o[f"{c}/{c}.position.z"].array()).to_numpy()
        r = np.hypot(x, y)
        rec += len(x)
        seedable += ((r > 33) & (r < 440) & (z > -1500) & (z < 1700)).sum()
    n_seeds = float(ak.mean(ak.num(o["CentralTrackSeeds/CentralTrackSeeds.perigee.x"].array())))
    n_trk = float(ak.mean(ak.num(o["CentralCKFTracks/CentralCKFTracks.ndf"].array())))

    stages = [
        ("MC particles in frame", n_mc, ""),
        ("particles hitting central tracker", sum(len(s) for s in pidx) / n_evt,
         "93% never touch it"),
        ("all tracker sim hits (12 systems)", tot_sim_all / n_evt,
         "what the AI filter is shown"),
        ("central-tracker sim hits", tot_sim / n_evt, ""),
        ("above digi threshold", tot_pass / n_evt, "threshold cuts only 4%"),
        ("raw hits (same-cell merge, full 2000 ns)", tot_cells / n_evt, "no time cut"),
        ("rec hits", rec / n_evt, ""),
        ("in seeder acceptance (33<r<440 mm)", seedable / n_evt, ""),
        ("track seeds", n_seeds, "helix must point to beamline"),
        ("CKF tracks", n_trk, ""),
    ]

    # cache the stage counts: the plotting scripts (48) read this instead of
    # re-reading two ROOT files every time a figure is restyled
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(
        {"file": IN, "events": n_evt,
         "stages": [{"label": s[0], "per_event": float(s[1]), "note": s[2]}
                    for s in stages]}, indent=2))
    print(f"saved {JSON}")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ys = np.arange(len(stages))[::-1]
    vals = np.array([s[1] for s in stages])
    # single-hue sequential ramp (dataviz reference palette, blue)
    ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
            "#5598e7", "#3987e5", "#256abf", "#104281"]
    ax.barh(ys, vals, height=0.62, edgecolor="none",
            color=[ramp[i % len(ramp)] for i in range(len(stages))])
    ax.set_xscale("log")
    ax.set_xlim(1, 2e5)
    for (label, v, note), yy in zip(stages, ys):
        ax.text(v * 1.15, yy, f"{v:,.0f}".replace(",", " "),
                va="center", fontsize=10, color=TEXT, fontweight="bold")
        if note:
            ax.text(1.7e5, yy - 0.30, note, va="center", ha="right",
                    fontsize=8.5, color="#52514e", style="italic")
    # reduction factors between consecutive stages
    for i in range(1, len(stages)):
        f = stages[i - 1][1] / stages[i][1]
        if f >= 1.5:
            ax.text(1.15, (ys[i - 1] + ys[i]) / 2, f"÷{f:,.0f}",
                    va="center", fontsize=9, color="#e34948", fontweight="bold")
    ax.set_yticks(ys, [s[0] for s in stages], fontsize=10)
    ax.set_xlabel("per event (log scale)", color=TEXT)
    ax.grid(True, axis="x", color="#d9d8d4", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Where the background goes (stock EICrecon, NC DIS 10x275 "
                 "+ bkg, per frame)", fontsize=11, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"saved {PLOT}")
    for label, v, _ in stages:
        print(f"  {label:42s} {v:10.1f}")


if __name__ == "__main__":
    main()

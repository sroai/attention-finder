#!/usr/bin/env python
"""STEP 253 — the phase-9 calorimeter showcase events, four views each.

Collects the 0022 events used across plots/phase9_cal_plan/ (standard pair
6/27, the frames500 dozen, the csv/threshold events 8/11, the six p-gas
showcases) and redraws every one in the same folder with four suffixes:

  _bytype.png  truth deposits, tracker color convention (signal green,
               background split by source band, NO albedo relabel)
  _energy.png  truth deposits colored by contribution energy (log), two
               spectra: Greens = signal-related, Reds = background
  _thresh.png  deposits vs digitization: bright = the deposit's CELL
               survives (its cellID appears in the RecHits), washed-out =
               discarded — as 252
  _digi.png    the digitized event: RecHits colored by energy (log) plus
               reconstructed clusters as hollow circles, area ~ energy —
               as 250's --clusters overlay

Truth comes from the dd4hep sim file; RecHits and clusters from the
CAMPAIGN reco of the same file (event order is 1:1). The deposit views
show every Geant4 contribution at its own time; the digitized view shows
one hit per cell whose time is the digitizer's single timestamp — the gap
between _thresh and _digi is the answer to "why does the long surviving
tail vanish after digitization".

Usage (run from OUTSIDE /work — its .venv is broken and hijacks uv):
    cd /tmp && uv run --no-project --with uproot,awkward,numpy,matplotlib \
        python ./scripts/253_cal_collected_views.py
    # subset / other files: --events 6 27 --sim ... --reco ...
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.temporal import COLOR_STYLE, status_category  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "plots" / "phase9_cal_plan" / "collected"

SIM_DEFAULT = ("/data/msf-2026-06/dd4hep-background/18x275/"
               "k_lambda_18x275_5000evt_0022.edm4hep.root")
RECO_DEFAULT = ("/data/msf-2026-06/reco-background/"
                "k_lambda_18x275_5000evt_0022.edm4eic.root")
# union of the events the phase-9 folders show, in SIM numbering: frames500
# entries 0-11 = sim 0-11; pair 6/27; the six p-gas showcases; the
# bkg_0022_ev1400 re-reco STARTED at sim event 1400 (and wrote 1411 before
# 1410), so its "evt 8" = sim 1408 and "evt 11" = sim 1410.
EVENTS_DEFAULT = list(range(12)) + [27, 1408, 1410, 2667, 2777, 3948, 4059, 4454]
EXCLUDE = ("Lumi",)

parser = argparse.ArgumentParser()
parser.add_argument("--sim", default=SIM_DEFAULT)
parser.add_argument("--reco", default=RECO_DEFAULT)
parser.add_argument("--events", type=int, nargs="+", default=EVENTS_DEFAULT)
parser.add_argument("--all-time", action="store_true",
                    help="no 2100 ns axis limit; filenames get _all-time")
args = parser.parse_args()

OUT.mkdir(parents=True, exist_ok=True)
SUFFIX = "_all-time" if args.all_time else ""
sim = uproot.open(args.sim)["events"]
reco = uproot.open(args.reco)["events"]
tag = Path(args.sim).name.rsplit("_", 1)[-1].split(".")[0]   # "0022"

sim_keys = {k.split("/")[0] for k in sim.keys()}
CAL = sorted(c[: -len("Contributions")] for c in sim_keys
             if c.endswith("HitsContributions")
             and not any(x in c for x in EXCLUDE))
reco_keys = {k.split("/")[0] for k in reco.keys()}

# reco entries can be written out of order (multithreaded eicrecon runs were:
# the ev1400 re-reco wrote 1411 before 1410) — align by EventHeader number.
_reco_evnums = np.asarray(
    reco["EventHeader/EventHeader.eventNumber"].array()).ravel()
RECO_ENTRY = {int(n): i for i, n in enumerate(_reco_evnums)}


def arr(tree, branch, evt):
    return np.asarray(tree[branch].array(entry_start=evt, entry_stop=evt + 1)[0])


def classify(gs):
    if gs in (1, 2):
        return 1
    if gs >= 1000:
        return 3
    return 0


def particle_origins(gs, pb, pe, pi):
    """Memoized parent walk, same as scripts 250/252."""
    memo = np.full(len(gs), -1, dtype=np.int8)
    for i in range(len(gs)):
        if memo[i] >= 0:
            continue
        if gs[i] != 0:
            memo[i] = classify(gs[i])
            continue
        chain, j, res = [i], i, 0
        for _ in range(200):
            if pb[j] == pe[j]:
                break
            p = pi[pb[j]]
            if gs[p] != 0:
                b = classify(gs[p])
                res = 2 if b == 1 else 4 if b == 3 else 0
                break
            if memo[p] >= 0:
                res = memo[p]
                break
            chain.append(p)
            j = p
        memo[chain] = res
    return memo


def read_sim_event(evt):
    """Every calorimeter contribution: t/z/r/e, particle status + origin
    (albedo-relabeled), parent-cell cellID, survived-digitization flag."""
    gs = arr(sim, "MCParticles/MCParticles.generatorStatus", evt)
    origins = particle_origins(
        gs,
        arr(sim, "MCParticles/MCParticles.parents_begin", evt),
        arr(sim, "MCParticles/MCParticles.parents_end", evt),
        arr(sim, "_MCParticles_parents/_MCParticles_parents.index", evt))
    vtx_time = arr(sim, "MCParticles/MCParticles.time", evt)

    cols = {k: [] for k in ("t", "z", "r", "e", "status", "origin",
                            "vtx", "kept")}
    for name in CAL:
        c = name + "Contributions"
        ct = arr(sim, f"{c}/{c}.time", evt)
        if not len(ct):
            continue
        pidx = arr(sim, f"_{c}_particle/_{c}_particle.index", evt)
        counts = (arr(sim, f"{name}/{name}.contributions_end", evt)
                  - arr(sim, f"{name}/{name}.contributions_begin", evt))
        hx = np.repeat(arr(sim, f"{name}/{name}.position.x", evt), counts)
        hy = np.repeat(arr(sim, f"{name}/{name}.position.y", evt), counts)
        cell = np.repeat(arr(sim, f"{name}/{name}.cellID", evt), counts)
        rec_name = name[: -len("Hits")] + "RecHits"
        rec_cells = (arr(reco, f"{rec_name}/{rec_name}.cellID", RECO_ENTRY[evt])
                     if rec_name in reco_keys else np.array([], np.uint64))
        cols["t"].append(ct)
        cols["z"].append(np.repeat(arr(sim, f"{name}/{name}.position.z", evt),
                                   counts))
        cols["r"].append(np.hypot(hx, hy))
        cols["e"].append(arr(sim, f"{c}/{c}.energy", evt))
        cols["status"].append(gs[pidx])
        cols["origin"].append(origins[pidx])
        cols["vtx"].append(vtx_time[pidx])
        cols["kept"].append(np.isin(cell, rec_cells))
    d = {k: np.concatenate(v) for k, v in cols.items()}
    # NO albedo relabel here (user decision 2026-10-01): late signal deposits
    # keep their signal category, so _bytype shows signal vs background by
    # source only, and _thresh matches the original 252 convention.
    return d


def read_digi_event(evt):
    revt = RECO_ENTRY[evt]
    """RecHits (t/z/r/e) of the calorimeter collections + clusters."""
    t, z, r, e = [], [], [], []
    for name in CAL:
        c = name[: -len("Hits")] + "RecHits"
        if c not in reco_keys:
            continue
        ct = arr(reco, f"{c}/{c}.time", revt)
        if not len(ct):
            continue
        cx = arr(reco, f"{c}/{c}.position.x", revt)
        cy = arr(reco, f"{c}/{c}.position.y", revt)
        t.append(ct)
        z.append(arr(reco, f"{c}/{c}.position.z", revt))
        r.append(np.hypot(cx, cy))
        e.append(arr(reco, f"{c}/{c}.energy", revt))
    hits = {k: np.concatenate(v) for k, v in
            [("t", t), ("z", z), ("r", r), ("e", e)]}

    colls = sorted(c for c in reco_keys if c.endswith("Clusters")
                   and not any(x in c for x in
                               ("Truth", "SplitMerge", "Baseline", "Lumi")))
    ct, cz, cr, ce = [], [], [], []
    for c in colls:
        tt = arr(reco, f"{c}/{c}.time", revt)
        if not len(tt):
            continue
        xx = arr(reco, f"{c}/{c}.position.x", revt)
        yy = arr(reco, f"{c}/{c}.position.y", revt)
        ct.append(tt)
        cz.append(arr(reco, f"{c}/{c}.position.z", revt))
        cr.append(np.hypot(xx, yy))
        ce.append(arr(reco, f"{c}/{c}.energy", revt))
    clus = ({k: np.concatenate(v) for k, v in
             [("t", ct), ("z", cz), ("r", cr), ("e", ce)]} if ct else None)
    return hits, clus


def two_panel(d_z, d_r, d_t, draw, title, out_path, colorbars=(), deco=None):
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7), sharex=True)
    for ax, yv, ylab in [(axes[0], d_z, "z [mm]"), (axes[1], d_r, "r [mm]")]:
        draw(ax, yv)
        if deco is not None:
            deco(ax, ylab.startswith("z"))
        ax.set_ylabel(ylab)
        if not args.all_time:
            ax.set_xlim(-100, 2100)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="upper right", fontsize=7, markerscale=2)
    axes[1].set_xlabel("time [ns]")
    axes[0].set_title(title, fontsize=11)
    fig.tight_layout()
    for mappable, label in colorbars:
        fig.colorbar(mappable, ax=axes, pad=0.01, label=label)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path.name}")


for evt in args.events:
    d = read_sim_event(evt)
    hits, clus = read_digi_event(evt)
    base = OUT / f"{tag}_evt{evt:04d}"
    n_late = int((d["t"] > 2100).sum())
    beyond = (f", t_max {d['t'].max():,.0f} ns" if args.all_time else
              (f", {n_late} beyond 2100 ns" if n_late else ""))
    cats = np.array([status_category(st, o)
                     for st, o in zip(d["status"], d["origin"])])

    # event-start (solid) and last-prompt-signal (dashed) lines; the prompt
    # cut uses the tracker two-rule only for the SPAN, not for coloring
    sig0 = d["origin"] <= 2
    t0 = (d["t"][d["origin"] == 1].min() if (d["origin"] == 1).any() else
          (d["t"][sig0].min() if sig0.any() else np.nan))
    prompt = sig0 & ~((d["t"] - d["vtx"] > 100) | (d["t"] - t0 > 300))
    t1 = d["t"][prompt].max() if prompt.any() else np.nan
    # z axis: cut before the lumi region; far-forward (ZDC) only when hit
    z_fwd = max(d["z"].max(), hits["z"].max() if len(hits["z"]) else 0.0)
    zlim = (-4800.0, 8000.0 if z_fwd <= 8000 else z_fwd * 1.02)

    def decorate(ax, is_z):
        for x, ls in ((t0, "-"), (t1, "--")):
            if np.isfinite(x):
                ax.axvline(x, color="#1a7f37", ls=ls, lw=1.1, alpha=0.85)
        if is_z:
            ax.set_ylim(*zlim)

    # 1 — by type (tracker color convention)
    def draw_bytype(ax, yv):
        for cat, (c, szp, zo) in COLOR_STYLE.items():
            m = cats == cat
            if m.any():
                ax.scatter(d["t"][m], yv[m], s=szp, c=c, zorder=zo,
                           label=f"{cat} ({m.sum()})", linewidths=0)
    two_panel(d["z"], d["r"], d["t"], draw_bytype,
              f"{tag} event {evt} truth deposits by source: "
              f"{len(d['t'])} contributions, "
              f"signal span {t0:.0f}..{t1:.0f} ns{beyond}",
              base.with_name(base.name + f"_bytype{SUFFIX}.png"), deco=decorate)

    # 2 — by energy, two spectra (signal-related greens, background reds)
    sig_rel = d["origin"].astype(np.int64)
    sig_m = np.isin(d["origin"], (1, 2, 5))
    norm = LogNorm(vmin=max(d["e"].min(), 1e-7), vmax=d["e"].max())
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7), sharex=True)
    sc_s = sc_b = None
    for ax, yv, ylab in [(axes[0], d["z"], "z [mm]"), (axes[1], d["r"], "r [mm]")]:
        sc_b = ax.scatter(d["t"][~sig_m], yv[~sig_m], c=d["e"][~sig_m], s=3,
                          cmap="Reds", norm=norm, linewidths=0,
                          label=f"background ({int((~sig_m).sum())})")
        sc_s = ax.scatter(d["t"][sig_m], yv[sig_m], c=d["e"][sig_m], s=5,
                          cmap="Greens", norm=norm, linewidths=0, zorder=3,
                          label=f"signal-related ({int(sig_m.sum())})")
        decorate(ax, ylab.startswith("z"))
        ax.set_ylabel(ylab)
        if not args.all_time:
            ax.set_xlim(-100, 2100)
    axes[0].legend(loc="upper right", fontsize=7, markerscale=2)
    axes[1].set_xlabel("deposit time [ns]")
    axes[0].set_title(f"{tag} event {evt} truth deposits by energy: "
                      f"signal-related vs background spectra{beyond}",
                      fontsize=11)
    fig.tight_layout()
    fig.colorbar(sc_s, ax=axes, pad=0.01, label="signal-related energy [GeV]")
    fig.colorbar(sc_b, ax=axes, pad=0.01, label="background energy [GeV]")
    fig.savefig(base.with_name(base.name + f"_energy{SUFFIX}.png"), dpi=110)
    plt.close(fig)
    print(f"wrote {base.name}_energy{SUFFIX}.png")

    # 3 — threshold survival (252 convention)
    sig = d["origin"] <= 2
    frac_n = d["kept"].mean()
    frac_e = d["e"][d["kept"]].sum() / max(d["e"].sum(), 1e-12)
    groups = [
        (~sig & ~d["kept"], "#cfbcbc", 2, 1, "background, discarded"),
        (sig & ~d["kept"], "#b9cdb9", 2, 2, "signal, discarded"),
        (~sig & d["kept"], "#d62728", 6, 4, "background, survived"),
        (sig & d["kept"], "#1a9641", 6, 5, "signal, survived"),
    ]

    def draw_thresh(ax, yv):
        for m, c, szp, zo, lab in groups:
            if m.any():
                ax.scatter(d["t"][m], yv[m], s=szp, c=c, zorder=zo,
                           label=f"{lab} ({m.sum()})", linewidths=0)
    two_panel(d["z"], d["r"], d["t"], draw_thresh,
              f"{tag} event {evt} deposits vs digitization: "
              f"{100*frac_n:.0f}% of deposits / {100*frac_e:.0f}% of energy "
              f"in surviving cells{beyond}",
              base.with_name(base.name + f"_thresh{SUFFIX}.png"), deco=decorate)

    # 4 — the digitized event: RecHits + clusters as circles
    hnorm = LogNorm(vmin=max(hits["e"].min(), 1e-6), vmax=hits["e"].max())
    sc_h = {}

    def draw_digi(ax, yv_is_z):
        yv = hits["z"] if yv_is_z else hits["r"]
        sc_h["h"] = ax.scatter(hits["t"], yv, c=hits["e"], s=7,
                               cmap="viridis", norm=hnorm, linewidths=0,
                               label=f"rec hits ({len(yv)})")
        if clus is not None:
            cy = clus["z"] if yv_is_z else clus["r"]
            ax.scatter(clus["t"], cy, s=np.maximum(clus["e"] * 60, 25),
                       facecolors="none", edgecolors="#2c5aa0",
                       linewidths=1.4, zorder=8,
                       label=f"clusters ({len(cy)})")

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7), sharex=True)
    for ax, is_z, ylab in [(axes[0], True, "z [mm]"), (axes[1], False, "r [mm]")]:
        draw_digi(ax, is_z)
        decorate(ax, is_z)
        ax.set_ylabel(ylab)
        if not args.all_time:
            ax.set_xlim(-100, 2100)
    axes[0].legend(loc="upper right", fontsize=7, markerscale=1.5)
    axes[1].set_xlabel("rec hit time [ns]")
    nc = len(clus["t"]) if clus is not None else 0
    axes[0].set_title(f"{tag} event {evt} digitized: {len(hits['t'])} rec "
                      f"hits, {hits['e'].sum():.1f} GeV, {nc} clusters",
                      fontsize=11)
    fig.tight_layout()
    fig.colorbar(sc_h["h"], ax=axes, pad=0.01, label="rec hit energy [GeV]")
    fig.savefig(base.with_name(base.name + f"_digi{SUFFIX}.png"), dpi=110)
    plt.close(fig)
    print(f"wrote {base.name}_digi{SUFFIX}.png")

print(f"done -> {OUT}")

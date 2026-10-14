#!/usr/bin/env python
"""STEP 252 — truth contributions vs the digitization threshold.

Redraws the script-250 contribution displays from a SELF-CONTAINED reco file
(sim cal hits + contributions + RecHits in one file, as produced by the
extended-output eicrecon runs), two variants per event:

  <stem>_evtNNNN_egrad.png  — every contribution colored by its energy (log)
  <stem>_evtNNNN_thresh.png — colored by whether its CELL survived
      digitization (cellID present in the RecHits of the same frame):
      bright green = signal-related, survived; bright red = background,
      survived; washed-out gray-green / gray-red = discarded by threshold.

"Survived" is the cell-level truth: a tiny deposit in a cell that another
particle pushed over threshold counts as kept — that is what reaches
reconstruction.

Usage (run from OUTSIDE /work — its .venv is broken and hijacks uv):
    cd /tmp && uv run --no-project --with uproot,awkward,numpy,matplotlib \
        python ./scripts/252_cal_threshold_displays.py \
        --input <self-contained>.edm4eic.root --events 6 27
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import uproot

OUT = Path(__file__).resolve().parents[1] / "plots" / "phase9_cal_plan" / "threshold"

DEFAULT_INPUT = ("/data/msf-2026-06/rereco-selfcontained/"
                 "bkg_0022_first500.edm4eic.root")   # pass --input elsewhere

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=DEFAULT_INPUT)
parser.add_argument("--events", type=int, nargs="+", default=[6, 27])
parser.add_argument("--all-time", action="store_true",
                    help="no 2100 ns axis limit — show the full time range "
                         "so the tails' ends are visible; filenames get an "
                         "_all-time suffix")
args = parser.parse_args()

SUFFIX = "_all-time" if args.all_time else ""
tree = uproot.open(args.input)["events"]
tag = Path(args.input).name.split(".")[0]
OUT.mkdir(parents=True, exist_ok=True)

all_keys = {k.split("/")[0] for k in tree.keys()}
CAL = sorted(c[: -len("Contributions")] for c in all_keys
             if c.endswith("HitsContributions"))


def arr(branch, evt):
    return np.asarray(tree[branch].array(entry_start=evt, entry_stop=evt + 1)[0])


def classify(gs):
    if gs in (1, 2):
        return 1
    if gs >= 1000:
        return 3
    return 0


def particle_origins(gs, pb, pe, pi):
    """Same memoized parent walk as script 250."""
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


def read_event(evt):
    gs = arr("MCParticles/MCParticles.generatorStatus", evt)
    origins = particle_origins(
        gs,
        arr("MCParticles/MCParticles.parents_begin", evt),
        arr("MCParticles/MCParticles.parents_end", evt),
        arr("_MCParticles_parents/_MCParticles_parents.index", evt))

    t, z, r, e, sig, kept = [], [], [], [], [], []
    for name in CAL:
        c = name + "Contributions"
        ct = arr(f"{c}/{c}.time", evt)
        if not len(ct):
            continue
        pidx = arr(f"_{c}_particle/_{c}_particle.index", evt)
        counts = (arr(f"{name}/{name}.contributions_end", evt)
                  - arr(f"{name}/{name}.contributions_begin", evt))
        hx = np.repeat(arr(f"{name}/{name}.position.x", evt), counts)
        hy = np.repeat(arr(f"{name}/{name}.position.y", evt), counts)
        hz = np.repeat(arr(f"{name}/{name}.position.z", evt), counts)
        cell = np.repeat(arr(f"{name}/{name}.cellID", evt), counts)
        rec_name = name[: -len("Hits")] + "RecHits"
        rec_cells = set(arr(f"{rec_name}/{rec_name}.cellID", evt).tolist()) \
            if rec_name in all_keys else set()
        t.append(ct)
        z.append(hz)
        r.append(np.hypot(hx, hy))
        e.append(arr(f"{c}/{c}.energy", evt))
        sig.append(origins[pidx] <= 2)
        kept.append(np.isin(cell, list(rec_cells)))
    return {k: np.concatenate(v) for k, v in
            [("t", t), ("z", z), ("r", r), ("e", e), ("sig", sig),
             ("kept", kept)]}


def two_panel(d, draw, title, out_path, colorbar_from=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, yv, ylab in [(axes[0], d["z"], "z [mm]"),
                         (axes[1], d["r"], "r [mm]")]:
        sc = draw(ax, yv)
        ax.set_ylabel(ylab)
        if not args.all_time:
            ax.set_xlim(-100, 2100)
    axes[0].legend(loc="upper right", fontsize=7, markerscale=2)
    axes[1].set_xlabel("deposit time [ns]")
    axes[0].set_title(title, fontsize=11)
    fig.tight_layout()
    if colorbar_from is not None:
        fig.colorbar(colorbar_from, ax=axes, pad=0.01,
                     label="contribution energy [GeV]")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


for evt in args.events:
    d = read_event(evt)
    n_beyond = int((d["t"] > 2100).sum())
    if args.all_time:
        beyond = f", t_max {d['t'].max():,.0f} ns"
    else:
        beyond = f", {n_beyond} beyond 2100 ns" if n_beyond else ""

    # 1 — energy gradient (log color scale; deposits span ~6 decades)
    norm = LogNorm(vmin=max(d["e"].min(), 1e-7), vmax=d["e"].max())
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, yv, ylab in [(axes[0], d["z"], "z [mm]"),
                         (axes[1], d["r"], "r [mm]")]:
        sc = ax.scatter(d["t"], yv, c=d["e"], s=3, cmap="viridis", norm=norm,
                        linewidths=0)
        ax.set_ylabel(ylab)
        if not args.all_time:
            ax.set_xlim(-100, 2100)
    axes[1].set_xlabel("deposit time [ns]")
    axes[0].set_title(f"{tag} event {evt} truth deposits by energy: "
                      f"{len(d['t'])} contributions{beyond}", fontsize=11)
    fig.colorbar(sc, ax=axes, pad=0.01, label="contribution energy [GeV]")
    fig.savefig(OUT / f"{tag}_evt{evt:04d}_egrad{SUFFIX}.png", dpi=110)
    plt.close(fig)
    print(f"wrote {OUT / f'{tag}_evt{evt:04d}_egrad{SUFFIX}.png'}")

    # 2 — survived digitization vs discarded
    e_kept = d["e"][d["kept"]].sum()
    frac_n = d["kept"].mean()
    frac_e = e_kept / max(d["e"].sum(), 1e-12)
    groups = [  # drawn in order: discarded under, survived on top
        (~d["sig"] & ~d["kept"], "#cfbcbc", 2, 1, "background, discarded"),
        (d["sig"] & ~d["kept"], "#b9cdb9", 2, 2, "signal, discarded"),
        (~d["sig"] & d["kept"], "#d62728", 6, 4, "background, survived"),
        (d["sig"] & d["kept"], "#1a9641", 6, 5, "signal, survived"),
    ]

    def draw_thresh(ax, yv):
        for m, c, s, zo, lab in groups:
            if m.any():
                ax.scatter(d["t"][m], yv[m], s=s, c=c, zorder=zo,
                           label=f"{lab} ({m.sum()})", linewidths=0)
        return None

    two_panel(d, draw_thresh,
              f"{tag} event {evt} deposits vs digitization: "
              f"{100*frac_n:.0f}% of deposits / {100*frac_e:.0f}% of energy "
              f"in surviving cells{beyond}",
              OUT / f"{tag}_evt{evt:04d}_thresh{SUFFIX}.png")

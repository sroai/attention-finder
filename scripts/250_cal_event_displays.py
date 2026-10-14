#!/usr/bin/env python
"""STEP 250 — calorimeter frame displays, colored like the tracker ones.

Calorimeter counterpart of plots/temporal/events_color/ (script 240): one
z-vs-t / r-vs-t picture per event, every energy deposit colored by background
source (generator-status band) and, for Geant4 secondaries, by ancestry.
Direct visual comparison with the tracker display of the same event.

WHAT GOES IN  : a dd4hep background sim file (.edm4hep.root) — calorimeter
                truth lives ONLY there (SimCalorimeterHit + contributions
                with MCParticle links; the reco files carry no calorimeter
                hit associations, see /work/cal_hit_bkg_plan.md).
WHAT COMES OUT: plots/phase9_cal_plan/0022_evtNNNN.png          (all detectors)
                plots/phase9_cal_plan/0022_evtNNNN_central.png  (no far-forward)

One marker per CONTRIBUTION (a cell aggregates several particles at several
times, so per-hit plotting would hide the time structure): position from the
parent cell, time from the contribution, category from the contributing
particle's own generatorStatus + origin — the same convention the tracker
CSVs use.

Albedo (origin 5) uses the tracker two-rule cut on RAW deposit times
(contribution time − particle vertex time > 100 ns, or − event t0 > 300 ns).
Tracker hit times are TOF-corrected toward the origin; these are not — the
same cut on raw times marks slightly more late deposits. Fine for a display.

Usage (run from OUTSIDE /work — its .venv is broken and hijacks uv):
    cd /tmp && uv run --no-project \
        --with uproot,awkward,numpy,matplotlib,polars \
        python ./scripts/250_cal_event_displays.py
    # other events / file:
    ... 250_cal_event_displays.py --events 6 27 --input <file.edm4hep.root>
    # proton-beam-gas showcase: scan the file, plot the 6 events whose
    # p-gas burst deposits most in the calorimeters (into pgas/):
    ... 250_cal_event_displays.py --pgas 6
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.temporal import COLOR_STYLE, status_category  # noqa: E402

DEFAULT_INPUT = ("/data/msf-2026-06/dd4hep-background/18x275/"
                 "k_lambda_18x275_5000evt_0022.edm4hep.root")
OUT = Path(__file__).resolve().parents[1] / "plots" / "phase9_cal_plan"

# Scope decision (cal_hit_bkg_plan.md §5): central + far-forward, no Lumi.
EXCLUDE = ("Lumi",)
FAR_FORWARD = {"B0ECalHits", "EcalFarForwardZDCHits", "HcalFarForwardZDCHits"}

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=DEFAULT_INPUT)
parser.add_argument("--events", type=int, nargs="+", default=[6, 27],
                    help="event numbers inside the file (default: 6 and 27, "
                         "the events with existing tracker displays)")
parser.add_argument("--clusters", default=None, metavar="RECO_FILE",
                    help="paired .edm4eic.root file: overlay reconstructed "
                         "clusters as hollow circles at (time, z/r), area "
                         "proportional to cluster energy")
parser.add_argument("--pgas", type=int, default=0, metavar="N",
                    help="instead of --events: scan the file for the N "
                         "events whose proton-beam-gas burst (status band "
                         "6000) deposits most in the calorimeters, and plot "
                         "those into a pgas/ subfolder")
args = parser.parse_args()

tree = uproot.open(args.input)["events"]
tag = Path(args.input).name.split("_")[-1].split(".")[0].lstrip("0") or "0"
tag = Path(args.input).name.rsplit("_", 1)[-1].split(".")[0]  # e.g. "0022"
OUT.mkdir(parents=True, exist_ok=True)

# calorimeter collections = those with a <name>Contributions companion
all_keys = {k.split("/")[0] for k in tree.keys()}
CAL = sorted(c[: -len("Contributions")] for c in all_keys
             if c.endswith("HitsContributions")
             and not any(x in c for x in EXCLUDE))
print(f"calorimeter collections: {', '.join(CAL)}")


def classify_gen_status(gen_status):
    """1 signal / 3 background / 0 unknown — same bands as csv_convert."""
    if gen_status in (1, 2):
        return 1
    if gen_status >= 1000:
        return 3
    return 0


def particle_origins(gs, par_begin, par_end, par_idx):
    """prt_origin per MCParticle: generator particles from their own status
    band; Geant4 secondaries (status 0) from the first generator ancestor.
    Memoized iterative walk — background showers reuse long parent chains."""
    memo = np.full(len(gs), -1, dtype=np.int8)
    for i in range(len(gs)):
        if memo[i] >= 0:
            continue
        if gs[i] != 0:
            memo[i] = classify_gen_status(gs[i])
            continue
        chain, j, res = [i], i, 0
        for _ in range(200):
            if par_begin[j] == par_end[j]:
                break
            p = par_idx[par_begin[j]]
            if gs[p] != 0:
                band = classify_gen_status(gs[p])
                res = 2 if band == 1 else 4 if band == 3 else 0
                break
            if memo[p] >= 0:            # chain already resolved from here
                res = memo[p]
                break
            chain.append(p)
            j = p
        memo[chain] = res
    return memo


def read_event(evt):
    """All calorimeter deposits of one event: t, z, r, plus the contributing
    particle's status/origin/vertex-time and the source collection."""
    lo, hi = evt, evt + 1

    def arr(branch):
        return np.asarray(tree[branch].array(entry_start=lo,
                                             entry_stop=hi)[0])

    gs = arr("MCParticles/MCParticles.generatorStatus")
    origins = particle_origins(
        gs,
        arr("MCParticles/MCParticles.parents_begin"),
        arr("MCParticles/MCParticles.parents_end"),
        arr("_MCParticles_parents/_MCParticles_parents.index"))
    vtx_time = arr("MCParticles/MCParticles.time")

    t, z, r, stat, org, vtx, coll = [], [], [], [], [], [], []
    for name in CAL:
        c = name + "Contributions"
        ct = arr(f"{c}/{c}.time")
        if not len(ct):
            continue
        pidx = arr(f"_{c}_particle/_{c}_particle.index")
        # parent cell per contribution: podio stores contributions
        # contiguously per hit, so repeat each hit over its range length
        counts = (arr(f"{name}/{name}.contributions_end")
                  - arr(f"{name}/{name}.contributions_begin"))
        hx = np.repeat(arr(f"{name}/{name}.position.x"), counts)
        hy = np.repeat(arr(f"{name}/{name}.position.y"), counts)
        hz = np.repeat(arr(f"{name}/{name}.position.z"), counts)
        t.append(ct)
        z.append(hz)
        r.append(np.hypot(hx, hy))
        stat.append(gs[pidx])
        org.append(origins[pidx])
        vtx.append(vtx_time[pidx])
        coll.append(np.full(len(ct), name))
    d = {k: np.concatenate(v) for k, v in
         [("t", t), ("z", z), ("r", r), ("status", stat),
          ("origin", org), ("vtx_time", vtx), ("coll", coll)]}

    # albedo relabel, tracker two-rule convention on raw deposit times
    sig = d["origin"] <= 2
    t0 = d["t"][d["origin"] == 1].min() if (d["origin"] == 1).any() else np.nan
    late = (d["t"] - d["vtx_time"] > 100) | (d["t"] - t0 > 300)
    d["origin"] = np.where(sig & late, 5, d["origin"])
    return d


def read_clusters(reco_tree, evt):
    """Reconstructed clusters of one event: t, z, r, energy. Main cluster
    collections only (Truth/SplitMerge/Baseline duplicates and Lumi skipped)."""
    keys = {k.split("/")[0] for k in reco_tree.keys()}
    colls = sorted(c for c in keys if c.endswith("Clusters")
                   and not any(x in c for x in
                               ("Truth", "SplitMerge", "Baseline", "Lumi")))
    t, z, r, en = [], [], [], []
    for c in colls:
        ct = np.asarray(reco_tree[f"{c}/{c}.time"]
                        .array(entry_start=evt, entry_stop=evt + 1)[0])
        if not len(ct):
            continue
        cx = np.asarray(reco_tree[f"{c}/{c}.position.x"]
                        .array(entry_start=evt, entry_stop=evt + 1)[0])
        cy = np.asarray(reco_tree[f"{c}/{c}.position.y"]
                        .array(entry_start=evt, entry_stop=evt + 1)[0])
        cz = np.asarray(reco_tree[f"{c}/{c}.position.z"]
                        .array(entry_start=evt, entry_stop=evt + 1)[0])
        ce = np.asarray(reco_tree[f"{c}/{c}.energy"]
                        .array(entry_start=evt, entry_stop=evt + 1)[0])
        t.append(ct); z.append(cz); r.append(np.hypot(cx, cy)); en.append(ce)
    if not t:
        return None
    return {k: np.concatenate(v) for k, v in
            [("t", t), ("z", z), ("r", r), ("e", en)]}


def display(d, mask, out_path, title, clusters=None):
    t, z, r = d["t"][mask], d["z"][mask], d["r"][mask]
    org, stat = d["origin"][mask], d["status"][mask]
    cats = np.array([status_category(st, o) for st, o in zip(stat, org)])
    sig = org <= 2
    t0, t1 = (t[sig].min(), t[sig].max()) if sig.any() else (np.nan, np.nan)
    n_late = int((t > 2100).sum())

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, yv, ylab in [(axes[0], z, "z [mm]"), (axes[1], r, "r [mm]")]:
        for cat, (c, sz, zo) in COLOR_STYLE.items():
            m = cats == cat
            if m.any():
                ax.scatter(t[m], yv[m], s=sz, c=c, zorder=zo,
                           label=f"{cat} ({m.sum()})", linewidths=0)
        if clusters is not None:
            cy = clusters["z"] if ylab.startswith("z") else clusters["r"]
            # area ~ energy: 60 pt^2 per GeV, floored so MeV clusters show
            ax.scatter(clusters["t"], cy, s=np.maximum(clusters["e"] * 60, 25),
                       facecolors="none", edgecolors="#2c5aa0", linewidths=1.4,
                       zorder=8, label=f"clusters ({len(cy)})")
        for x, ls in [(t0, "-"), (t1, "--")]:
            ax.axvline(x, color="#1a7f37", ls=ls, lw=1, alpha=0.8)
        ax.set_ylabel(ylab)
        if t.max() > 500:               # background frame: fixed 2 us view;
            ax.set_xlim(-100, 2100)     # signal-only file: autoscale (~TOF)
    axes[0].legend(loc="upper right", fontsize=7, markerscale=2)
    axes[1].set_xlabel("deposit time [ns]")
    suffix = f" ({n_late} beyond 2100 ns)" if n_late else ""
    axes[0].set_title(f"{title}: {mask.sum()} deposits, "
                      f"signal span {t0:.0f}..{t1:.0f} ns{suffix}",
                      fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def ancestor_bands(gs, par_begin, par_end, par_idx, particle_ids):
    """Generator-status band (2000, 3000, ... or 0 for signal) of each
    requested particle's first generator ancestor. Only the referenced
    particles are walked — contributions touch a small subset of the tens
    of thousands of MCParticles in a background frame."""
    memo = {}

    def band_of(i):
        if i in memo:
            return memo[i]
        j = int(i)
        for _ in range(200):
            if gs[j] != 0:
                b = (int(gs[j]) // 1000) * 1000 if gs[j] >= 1000 else 0
                memo[i] = b
                return b
            if par_begin[j] == par_end[j]:
                break
            j = par_idx[par_begin[j]]
        memo[i] = 0
        return 0

    return np.array([band_of(i) for i in particle_ids])


def scan_pgas(n_wanted):
    """Find the events whose proton-beam-gas burst reaches the calorimeters.

    Two passes: (1) a cheap sweep over MCParticles.generatorStatus counts
    band-6000 particles per event — most frames have none; (2) for the
    richest candidates, count calorimeter deposits whose ancestor band is
    6000 and rank by that. Prints a table with the burst time vs signal t0
    so bursts before and after the main event are visible in the pick."""
    import awkward as ak
    counts = []
    for chunk in tree.iterate(["MCParticles/MCParticles.generatorStatus"],
                              step_size="200 MB", library="ak"):
        gs = chunk["MCParticles/MCParticles.generatorStatus"]
        counts.append(ak.to_numpy(ak.sum((gs >= 6000) & (gs < 7000),
                                         axis=1)))
    n6000 = np.concatenate(counts)
    candidates = np.argsort(-n6000)[:40]
    candidates = [int(e) for e in candidates if n6000[e] > 0]
    print(f"events with p-gas particles: {(n6000 > 0).sum()} of {len(n6000)}"
          f"; checking top {len(candidates)} candidates")

    ranked = []
    for evt in candidates:
        lo, hi = evt, evt + 1
        gs = np.asarray(tree["MCParticles/MCParticles.generatorStatus"]
                        .array(entry_start=lo, entry_stop=hi)[0])
        pb = np.asarray(tree["MCParticles/MCParticles.parents_begin"]
                        .array(entry_start=lo, entry_stop=hi)[0])
        pe = np.asarray(tree["MCParticles/MCParticles.parents_end"]
                        .array(entry_start=lo, entry_stop=hi)[0])
        pi = np.asarray(tree["_MCParticles_parents/_MCParticles_parents"
                             ".index"].array(entry_start=lo, entry_stop=hi)[0])
        t_pg, n_pg = [], 0
        t0 = np.inf
        for name in CAL:
            c = name + "Contributions"
            ct = np.asarray(tree[f"{c}/{c}.time"]
                            .array(entry_start=lo, entry_stop=hi)[0])
            if not len(ct):
                continue
            pidx = np.asarray(tree[f"_{c}_particle/_{c}_particle.index"]
                              .array(entry_start=lo, entry_stop=hi)[0])
            bands = ancestor_bands(gs, pb, pe, pi, pidx)
            m = bands == 6000
            n_pg += int(m.sum())
            if m.any():
                t_pg.append(ct[m])
            sig = (gs[pidx] == 1) | (gs[pidx] == 2)
            if sig.any():
                t0 = min(t0, ct[sig].min())
        t_burst = float(np.median(np.concatenate(t_pg))) if t_pg else np.nan
        ranked.append((n_pg, evt, t_burst, t0))
        print(f"  evt {evt:4d}: {n_pg:6d} p-gas cal deposits, "
              f"burst t ~{t_burst:6.0f} ns, signal t0 {t0:6.0f} ns "
              f"({'before' if t_burst < t0 else 'after'} signal)")
    ranked.sort(reverse=True)
    return [(evt, tb, t0) for n, evt, tb, t0 in ranked[:n_wanted]]


if args.pgas:
    picks = scan_pgas(args.pgas)
    sub = OUT / "pgas"
    sub.mkdir(exist_ok=True)
    print(f"\nplotting {len(picks)} p-gas showcase events to {sub}")
    events, dest = [e for e, _, _ in picks], sub
else:
    events, dest = args.events, OUT

reco_tree = uproot.open(args.clusters)["events"] if args.clusters else None

for evt in events:
    d = read_event(evt)
    clus = read_clusters(reco_tree, evt) if reco_tree is not None else None
    every = np.ones(len(d["t"]), bool)
    central = ~np.isin(d["coll"], list(FAR_FORWARD))
    base = f"{tag} event {evt} calorimetry by source & G4 ancestry"
    display(d, every, dest / f"{tag}_evt{evt:04d}.png", base, clusters=clus)
    clus_central = clus
    if clus is not None:                # keep the central z axis central
        m = np.abs(clus["z"]) < 5200
        clus_central = {k: v[m] for k, v in clus.items()}
    display(d, central, dest / f"{tag}_evt{evt:04d}_central.png",
            base + " (no ZDC/B0)", clusters=clus_central)

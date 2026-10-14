#!/usr/bin/env python
"""STEP 251 — calorimeter frame displays from a cal_hits CSV.

Same picture as script 250 (z-vs-t / r-vs-t, tracker color convention), but
driven by the OUTPUT of the csv_cal pipeline stage (edm4eic_cal_hits.cxx:
digitized rec hits truth-labeled via the dd4hep join) instead of raw edm4hep
files — so the plot validates exactly what a training would consume.

One marker per rec hit; category from the hit's prt_gen_status + prt_origin
columns; albedo relabel applies the tracker two-rule cut on the DIGITIZED hit
time (cal_hit_time − prt_vtx_time > 100 ns, or − event t0 > 300 ns).

WHAT GOES IN  : <file>.cal_hits.csv[.zip]   (csv_cal stage output)
WHAT COMES OUT: plots/phase9_cal_plan/csv/<stem>_evtNNNN.png

Usage (run from OUTSIDE /work — its .venv is broken and hijacks uv):
    cd /tmp && uv run --no-project --with numpy,matplotlib,polars \
        python ./scripts/251_cal_csv_event_displays.py \
        --csv <path>.cal_hits.csv --events 6 27
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.temporal import COLOR_STYLE, status_category  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "plots" / "phase9_cal_plan" / "csv"

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="cal_hits.csv[.zip] to plot")
parser.add_argument("--events", type=int, nargs="+", default=[6, 27])
parser.add_argument("--all", action="store_true",
                    help="plot every event in the CSV (ignores --events)")
parser.add_argument("--no-tcut", action="store_true",
                    help="no 2100 ns axis limit — show the full time range "
                         "so late tails stay visible")
parser.add_argument("--outdir", default=None,
                    help="output folder (default: plots/phase9_cal_plan/csv)")
args = parser.parse_args()

OUT = Path(args.outdir) if args.outdir else OUT
OUT.mkdir(parents=True, exist_ok=True)
stem = Path(args.csv).name.split(".")[0]

df = pl.read_csv(args.csv)
# CSVs converted before 2026-09-30 carry the macro-era names (simulation-pipeline/eic2ai/column_renames.py).
df = df.rename({old: new for old, new in {"prt_status": "prt_gen_status", "rec_collection": "cal_hit_collection"}.items() if old in df.columns})
print(f"{len(df)} rec hits, {df['evt'].n_unique()} events; "
      f"origins: {dict(df['prt_origin'].value_counts().iter_rows())}")
print(f"label_purity < 1: {int((df['label_purity'] < 0.999).sum())} hits; "
      f"unmatched (n_contrib=0): {int((df['n_contrib'] == 0).sum())}")

events = (sorted(df["evt"].unique().to_list()) if args.all else args.events)
tail_counts = []
for evt in events:
    e = df.filter(pl.col("evt") == evt)
    if e.is_empty():
        print(f"evt {evt}: not in CSV -- skipped")
        continue
    t = e["cal_hit_time"].to_numpy()
    z = e["cal_hit_pos_z"].to_numpy()
    r = np.hypot(e["cal_hit_pos_x"].to_numpy(), e["cal_hit_pos_y"].to_numpy())
    org = e["prt_origin"].to_numpy().copy()
    stat = e["prt_gen_status"].to_numpy()
    vtx_t = e["prt_vtx_time"].to_numpy()
    energy = e["cal_hit_energy"].to_numpy()

    # albedo relabel, tracker two-rule convention on the digitized time
    sig = org <= 2
    t0_ev = t[org == 1].min() if (org == 1).any() else np.nan
    late = (t - vtx_t > 100) | (t - t0_ev > 300)
    org = np.where(sig & late, 5, org)

    cats = np.array([status_category(st, o) for st, o in zip(stat, org)])
    sig = org <= 2
    t0, t1 = (t[sig].min(), t[sig].max()) if sig.any() else (np.nan, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, yv, ylab in [(axes[0], z, "z [mm]"), (axes[1], r, "r [mm]")]:
        for cat, (c, sz, zo) in COLOR_STYLE.items():
            m = cats == cat
            if m.any():
                ax.scatter(t[m], yv[m], s=sz, c=c, zorder=zo,
                           label=f"{cat} ({m.sum()})", linewidths=0)
        for x, ls in [(t0, "-"), (t1, "--")]:
            ax.axvline(x, color="#1a7f37", ls=ls, lw=1, alpha=0.8)
        ax.set_ylabel(ylab)
        if t.max() > 500 and not args.no_tcut:
            ax.set_xlim(-100, 2100)
    axes[0].legend(loc="upper right", fontsize=7, markerscale=2)
    axes[1].set_xlabel("rec hit time [ns]")
    n_late = int((t > 2000).sum())
    tail_counts.append((evt, n_late, float(t.max())))
    if args.no_tcut:
        suffix = f" ({n_late} past frame end, t_max {t.max():,.0f} ns)"
    else:
        suffix = f" ({n_late} beyond 2000 ns)" if n_late and t.max() > 500 else ""
    axes[0].set_title(f"{stem} event {evt} cal_hits.csv (digitized rec hits): "
                      f"{len(e)} hits, {energy.sum():.2f} GeV, "
                      f"signal span {t0:.0f}..{t1:.0f} ns{suffix}", fontsize=11)
    fig.tight_layout()
    path = OUT / f"{stem}_evt{evt:04d}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    if not args.all or evt % 50 == 0:
        print(f"wrote {path}")

if args.all and tail_counts:
    n_tail = np.array([n for _, n, _ in tail_counts])
    t_max = np.array([m for _, _, m in tail_counts])
    print(f"\nTAIL SUMMARY over {len(tail_counts)} frames "
          f"(rec hits with t > 2000 ns):")
    print(f"  frames with a tail: {(n_tail > 0).sum()} "
          f"({100 * (n_tail > 0).mean():.0f}%)")
    print(f"  tail hits/frame: mean {n_tail.mean():.1f}, "
          f"median {np.median(n_tail):.0f}, max {n_tail.max()}")
    print(f"  t_max percentiles 50/90/99: {np.percentile(t_max, 50):,.0f} / "
          f"{np.percentile(t_max, 90):,.0f} / "
          f"{np.percentile(t_max, 99):,.0f} ns; overall {t_max.max():,.0f} ns")
    worst = sorted(tail_counts, key=lambda x: -x[1])[:8]
    print("  longest tails (evt, n_tail, t_max):",
          [(e, n, round(m)) for e, n, m in worst])

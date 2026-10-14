"""Data-understanding plots and stats for trk_hits parquet files.

Everything here is *descriptive*: it answers "what does the data look like
and which features separate signal from background?" before any ML is done.
All plots compare the two classes (signal=keep vs background=filter), because
class separation is what the model will have to learn.

Memory strategy: we load only the needed columns from parquet into numpy
(~1 GB for a 35M-hit file, fine on this machine); histograms are then cheap.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .labels import ORIGIN_NAMES
from .systems import SYSTEM_NAMES, system_name

# Columns needed for analysis (model inputs + a few context extras).
ANALYSIS_COLUMNS = [
    "evt", "prt_origin", "trk_hit_system_id",
    "trk_hit_pos_x", "trk_hit_pos_y", "trk_hit_pos_z",
    "trk_hit_time", "trk_hit_edep",
]

SIG_COLOR, BKG_COLOR = "#1a7f37", "#c0392b"


def load_analysis_frame(parquet_path: str | Path) -> pl.DataFrame:
    return pl.scan_parquet(parquet_path).select(ANALYSIS_COLUMNS).collect()


def compute_stats(df: pl.DataFrame) -> dict:
    """Headline numbers: class balance, per-origin/per-system composition."""
    origin_counts = {
        f"{o} ({ORIGIN_NAMES.get(o, '?')})": int(n)
        for o, n in df.group_by("prt_origin").len().sort("prt_origin").iter_rows()
    }
    n = df.height
    n_sig = df.filter(pl.col("prt_origin").is_in([1, 2])).height
    n_bkg = df.filter(pl.col("prt_origin").is_in([3, 4])).height
    n_unk = df.filter(pl.col("prt_origin") == 0).height

    per_system = (
        df.group_by("trk_hit_system_id")
        .agg(
            pl.len().alias("hits"),
            (pl.col("prt_origin") <= 2).sum().alias("signal_hits"),
        )
        .sort("trk_hit_system_id")
    )
    per_event = df.group_by("evt").agg(
        pl.len().alias("hits"),
        (pl.col("prt_origin") <= 2).sum().alias("signal_hits"),
    )

    return {
        "total_hits": n,
        "events": df["evt"].n_unique(),
        "origin_counts": origin_counts,
        "unknown_origin_hits": n_unk,
        "signal_hits": n_sig,
        "background_hits": n_bkg,
        "signal_fraction": n_sig / max(n_sig + n_bkg, 1),
        "hits_per_event": {
            "mean": float(per_event["hits"].mean()),
            "min": int(per_event["hits"].min()),
            "max": int(per_event["hits"].max()),
        },
        "signal_hits_per_event_mean": float(per_event["signal_hits"].mean()),
        "per_system": {
            str(sid): {"name": SYSTEM_NAMES.get(int(sid), "?"),
                       "hits": int(h), "signal_hits": int(s),
                       "signal_fraction": s / max(h, 1)}
            for sid, h, s in per_system.iter_rows()
        },
    }


def _class_masks(df: pl.DataFrame):
    """(signal, background) masks; albedo (5) belongs to NEITHER class."""
    origin = df["prt_origin"].to_numpy()
    return origin <= 2, (origin == 3) | (origin == 4)


def _hist_pair(ax, sig_vals, bkg_vals, bins, xlabel, log=True):
    """Overlaid density histograms so shapes are comparable despite the
    ~200x class imbalance (raw-count overlays would hide the signal)."""
    ax.hist(bkg_vals, bins=bins, density=True, histtype="stepfilled",
            alpha=0.45, color=BKG_COLOR, label="background (filter)")
    ax.hist(sig_vals, bins=bins, density=True, histtype="step",
            lw=1.8, color=SIG_COLOR, label="signal (keep)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8)


def plot_all(df: pl.DataFrame, out_dir: str | Path, tag: str) -> list[Path]:
    """Write the standard data-understanding plot set; returns file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    sig_m, bkg_m = _class_masks(df)
    x = df["trk_hit_pos_x"].to_numpy()
    y = df["trk_hit_pos_y"].to_numpy()
    z = df["trk_hit_pos_z"].to_numpy()
    t = df["trk_hit_time"].to_numpy()
    r = np.hypot(x, y)
    edep = df["trk_hit_edep"].to_numpy()

    def save(fig, name):
        p = out_dir / f"{tag}_{name}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)

    # 1. Class composition by origin flag
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df.group_by("prt_origin").len().sort("prt_origin")
    labels = [f"{o}\n{ORIGIN_NAMES.get(o, '?')}" for o in counts["prt_origin"]]
    colors = [SIG_COLOR if o <= 2 else ("#8e5aa0" if o == 5 else BKG_COLOR)
              for o in counts["prt_origin"]]
    ax.bar(labels, counts["len"], color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("hits")
    ax.set_title(f"{tag}: hit counts by prt_origin (log)")
    save(fig, "origin_counts")

    # 2. Hits per event (totals and signal-only)
    per_evt = df.group_by("evt").agg(
        pl.len().alias("hits"), (pl.col("prt_origin") <= 2).sum().alias("sig"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(per_evt["hits"].to_numpy(), bins=60, color="#444")
    axes[0].set_xlabel("hits / event")
    axes[0].set_ylabel("events")
    axes[1].hist(per_evt["sig"].to_numpy(), bins=60, color=SIG_COLOR)
    axes[1].set_xlabel("signal hits / event")
    fig.suptitle(f"{tag}: event sizes")
    save(fig, "hits_per_event")

    # 3. Per-system composition — which detectors are background-dominated
    per_sys = (df.group_by("trk_hit_system_id")
               .agg(pl.len().alias("hits"),
                    (pl.col("prt_origin") <= 2).sum().alias("sig"))
               .sort("trk_hit_system_id"))
    sids = [system_name(s) for s in per_sys["trk_hit_system_id"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].bar(sids, per_sys["hits"], color=BKG_COLOR, label="all hits")
    axes[0].bar(sids, per_sys["sig"], color=SIG_COLOR, label="signal")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("hits")
    axes[0].legend()
    frac = (per_sys["sig"] / per_sys["hits"]).to_numpy()
    axes[1].bar(sids, 100 * frac, color="#2c5aa0")
    axes[1].set_ylabel("signal fraction [%]")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.suptitle(f"{tag}: composition per detector system")
    save(fig, "per_system")

    # 4. Model-input feature distributions, signal vs background
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _hist_pair(axes[0, 0], z[sig_m], z[bkg_m],
               np.linspace(np.percentile(z, 0.1), np.percentile(z, 99.9), 120),
               "z [mm]")
    _hist_pair(axes[0, 1], r[sig_m], r[bkg_m],
               np.linspace(0, np.percentile(r, 99.9), 120), "r = sqrt(x²+y²) [mm]")
    tmax = np.percentile(t, 99.5)
    _hist_pair(axes[1, 0], t[sig_m], np.clip(t[bkg_m], None, tmax),
               np.linspace(t.min(), tmax, 120), "time [ns] (clipped @99.5%)")
    e_bins = np.logspace(np.log10(max(edep[edep > 0].min(), 1e-9)),
                         np.log10(edep.max()), 120)
    _hist_pair(axes[1, 1], edep[sig_m], edep[bkg_m], e_bins, "edep [GeV]")
    axes[1, 1].set_xscale("log")
    fig.suptitle(f"{tag}: model-input features by class")
    save(fig, "features")

    # 5. Time zoom around the physics window — background is expected to be
    #    out-of-time (beam gas / synchrotron arrive at other bunch crossings)
    fig, ax = plt.subplots(figsize=(7, 4))
    _hist_pair(ax, t[sig_m & (np.abs(t) < 60)], t[bkg_m & (np.abs(t) < 60)],
               np.linspace(-60, 60, 240), "time [ns] (zoom)")
    ax.set_title(f"{tag}: hit time near the trigger window")
    save(fig, "time_zoom")

    # 6. r-z occupancy maps: where in the detector each class lives
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    zr_bins = [np.linspace(np.percentile(z, 0.1), np.percentile(z, 99.9), 200),
               np.linspace(0, np.percentile(r, 99.9), 200)]
    for ax, m, name in [(axes[0], sig_m, "signal"), (axes[1], bkg_m, "background")]:
        h = ax.hist2d(z[m], r[m], bins=zr_bins, norm=matplotlib.colors.LogNorm())
        ax.set_xlabel("z [mm]")
        ax.set_title(name)
        fig.colorbar(h[3], ax=ax)
    axes[0].set_ylabel("r [mm]")
    fig.suptitle(f"{tag}: hit occupancy in r-z")
    save(fig, "occupancy_rz")

    return written


def run_analysis(parquet_path, plots_dir, stats_path, tag) -> dict:
    df = load_analysis_frame(parquet_path)
    stats = compute_stats(df)
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(stats, indent=2))
    plot_all(df, plots_dir, tag)
    return stats

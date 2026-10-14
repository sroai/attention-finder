"""Phase-2 temporal analysis: when does the signal live inside a frame?

Physics context (from the task): one event = one 2000 ns streaming-readout
frame. Exactly ONE physics interaction is placed at a random time inside the
frame; its final-state particles produce hits over ~100-200 ns. Everything
else is background, arriving continuously. Phase-1 concluded "time is
uniform for both classes" — true only because the signal start time t0 is
random per frame; ALIGNED to t0 the signal must be a sharp burst. This
module measures exactly that.

Definitions used throughout:
    t0      = earliest hit time of prt_origin in {1,2} in the event
    t_last  = latest such hit time
    dt      = hit_time - t0 (time since signal start)

All outputs go to plots/temporal/ (phase-1 plots are never touched).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ORIGIN_STYLE = {  # color, size, zorder — signal drawn on top of background
    1: ("#1a7f37", 14, 3, "signal (1)"),
    2: ("#7fc97f", 10, 3, "signal secondary (2)"),
    3: ("#c9c9c9", 2, 1, "background (3)"),
    4: ("#e78ac3", 4, 2, "background secondary (4)"),
    5: ("#8e5aa0", 18, 4, "signal albedo (5, don't-care)"),
}

COLUMNS = ["evt", "prt_origin", "prt_gen_status", "trk_hit_pos_x",
           "trk_hit_pos_y", "trk_hit_pos_z", "trk_hit_time",
           "trk_hit_system_id"]

# Parquet files built before 2026-09-30 carry the macro-era name of the generator
# status (the CSV vocabulary was unified in simulation-pipeline/eic2ai/column_renames.py).
LEGACY_COLUMNS = {"prt_status": "prt_gen_status"}

# Generator-status bands (merger convention, see simulation-pipeline stage 22
# docstring: each background source's generatorStatus is offset by a band —
# "synrad=2000, ebrems=3000, ...". Band 0 = signal; status 0 = created by
# Geant4, whose ancestry must be read from prt_origin instead).
# full source list from the 18x275 hgas cocktail JSON
# (submodules/simulation-pipeline/background_cocktails/):
STATUS_BAND_NAMES = {2000: "synrad", 3000: "e-gas brems",
                     4000: "e-gas coulomb", 5000: "e-gas touschek",
                     6000: "p-gas (pythia)"}


def status_category(status: int, origin: int) -> str:
    """Human label combining generator status band with G4 ancestry."""
    if origin == 5:
        return "signal albedo (late)"
    if status == 0:  # Geant4 secondary: band unknown, ancestry from origin
        return "G4 sec. of signal" if origin <= 2 else "G4 sec. of background"
    if status < 1000:
        return "signal (generator)"
    band = (status // 1000) * 1000
    return f"bkg {band} ({STATUS_BAND_NAMES.get(band, 'source?')})"


def load_events(parquet_path):
    """Returns dict of numpy arrays for the whole file, plus event ranges."""
    lazy = pl.scan_parquet(parquet_path)
    lazy = lazy.rename({old: new for old, new in LEGACY_COLUMNS.items() if old in lazy.collect_schema().names()})
    df = lazy.select(COLUMNS).collect()
    evt = df["evt"].to_numpy()
    change = np.flatnonzero(np.diff(evt.astype(np.int64))) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(evt)]])
    return {
        "evt": evt,
        "origin": df["prt_origin"].to_numpy(),
        "status": df["prt_gen_status"].to_numpy(),
        "t": df["trk_hit_time"].to_numpy(),
        "z": df["trk_hit_pos_z"].to_numpy(),
        "r": np.hypot(df["trk_hit_pos_x"].to_numpy(),
                      df["trk_hit_pos_y"].to_numpy()),
        "sys": df["trk_hit_system_id"].to_numpy(),
        "starts": starts, "ends": ends,
    }


def event_display(d, i, out_path, tag):
    """Classic picture for one event: z vs t and r vs t, colored by origin,
    with vertical lines at the signal's first and last hit time."""
    s, e = d["starts"][i], d["ends"][i]
    t, z, r, org = d["t"][s:e], d["z"][s:e], d["r"][s:e], d["origin"][s:e]
    sig = org <= 2
    t0, t1 = (t[sig].min(), t[sig].max()) if sig.any() else (np.nan, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, yv, ylab in [(axes[0], z, "z [mm]"), (axes[1], r, "r [mm]")]:
        for o, (c, sz, zo, lab) in ORIGIN_STYLE.items():
            m = org == o
            if m.any():
                ax.scatter(t[m], yv[m], s=sz, c=c, zorder=zo, label=lab,
                           linewidths=0)
        for x, ls in [(t0, "-"), (t1, "--")]:
            ax.axvline(x, color="#1a7f37", ls=ls, lw=1, alpha=0.8)
        ax.set_ylabel(ylab)
    axes[0].legend(loc="upper right", fontsize=7, markerscale=2)
    axes[1].set_xlabel("hit time [ns]")
    n_sig = int(sig.sum())
    axes[0].set_title(f"{tag} event {d['evt'][s]}: {e-s} hits, {n_sig} signal, "
                      f"signal span {t0:.0f}..{t1:.0f} ns "
                      f"(length {t1-t0:.0f} ns)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# display style per status category: color, size, zorder (signal on top)
STATUS_STYLE = {
    "signal albedo (late)":  ("#8e5aa0", 18, 6),
    "signal (generator)":    ("#1a7f37", 16, 5),
    "G4 sec. of signal":     ("#7fc97f", 11, 5),
    "G4 sec. of background": ("#e78ac3", 5, 3),
    "bkg 2000 (synrad)":     ("#c9c9c9", 2, 1),
    "bkg 3000 (ebrems)":     ("#2c5aa0", 8, 2),
    "bkg 4000 (source?)":    ("#e6a03c", 8, 2),
    "bkg 5000 (source?)":    ("#8e5aa0", 8, 2),
    "bkg 6000 (source?)":    ("#7a4a2b", 8, 2),
}


def event_display_status(d, i, out_path, tag):
    """Like event_display, but colored by GENERATOR-status category:
    which background source (status band) produced each hit, and for
    Geant4-created hits (status 0), whether their ancestor was signal or
    background (prt_origin). Same z/r vs t layout and t0/t_last lines."""
    s, e = d["starts"][i], d["ends"][i]
    t, z, r = d["t"][s:e], d["z"][s:e], d["r"][s:e]
    org, stat = d["origin"][s:e], d["status"][s:e]
    cats = np.array([status_category(st, o) for st, o in zip(stat, org)])
    sig = org <= 2
    t0, t1 = (t[sig].min(), t[sig].max()) if sig.any() else (np.nan, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, yv, ylab in [(axes[0], z, "z [mm]"), (axes[1], r, "r [mm]")]:
        for cat, (c, sz, zo) in STATUS_STYLE.items():
            m = cats == cat
            if m.any():
                ax.scatter(t[m], yv[m], s=sz, c=c, zorder=zo,
                           label=f"{cat} ({m.sum()})", linewidths=0)
        for x, ls in [(t0, "-"), (t1, "--")]:
            ax.axvline(x, color="#1a7f37", ls=ls, lw=1, alpha=0.8)
        ax.set_ylabel(ylab)
    axes[0].legend(loc="upper right", fontsize=7, markerscale=2)
    axes[1].set_xlabel("hit time [ns]")
    axes[0].set_title(f"{tag} event {d['evt'][s]} by generator status: "
                      f"{e-s} hits, signal span {t0:.0f}..{t1:.0f} ns")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# fully-colored variant: every background SOURCE gets a real color (synrad
# included — not gray), and G4-created hits (status 0) are colored by their
# ANCESTRY from prt_origin: made by a signal particle vs by background.
COLOR_STYLE = {
    "signal (generator)":    ("#1a7f37", 16, 7),   # dark green, on top
    "G4 sec. of signal":     ("#7fc97f", 11, 6),   # light green
    "signal albedo (late)":  ("#8e5aa0", 18, 6),   # purple
    "G4 sec. of background": ("#d62728", 5, 4),    # red
    "bkg 2000 (synrad)":         ("#f0b429", 2, 1),  # amber haze
    "bkg 3000 (e-gas brems)":    ("#2c5aa0", 8, 3),  # blue
    "bkg 4000 (e-gas coulomb)":  ("#17becf", 9, 3),  # teal
    "bkg 5000 (e-gas touschek)": ("#d02090", 9, 3),  # magenta
    "bkg 6000 (p-gas (pythia))": ("#7a4a2b", 8, 3),  # brown
}


def event_display_color(d, i, out_path, tag):
    """Classic z/r-vs-t picture, background colored BY SOURCE (status band:
    synrad amber, e-gas blue/teal/magenta, p-gas brown) and status-0 G4
    secondaries colored by ancestry (prt_origin): green family = made by the
    signal, red = made by background. Same t0/t_last lines as event_display."""
    s, e = d["starts"][i], d["ends"][i]
    t, z, r = d["t"][s:e], d["z"][s:e], d["r"][s:e]
    org, stat = d["origin"][s:e], d["status"][s:e]
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
    axes[0].legend(loc="upper right", fontsize=7, markerscale=2)
    axes[1].set_xlabel("hit time [ns]")
    axes[0].set_title(f"{tag} event {d['evt'][s]} by background source & G4 "
                      f"ancestry: {e-s} hits, "
                      f"signal span {t0:.0f}..{t1:.0f} ns")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def window_display(t, z, origin, t0_true, t1_true, win_lo, win_hi,
                   out_path, title):
    """Phase-3 validation view: one frame's z vs t with the TRUE signal span
    as solid/dashed lines and the ESTIMATED window as a transparent band —
    at a glance: does the band cover the green burst, and how much haze
    rides along?"""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for o, (c, sz, zo, lab) in ORIGIN_STYLE.items():
        m = origin == o
        if m.any():
            ax.scatter(t[m], z[m], s=sz, c=c, zorder=zo, label=lab,
                       linewidths=0)
    ax.axvspan(win_lo, win_hi, color="#2c5aa0", alpha=0.15, zorder=0,
               label="estimated window")
    ax.axvline(t0_true, color="#1a7f37", ls="-", lw=1.4, label="true start")
    ax.axvline(t1_true, color="#1a7f37", ls="--", lw=1.4, label="true end")
    ax.set_xlabel("hit time [ns]")
    ax.set_ylabel("z [mm]")
    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def per_event_signal_stats(d) -> dict:
    """Per-event timing quantities of signal (origin 1/2) hits.

    Returns arrays over events: t0, t_last, duration, p95 duration
    (t0 -> 95th percentile, robust against single late stragglers),
    max_gap (largest gap between consecutive signal hit times — a second
    interaction in the frame would show up as a huge gap), n_sig,
    frac_within_200 (fraction of signal hits in [t0, t0+200 ns]).
    """
    n_ev = len(d["starts"])
    out = {k: np.full(n_ev, np.nan) for k in
           ["t0", "t_last", "duration", "duration_p95", "max_gap",
            "n_sig", "frac_within_200"]}
    for i, (s, e) in enumerate(zip(d["starts"], d["ends"])):
        ts = np.sort(d["t"][s:e][d["origin"][s:e] <= 2])
        if not len(ts):
            continue
        out["n_sig"][i] = len(ts)
        out["t0"][i] = ts[0]
        out["t_last"][i] = ts[-1]
        out["duration"][i] = ts[-1] - ts[0]
        out["duration_p95"][i] = np.percentile(ts, 95) - ts[0]
        out["max_gap"][i] = np.diff(ts).max() if len(ts) > 1 else 0.0
        out["frac_within_200"][i] = (ts <= ts[0] + 200).mean()
    return out


def aligned_dt(d, stats):
    """dt = t - t0(event) for every hit; returns (dt, is_signal)."""
    t0_per_hit = np.repeat(stats["t0"],
                           (d["ends"] - d["starts"]).astype(np.int64))
    return d["t"] - t0_per_hit, d["origin"] <= 2


def t0_recovery_check(d, stats, bin_ns=25.0, systems=None):
    """Can the signal start time be found WITHOUT labels?

    Method: histogram all hit times of the event in `bin_ns` bins, optionally
    restricted to given detector systems, take the fullest bin as the t0
    estimate. Returns per-event |estimate - true t0| (lower edge of peak bin).
    This directly measures whether a deployment-time model could self-locate
    the physics burst inside the frame.
    """
    n_ev = len(d["starts"])
    err = np.full(n_ev, np.nan)
    edges = np.arange(-100, 2100 + bin_ns, bin_ns)
    for i, (s, e) in enumerate(zip(d["starts"], d["ends"])):
        t = d["t"][s:e]
        if systems is not None:
            t = t[np.isin(d["sys"][s:e], systems)]
        if len(t) < 5 or np.isnan(stats["t0"][i]):
            continue
        counts, _ = np.histogram(t, bins=edges)
        err[i] = edges[np.argmax(counts)] - stats["t0"][i]
    return err

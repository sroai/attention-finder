#!/usr/bin/env python
"""Phase-2 temporal analysis: event displays, signal timing stats, t0 studies.

Usage (from the repo root):
    uv run python scripts/220_temporal_analysis.py

Outputs (all NEW, phase-1 plots untouched):
    plots/temporal/events/   z-vs-t and r-vs-t displays, first 50 events of 0022
    plots/temporal/*.png     timing histograms / aligned distributions
    reports/temporal_stats.json  headline numbers (both files)
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
from trkbkg import temporal as T

ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = (CFG.parquet_dir / "18x275_v3")
PLOTS = ROOT / "plots" / "temporal"
EVENTS_DIR = PLOTS / "events"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

SIG_COLOR, BKG_COLOR = "#1a7f37", "#c0392b"
summary = {}

# ---------------------------------------------------------------- load ----
data = {tag: T.load_events(PARQUET_DIR / f"{tag}.trk_hits.parquet")
        for tag in ("0022", "0023")}

# ------------------------------------------------- 1. event displays ------
STATUS_DIR = PLOTS / "events_status"
STATUS_DIR.mkdir(parents=True, exist_ok=True)
print("[events] plotting first 50 events of 0022 (origin + status views)",
      flush=True)
for i in range(50):
    T.event_display(data["0022"], i, EVENTS_DIR / f"0022_evt{i:04d}.png", "0022")
    T.event_display_status(data["0022"], i,
                           STATUS_DIR / f"0022_evt{i:04d}_status.png", "0022")

# ------------------------------------- 2. per-event signal timing stats ---
stats = {tag: T.per_event_signal_stats(d) for tag, d in data.items()}
st = {k: np.concatenate([stats["0022"][k], stats["0023"][k]])
      for k in stats["0022"]}
valid = ~np.isnan(st["t0"])
summary["events_total"] = int(len(valid))
summary["events_without_signal_hits"] = int((~valid).sum())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist(st["t0"][valid], bins=80, color=SIG_COLOR)
axes[0, 0].set_xlabel("signal start t0 [ns]")
axes[0, 0].set_ylabel("events")
axes[0, 0].set_title("where the interaction lands in the frame")
axes[0, 1].hist(st["duration"][valid], bins=80, color="#2c5aa0")
axes[0, 1].set_xlabel("signal duration t_last - t0 [ns]")
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("full signal span (log)")
axes[1, 0].hist(st["duration_p95"][valid], bins=80, color="#2c5aa0")
axes[1, 0].set_xlabel("robust duration t0 -> p95 [ns]")
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("span holding 95% of signal hits (log)")
axes[1, 1].hist(st["frac_within_200"][valid], bins=50, color="#2c5aa0")
axes[1, 1].set_xlabel("fraction of signal hits within t0+200 ns")
axes[1, 1].set_title("how much signal is 'prompt'")
fig.suptitle("signal timing per event (0022+0023, 10k events)")
fig.tight_layout()
fig.savefig(PLOTS / "signal_timing.png", dpi=130)
plt.close(fig)

for k in ("t0", "duration", "duration_p95", "max_gap", "frac_within_200"):
    v = st[k][valid]
    summary[k] = {"mean": float(np.mean(v)), "median": float(np.median(v)),
                  "p95": float(np.percentile(v, 95)),
                  "max": float(np.max(v))}

# --------------------------- 3. single-interaction (cluster gap) check ----
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(st["max_gap"][valid], bins=100, color="#2c5aa0")
ax.set_xlabel("largest gap between consecutive signal hit times [ns]")
ax.set_ylabel("events")
ax.set_yscale("log")
ax.axvline(500, color="k", ls="--", lw=1, label="500 ns")
ax.legend()
ax.set_title("second-interaction check: max intra-signal time gap")
fig.tight_layout()
fig.savefig(PLOTS / "signal_max_gap.png", dpi=130)
plt.close(fig)
summary["events_with_signal_gap_gt_500ns"] = int((st["max_gap"][valid] > 500).sum())
summary["events_with_signal_gap_gt_200ns"] = int((st["max_gap"][valid] > 200).sum())

# ------------------------- 4. aligned time: dt = t - t0 per hit -----------
dts, sigs = [], []
for tag in data:
    dt, is_sig = T.aligned_dt(data[tag], stats[tag])
    ok = ~np.isnan(dt)
    dts.append(dt[ok])
    sigs.append(is_sig[ok])
dt = np.concatenate(dts)
is_sig = np.concatenate(sigs)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
bins = np.linspace(-2000, 2000, 400)
for ax in axes:
    ax.hist(dt[~is_sig], bins=bins, density=True, alpha=0.45,
            color=BKG_COLOR, label="background")
    ax.hist(dt[is_sig], bins=bins, density=True, histtype="step",
            lw=1.6, color=SIG_COLOR, label="signal")
    ax.set_xlabel("t - t0(event) [ns]")
    ax.set_ylabel("density")
    ax.legend()
axes[0].set_yscale("log")
axes[1].set_xlim(-100, 400)
axes[1].set_title("zoom on the burst")
fig.suptitle("hit times ALIGNED to the event's signal start "
             "(the feature phase 1 was missing)")
fig.tight_layout()
fig.savefig(PLOTS / "aligned_dt.png", dpi=130)
plt.close(fig)

# purity vs dt: what fraction of hits in [t0+a, t0+b) windows is signal
edges = np.array([-2000, -100, 0, 25, 50, 100, 150, 200, 300, 500, 1000, 2000])
purity = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (dt >= a) & (dt < b)
    purity.append(float(is_sig[m].mean()) if m.any() else 0.0)
summary["signal_fraction_by_dt_window"] = {
    f"[{int(a)},{int(b)})": p for a, b, p in zip(edges[:-1], edges[1:], purity)}

# ---------------- 5. is t0 recoverable without labels? --------------------
recovery = {}
for label, systems in [
        ("all_systems", None),
        ("vertex_barrel_31", [31]),
        ("barrel_trackers_31_59_60_61_64", [31, 59, 60, 61, 64]),
        ("barrel_tof_92", [92])]:
    errs = np.concatenate([T.t0_recovery_check(d, stats[tag], 25.0, systems)
                           for tag, d in data.items()])
    ok = ~np.isnan(errs)
    recovery[label] = {
        "events_evaluated": int(ok.sum()),
        "median_abs_err_ns": float(np.median(np.abs(errs[ok]))),
        "frac_within_50ns": float((np.abs(errs[ok]) <= 50).mean()),
        "frac_within_100ns": float((np.abs(errs[ok]) <= 100).mean()),
    }
summary["t0_recovery_peak_bin_method"] = recovery

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for label, systems, color in [("all systems", None, "#777777"),
                              ("VertexBarrel(31)", [31], "#2c5aa0"),
                              ("barrel trackers", [31, 59, 60, 61, 64], SIG_COLOR),
                              ("BarrelTOF(92)", [92], "#c0392b")]:
    errs = np.concatenate([T.t0_recovery_check(d, stats[tag], 25.0, systems)
                           for tag, d in data.items()])
    errs = errs[~np.isnan(errs)]
    ax.hist(np.clip(errs, -500, 500), bins=100, histtype="step", lw=1.5,
            label=label, color=color)
ax.set_xlabel("t0 estimate - true t0 [ns] (peak 25 ns bin method, clipped)")
ax.set_ylabel("events")
ax.set_yscale("log")
ax.legend(fontsize=8)
ax.set_title("label-free t0 recovery from hit-time histogram peak")
fig.tight_layout()
fig.savefig(PLOTS / "t0_recovery.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------- report ----
# merge-preserving write: keys added by other scripts (e.g. 09's
# score-weighted t0 check) must survive a re-run of this one
out = ROOT / "reports" / "temporal_stats.json"
if out.exists():
    summary = {**json.loads(out.read_text()), **summary}
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"wrote {out} and plots to {PLOTS}")

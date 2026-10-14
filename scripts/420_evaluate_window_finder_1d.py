#!/usr/bin/env python
"""Evaluate the window finder on the test fold.

Usage (from the repo root):
    uv run python scripts/420_evaluate_window_finder_1d.py <run_name>

Outputs:
    runs/<run_name>/metrics.json
    plots/window/<run_name>_*.png
        _error_hist    signed t0 error (late = cuts the burst front = BAD)
        _containment   signal containment / background-inside vs (front, back)
        _examples      a few test frames with the predicted bin marked

The containment table is the phase-4 input: it answers "if we keep
[t0_est - front, t0_est + back], how much signal survives and how much
background comes along" for a grid of margins. No filtering is applied here.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.dataset import load_fold
from trkbkg.windows import BIN_NS, T_MIN, WindowFinderCNN, containment_scan

HIT_DATA = (CFG.dataset_dir / "18x275_v3")
WIN_DATA = (CFG.dataset_dir / "windows_v4")
RUNS = CFG.runs_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "window"
PLOTS.mkdir(parents=True, exist_ok=True)

run_name = sys.argv[1]
run_dir = RUNS / run_name
ckpt = torch.load(run_dir / "best.pt", weights_only=False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = WindowFinderCNN(width=ckpt["config"]["width"]).to(dev).eval()
model.load_state_dict(ckpt["model_state"])

te = np.load(WIN_DATA / "test.npz")
with torch.no_grad():
    logits = model(torch.from_numpy(te["frames"]).to(dev)).cpu().numpy()
pred_bin = logits.argmax(axis=1)
t0_est = pred_bin * BIN_NS + T_MIN          # bin lower edge
err = t0_est - te["t0"]                     # >0 = LATE (cuts burst front)
err_c = err + BIN_NS / 2                    # bin-center error for tolerances

metrics = {
    "run": run_name, "epoch": ckpt["epoch"], "test_frames": len(err),
    "exact_bin": float((pred_bin == te["target"]).mean()),
    "within_25ns": float((np.abs(err_c) <= 25).mean()),
    "within_50ns": float((np.abs(err_c) <= 50).mean()),
    "within_100ns": float((np.abs(err_c) <= 100).mean()),
    "median_abs_err_ns": float(np.median(np.abs(err_c))),
    "late_by_gt25ns": float((err > 25).mean()),   # front-cut danger cases
    "early_by_gt100ns": float((err < -100).mean()),
}

# ---- containment: needs per-hit times/labels of the same test frames ------
X, sysi, y, evt = load_fold(HIT_DATA, "test")
uevt, evt_index = np.unique(evt, return_inverse=True)
# align per-frame t0_est to the hit-fold event order (windows fold dropped
# signal-less frames, so map by global event id; missing -> NaN)
est_map = dict(zip(te["gevt"].tolist(), t0_est.tolist()))
t0_est_full = np.array([est_map.get(g, np.nan) for g in uevt])
has_est = np.isfinite(t0_est_full[evt_index])
scan = containment_scan(X[has_est, 3], (y == 0)[has_est],
                        evt_index[has_est], np.nan_to_num(t0_est_full),
                        fronts=[25, 50, 100], backs=[150, 200, 300, 500, 1000])
metrics["containment"] = scan
(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

# ------------------------------------------------------------- plots ------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(np.clip(err, -500, 500), bins=100, color="#2c5aa0")
axes[0].axvline(0, color="k", lw=1)
axes[0].set_xlabel("t0_est - t0 [ns] (clipped; >0 = late = cuts burst front)")
axes[0].set_ylabel("frames"); axes[0].set_yscale("log")
axes[1].hist(np.clip(err, -100, 100), bins=80, color="#2c5aa0")
axes[1].axvline(0, color="k", lw=1)
axes[1].set_xlabel("zoom [ns]")
fig.suptitle(f"{run_name}: signed t0 error on test "
             f"(within 50 ns: {metrics['within_50ns']:.1%})")
fig.tight_layout(); fig.savefig(PLOTS / f"{run_name}_error_hist.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
backs = [150, 200, 300, 500, 1000]
for f, color in zip([25, 50, 100], ["#c0392b", "#1a7f37", "#2c5aa0"]):
    cont = [scan[f"front{f}_back{b}"]["signal_containment"] for b in backs]
    bkg = [scan[f"front{f}_back{b}"]["background_inside"] for b in backs]
    ax.plot(backs, cont, "o-", color=color, label=f"signal, front={f} ns")
    ax.plot(backs, bkg, "s--", color=color, alpha=0.5,
            label=f"bkg inside, front={f} ns")
ax.set_xlabel("back extent [ns]"); ax.set_ylabel("fraction of hits inside window")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title(f"{run_name}: window containment on test (phase-4 decision plot)")
fig.tight_layout(); fig.savefig(PLOTS / f"{run_name}_containment.png", dpi=130)
plt.close(fig)

# example frames: feature image + logits + true/predicted t0
fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
bins_t = np.arange(88) * BIN_NS + T_MIN
for ax, i in zip(axes, range(4)):
    ax.imshow(te["frames"][i], aspect="auto", origin="lower",
              extent=[T_MIN, 2100, -0.5, 7.5], cmap="Greys")
    ax2 = ax.twinx()
    ax2.plot(bins_t + BIN_NS / 2, logits[i], color="#2c5aa0", lw=1)
    ax2.set_yticks([])
    ax.axvline(te["t0"][i], color="#1a7f37", lw=1.5, label="true t0")
    ax.axvline(t0_est[i], color="#c0392b", lw=1.5, ls="--", label="predicted")
    ax.set_ylabel("feature #")
axes[0].legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time [ns]")
fig.suptitle(f"{run_name}: frame feature images, logits (blue), true vs "
             "predicted t0")
fig.tight_layout(); fig.savefig(PLOTS / f"{run_name}_examples.png", dpi=130)
plt.close(fig)

# ---- per-frame validation displays: true span (lines) vs estimated window
# (transparent band). Window drawn with front-50/back-300 ns margins — a
# visualization default, NOT a phase-4 decision. Hits re-read from parquet so
# the full 4-class prt_origin coloring matches the phase-2 displays.
import polars as pl  # noqa: E402
from trkbkg.temporal import window_display  # noqa: E402

EV_DIR = PLOTS / "events"
EV_DIR.mkdir(exist_ok=True)
FRONT_VIS, BACK_VIS = 50.0, 300.0
PARQUET = {0: str(CFG.parquet_dir / "18x275_v3/0022.trk_hits.parquet"),
           1: str(CFG.parquet_dir / "18x275_v3/0023.trk_hits.parquet")}
for k, g in enumerate(te["gevt"][:50].tolist()):
    file_idx, local_evt = divmod(g, 1_000_000)
    df = (pl.scan_parquet(PARQUET[file_idx])
          .filter(pl.col("evt") == local_evt)
          .select(["prt_origin", "trk_hit_pos_z", "trk_hit_time"]).collect())
    ts = df["trk_hit_time"].to_numpy()
    org = df["prt_origin"].to_numpy()
    t1_true = ts[org <= 2].max()
    window_display(
        ts, df["trk_hit_pos_z"].to_numpy(), org,
        t0_true=te["t0"][k], t1_true=t1_true,
        win_lo=t0_est[k] - FRONT_VIS, win_hi=t0_est[k] + BACK_VIS,
        out_path=EV_DIR / f"{run_name}_gevt{g}.png",
        title=(f"{run_name} test frame gevt={g}: est window "
               f"[{t0_est[k]-FRONT_VIS:.0f}, {t0_est[k]+BACK_VIS:.0f}] ns "
               f"(err {t0_est[k]-te['t0'][k]:+.0f} ns)"))
print(f"wrote 50 window validation displays to {EV_DIR}")

print(json.dumps({k: v for k, v in metrics.items() if k != "containment"},
                 indent=2))
print("containment (signal / bkg inside):")
for k, v in scan.items():
    print(f"  {k:>20}: {v['signal_containment']:.4f} / "
          f"{v['background_inside']:.4f}")

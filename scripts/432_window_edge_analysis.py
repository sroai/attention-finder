#!/usr/bin/env python
"""Where is the window ineffective — the START edge or the END edge?

Usage (from the repo root):
    python scripts/432_window_edge_analysis.py

For every TEST signal hit, dt = t - t0_est(wf_v2). The window keeps
dt in [-FRONT, +BACK] = [-50, +300] ns. Signal loss splits into:
    front loss: dt < -50   (hit before the window opens)
    back  loss: dt > +300  (hit after the window closes)
and each splits again by finder quality of its frame:
    found  frames: |t0_est - t0_true| <= 50 ns  -> loss is a genuine EDGE
                   effect (margin too tight)
    missed frames: |err| > 50 ns -> loss is a FINDER effect (burst is
                   simply elsewhere; both "edges" are meaningless there)

Outputs:
    plots/window/wf_v2_edge_signal_dt.png   where lost signal actually sits
    plots/window/wf_v2_edge_margins.png     loss vs each margin, other fixed
    numbers appended to runs/wf_v2/metrics.json under "edge_analysis"
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
from trkbkg.windows import BIN_NS, T_MIN, WindowFinderCNN

HIT_DATA = (CFG.dataset_dir / "18x275_v3")
WIN_DATA = (CFG.dataset_dir / "windows_v4")
RUNS = CFG.runs_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "window"

FRONT, BACK = 50.0, 300.0
FOUND_TOL = 50.0
WF_RUN = sys.argv[1] if len(sys.argv) > 1 else "wf_v2"

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(RUNS / WF_RUN / "best.pt", weights_only=False)
wf = WindowFinderCNN(width=ckpt["config"]["width"]).to(dev).eval()
wf.load_state_dict(ckpt["model_state"])

win = np.load(WIN_DATA / "test.npz")
with torch.no_grad():
    logits = wf(torch.from_numpy(win["frames"]).to(dev)).cpu().numpy()
t0_est = logits.argmax(axis=1) * BIN_NS + T_MIN
frame_err = t0_est - win["t0"]
est_map = dict(zip(win["gevt"].tolist(), t0_est.tolist()))
err_map = dict(zip(win["gevt"].tolist(), frame_err.tolist()))

X, sysi, y, evt = load_fold(HIT_DATA, "test")
uevt, evt_index = np.unique(evt, return_inverse=True)
t0_hit = np.array([est_map.get(g, np.nan) for g in uevt])[evt_index]
err_hit = np.array([err_map.get(g, np.nan) for g in uevt])[evt_index]
dt = X[:, 3] - t0_hit

sig = (y == 0) & np.isfinite(dt)
bkg = (y == 1) & np.isfinite(dt)
found = np.abs(err_hit) <= FOUND_TOL

def frac(m, base):
    return float(m.sum() / base.sum())

res = {"window_ns": [-FRONT, BACK], "found_tolerance_ns": FOUND_TOL,
       "frames_found_frac": float((np.abs(frame_err) <= FOUND_TOL).mean())}
for name, fm in [("all_frames", np.ones_like(found)),
                 ("found_frames", found), ("missed_frames", ~found)]:
    s = sig & fm
    res[name] = {
        "share_of_all_signal": frac(s, sig),
        "front_loss": frac(s & (dt < -FRONT), sig),
        "back_loss": frac(s & (dt > BACK), sig),
        "inside": frac(s & (dt >= -FRONT) & (dt <= BACK), sig),
    }

# ---- plot 1: where the signal sits relative to the WINDOW edges ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
bins_wide = np.linspace(-2000, 2000, 200)
bins_zoom = np.linspace(-200, 500, 141)
for ax, bins in [(axes[0], bins_wide), (axes[1], bins_zoom)]:
    ax.hist(dt[sig & found], bins=bins, color="#1a7f37", alpha=0.75,
            label="signal, finder FOUND frame")
    ax.hist(dt[sig & ~found], bins=bins, color="#c0392b", alpha=0.6,
            label="signal, finder MISSED frame")
    ax.axvspan(-FRONT, BACK, color="#2c5aa0", alpha=0.12, label="window")
    ax.axvline(-FRONT, color="#2c5aa0", lw=1.2)
    ax.axvline(BACK, color="#2c5aa0", lw=1.2, ls="--")
    ax.set_xlabel("t - t0_est [ns]")
    ax.set_yscale("log")
axes[0].set_ylabel("signal hits")
axes[0].legend(fontsize=8)
axes[1].set_title("zoom on the edges")
fig.suptitle("test signal hits relative to the ESTIMATED window "
             f"(front loss {100*res['all_frames']['front_loss']:.2f}%, "
             f"back loss {100*res['all_frames']['back_loss']:.2f}%)")
fig.tight_layout()
fig.savefig(PLOTS / f"{WF_RUN}_edge_signal_dt.png", dpi=130)
plt.close(fig)

# ---- plot 2: marginal scans — one edge varies, the other fixed ------------
fronts = np.arange(0, 205, 5)
backs = np.arange(100, 1010, 10)
def losses(front, back, m):
    inside = (dt >= -front) & (dt <= back)
    return (1 - frac(sig & m & inside, sig & m),
            frac(bkg & inside, bkg))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for m, lab, color in [(np.ones_like(found), "all frames", "#2c5aa0"),
                      (found, "found frames only", "#1a7f37")]:
    fl = [losses(f, BACK, m)[0] for f in fronts]
    bl = [losses(FRONT, b, m)[0] for b in backs]
    axes[0].plot(fronts, fl, "-", color=color, label=f"signal loss, {lab}")
    axes[1].plot(backs, bl, "-", color=color, label=f"signal loss, {lab}")
axes[0].plot(fronts, [losses(f, BACK, found)[1] for f in fronts], "--",
             color="#c0392b", label="bkg inside window")
axes[1].plot(backs, [losses(FRONT, b, found)[1] for b in backs], "--",
             color="#c0392b", label="bkg inside window")
axes[0].axvline(FRONT, color="k", lw=1, ls=":", label=f"current front={FRONT:.0f}")
axes[1].axvline(BACK, color="k", lw=1, ls=":", label=f"current back={BACK:.0f}")
for ax, xl in [(axes[0], "front margin [ns] (back fixed at 300)"),
               (axes[1], "back margin [ns] (front fixed at 50)")]:
    ax.set_xlabel(xl)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("fraction")
fig.suptitle("START vs END edge effectiveness: signal loss as each margin "
             "varies (log scale)")
fig.tight_layout()
fig.savefig(PLOTS / f"{WF_RUN}_edge_margins.png", dpi=130)
plt.close(fig)

mfile = RUNS / WF_RUN / "metrics.json"
m = json.loads(mfile.read_text())
m["edge_analysis"] = res
mfile.write_text(json.dumps(m, indent=2))
print(json.dumps(res, indent=2))
print(f"wrote plots to {PLOTS}")

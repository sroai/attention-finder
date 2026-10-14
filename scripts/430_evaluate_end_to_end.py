#!/usr/bin/env python
"""END-TO-END evaluation of the full phase-3+4 pipeline on the test fold.

Usage (from the repo root):
    python scripts/430_evaluate_end_to_end.py <win_run> \
        [--wf-run wf_v2] [--outside-run mlp_v1]

Pipeline being scored (per frame):
  1. window finder (wf_run) -> t0_est
  2. hits INSIDE [t0_est-50, t0_est+300] -> scores from <win_run> (dt model)
  3. hits OUTSIDE the window            -> scores from <outside-run> (mlp_v1)
     The window is a FEATURE, not a hard cut: the finder misses ~20% of
     frames, so hard-cutting outside hits would cap signal efficiency at
     ~88% — instead outside hits fall back to the phase-1 per-hit model.
  4. two thresholds (thr_in, thr_out), calibrated ON VAL by grid search:
     among (eff_in, eff_out) splits achieving >=99% TOTAL signal efficiency,
     pick the pair with maximum total background rejection.

Efficiency here is charged against ALL signal hits of the frame — window
losses included — so numbers are directly comparable to phase 1 (mlp_v1:
14.1% rejection @ 98.8% eff).

Outputs: runs/<win_run>/metrics_end_to_end.json,
         plots/window/<win_run>_e2e_*.png
"""

import argparse
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
from trkbkg.attention import WindowNet
from trkbkg.dataset import load_fold, load_meta
from trkbkg.models import build_model
from trkbkg.windows import BIN_NS, T_MIN, WindowFinderCNN

HIT_DATA = (CFG.dataset_dir / "18x275_v3")
WIN_DATA = (CFG.dataset_dir / "windows_v4")
WHITS = (CFG.dataset_dir / "window_hits_v3")
RUNS = CFG.runs_dir
parser = argparse.ArgumentParser()
parser.add_argument("win_run")
parser.add_argument("--wf-run", default="wf_v2")
parser.add_argument("--outside-run", default="mlp_v1")
parser.add_argument("--t0-dir", default=None,
                    help="dir with {fold}.npz refined t0 estimates (phase 5 "
                         "candidate verifier); overrides the finder argmax")
parser.add_argument("--tag", default=None,
                    help="suffix for output metrics/plot names (comparisons)")
args = parser.parse_args()
TAG = f"_{args.tag}" if args.tag else ""
# phase-5 comparison outputs go to their own folder (plots/phase5/)
PLOTS = (Path(__file__).resolve().parents[1] / "plots"
         / ("phase5" if (args.t0_dir or args.tag) else "window"))
PLOTS.mkdir(parents=True, exist_ok=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hit_meta = load_meta(HIT_DATA)
wmeta = json.loads((WHITS / "meta.json").read_text())
FRONT, BACK = wmeta["front_ns"], wmeta["back_ns"]

wf_ckpt = torch.load(RUNS / args.wf_run / "best.pt", weights_only=False)
wf = WindowFinderCNN(width=wf_ckpt["config"]["width"]).to(dev).eval()
wf.load_state_dict(wf_ckpt["model_state"])

win_ckpt = torch.load(RUNS / args.win_run / "best.pt", weights_only=False)
wnet = WindowNet(wmeta["norm"]["mean"], wmeta["norm"]["std"],
                 len(hit_meta["system_ids"]), d=win_ckpt["config"]["dim"],
                 n_blocks=win_ckpt["config"]["blocks"]).to(dev).eval()
wnet.load_state_dict(win_ckpt["model_state"])

out_ckpt = torch.load(RUNS / args.outside_run / "best.pt", weights_only=False)
onet = build_model(out_ckpt["config"]["model"], hit_meta,
                   **out_ckpt["config"]["model_kwargs"]).to(dev).eval()
onet.load_state_dict(out_ckpt["model_state"])


def score_fold_e2e(fold):
    """Returns per-hit (y, inside, score) for one fold, plus event ids."""
    if args.t0_dir:  # refined estimates from the phase-5 candidate verifier
        ref = np.load(Path(args.t0_dir) / f"{fold}.npz")
        est_map = dict(zip(ref["gevt"].tolist(), ref["t0_est"].tolist()))
    else:
        win = np.load(WIN_DATA / f"{fold}.npz")
        with torch.no_grad():
            logits = wf(torch.from_numpy(win["frames"]).to(dev)).cpu().numpy()
        t0_est = logits.argmax(axis=1) * BIN_NS + T_MIN
        est_map = dict(zip(win["gevt"].tolist(), t0_est.tolist()))

    X, sysi, y, evt = load_fold(HIT_DATA, fold)
    uevt, evt_index = np.unique(evt, return_inverse=True)
    t0_per_hit = np.array([est_map.get(g, np.nan) for g in uevt])[evt_index]
    dt = X[:, 3] - t0_per_hit
    inside = (dt >= -FRONT) & (dt <= BACK)

    scores = np.empty(len(y), np.float32)
    with torch.no_grad():
        # outside hits: phase-1 model on raw (x,y,z,t)
        out_idx = np.flatnonzero(~inside)
        for i in range(0, len(out_idx), 2_000_000):
            j = out_idx[i:i + 2_000_000]
            scores[j] = torch.sigmoid(onet(
                torch.from_numpy(X[j]).to(dev),
                torch.from_numpy(sysi[j]).to(dev))).cpu().numpy()
        # inside hits: window model on (x,y,z,dt), one event at a time
        Xin = X[inside].copy()
        Xin[:, 3] = dt[inside]
        sin = sysi[inside]
        ein = evt[inside]
        sc_in = np.empty(len(ein), np.float32)
        change = np.flatnonzero(np.diff(ein.astype(np.int64))) + 1
        for s, e in zip(np.concatenate([[0], change]),
                        np.concatenate([change, [len(ein)]])):
            sc_in[s:e] = torch.sigmoid(wnet(
                torch.from_numpy(Xin[s:e]).to(dev),
                torch.from_numpy(sin[s:e]).to(dev))).cpu().numpy()
        scores[inside] = sc_in
    return y, inside, scores, sysi


def calibrate(y, inside, scores, target_total):
    """Grid-search (thr_in, thr_out) on this fold: max rejection s.t.
    total signal efficiency >= target_total."""
    sig_in = np.sort(scores[inside & (y == 0)])
    sig_out = np.sort(scores[~inside & (y == 0)])
    n_sig = len(sig_in) + len(sig_out)
    containment = len(sig_in) / n_sig
    best = None
    for eff_in in np.linspace(max(0.9, target_total - 0.02), 0.9999, 40):
        need_out = (target_total - containment * eff_in) / (1 - containment)
        if need_out > 1:
            continue
        need_out = max(need_out, 0.0)
        k_in = min(int(np.ceil(eff_in * len(sig_in))), len(sig_in) - 1)
        thr_in = np.nextafter(sig_in[k_in], np.inf)
        k_out = min(int(np.ceil(need_out * len(sig_out))), len(sig_out) - 1)
        thr_out = np.nextafter(sig_out[k_out], np.inf)
        keep = np.where(inside, scores < thr_in, scores < thr_out)
        rej = 1.0 - keep[y == 1].mean()
        if best is None or rej > best[0]:
            best = (rej, float(thr_in), float(thr_out))
    return best[1], best[2]


def apply_thresholds(y, inside, scores, thr_in, thr_out):
    keep = np.where(inside, scores < thr_in, scores < thr_out)
    out = {
        "signal_efficiency": float(keep[y == 0].mean()),
        "background_rejection": float(1.0 - keep[y == 1].mean()),
        "kept_fraction_of_all_hits": float(keep.mean()),
    }
    if (y == 2).any():  # albedo: not in either metric, reported separately
        out["albedo_filtered_fraction"] = float(1.0 - keep[y == 2].mean())
    return out


print("[scoring val]", flush=True)
y_va, in_va, sc_va, _ = score_fold_e2e("val")
print("[scoring test]", flush=True)
y_te, in_te, sc_te, sys_te = score_fold_e2e("test")

thr_in, thr_out = calibrate(y_va, in_va, sc_va, 0.99)
result = {
    "win_run": args.win_run, "wf_run": args.wf_run,
    "outside_run": args.outside_run,
    "window_ns": [-FRONT, BACK],
    "thresholds": {"inside": thr_in, "outside": thr_out},
    "val": apply_thresholds(y_va, in_va, sc_va, thr_in, thr_out),
    "test": apply_thresholds(y_te, in_te, sc_te, thr_in, thr_out),
}

# end-to-end efficiency/rejection curve: recalibrate on val per target
curve = []
for tgt in [0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.998]:
    ti, to = calibrate(y_va, in_va, sc_va, tgt)
    m = apply_thresholds(y_te, in_te, sc_te, ti, to)
    curve.append({"target": tgt, **m})
result["test_curve"] = curve

(RUNS / args.win_run / f"metrics_end_to_end{TAG}.json").write_text(
    json.dumps(result, indent=2))
print(json.dumps({k: result[k] for k in ("thresholds", "val", "test")},
                 indent=2))

# ---- plots ---------------------------------------------------------------
mlp_metrics = json.loads((RUNS / args.outside_run / "metrics.json").read_text())
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot([c["signal_efficiency"] for c in curve],
        [c["background_rejection"] for c in curve], "o-", color="#2c5aa0",
        label=f"window pipeline ({args.win_run})")
ax.plot(mlp_metrics["test"]["signal_efficiency"],
        mlp_metrics["test"]["background_rejection"], "s", ms=9,
        color="#c0392b", label=f"phase-1 {args.outside_run} @99% target")
ax.axvline(0.99, color="#1a7f37", ls="--", lw=1, label="99% target")
ax.set_xlabel("END-TO-END signal efficiency (window losses included)")
ax.set_ylabel("background rejection")
ax.grid(alpha=0.3); ax.legend()
ax.set_title("phase 3+4 pipeline vs phase-1 per-hit model (test fold)")
fig.tight_layout()
fig.savefig(PLOTS / f"{args.win_run}_e2e_curve{TAG}.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
bins = np.linspace(0, 1, 120)
ax.hist(sc_te[in_te & (y_te == 1)], bins=bins, density=True, alpha=0.4,
        color="#c0392b", label="bkg (in window)")
ax.hist(sc_te[in_te & (y_te == 0)], bins=bins, density=True, histtype="step",
        lw=1.6, color="#1a7f37", label="signal (in window)")
ax.axvline(thr_in, color="k", ls="--", lw=1, label=f"thr_in={thr_in:.3f}")
ax.set_yscale("log"); ax.legend(fontsize=8)
ax.set_xlabel("in-window background score"); ax.set_ylabel("density")
ax.set_title(f"{args.win_run}: in-window score separation (test)")
fig.tight_layout()
fig.savefig(PLOTS / f"{args.win_run}_e2e_scores{TAG}.png", dpi=130)
plt.close(fig)
print(f"wrote plots to {PLOTS}")

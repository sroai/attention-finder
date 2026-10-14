#!/usr/bin/env python
"""Phase 4 data: hits inside the ESTIMATED window, with dt feature.

Usage (from the repo root):
    python scripts/140_build_window_hits.py <wf_run>

For every frame of every fold:
  1. t0_est from the phase-3 window finder <wf_run> (argmax bin).
  2. TRAIN ONLY: t0_est jittered by U(-25,+25) ns — stage 2 must tolerate
     stage-1 error, so it trains on imperfect windows (val/test stay
     deployment-faithful).
  3. keep hits with dt = t - t0_est in [-FRONT, +BACK]; save (x,y,z,dt),
     dense system idx, label, gevt.

Window build parameters (FRONT=50, BACK=300 ns) are generous on purpose —
phase-2 containment says 99.3% of signal lies within t0+200 ns and the
front must never be starved. meta.json records per-fold signal containment
of the built windows: END-TO-END efficiency = containment x in-window
efficiency, so evaluation (script 16) charges window losses honestly.

Output: <dataset_dir>/window_hits_v3/{fold}.npz + meta.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.dataset import load_fold
from trkbkg.windows import BIN_NS, T_MIN, WindowFinderCNN

HIT_DATA = (CFG.dataset_dir / "18x275_v3")
WIN_DATA = (CFG.dataset_dir / "windows_v4")
RUNS = CFG.runs_dir
OUT = (CFG.dataset_dir / "window_hits_v3")
OUT.mkdir(parents=True, exist_ok=True)

FRONT, BACK = 50.0, 300.0
JITTER = 25.0

wf_run = sys.argv[1] if len(sys.argv) > 1 else "wf_v2"
ckpt = torch.load(RUNS / wf_run / "best.pt", weights_only=False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = WindowFinderCNN(width=ckpt["config"]["width"]).to(dev).eval()
model.load_state_dict(ckpt["model_state"])

rng = np.random.default_rng(7)
meta = {"wf_run": wf_run, "front_ns": FRONT, "back_ns": BACK,
        "train_jitter_ns": JITTER, "folds": {}}

for fold in ("train", "val", "test"):
    win = np.load(WIN_DATA / f"{fold}.npz")
    with torch.no_grad():
        logits = model(torch.from_numpy(win["frames"]).to(dev)).cpu().numpy()
    t0_est = logits.argmax(axis=1) * BIN_NS + T_MIN
    if fold == "train":
        t0_est = t0_est + rng.uniform(-JITTER, JITTER, len(t0_est))
    est_map = dict(zip(win["gevt"].tolist(), t0_est.tolist()))

    X, sysi, y, evt = load_fold(HIT_DATA, fold)
    uevt, evt_index = np.unique(evt, return_inverse=True)
    t0_per_hit = np.array([est_map.get(g, np.nan) for g in uevt])[evt_index]
    dt = X[:, 3] - t0_per_hit
    inside = (dt >= -FRONT) & (dt <= BACK)  # NaN -> False (frame w/o estimate)

    Xw = X[inside].copy()
    Xw[:, 3] = dt[inside]  # 4th feature is now dt, not absolute t
    np.savez(OUT / f"{fold}.npz", X=Xw, sys=sysi[inside], y=y[inside],
             evt=evt[inside])

    n_sig_all = int((y == 0).sum())
    n_sig_in = int((y[inside] == 0).sum())
    meta["folds"][fold] = {
        "hits_inside": int(inside.sum()), "hits_total": len(y),
        "signal_total": n_sig_all, "signal_inside": n_sig_in,
        "signal_containment": n_sig_in / n_sig_all,
        "background_inside_frac": float(inside[y == 1].mean()),
    }
    print(fold, json.dumps(meta["folds"][fold]), flush=True)

# stage-2 normalization stats (x,y,z,dt,r) from the train window hits
tr = np.load(OUT / "train.npz")["X"]
r = np.hypot(tr[:, 0], tr[:, 1])
meta["norm"] = {"mean": [float(v) for v in tr.mean(0)] + [float(r.mean())],
                "std": [float(v) for v in tr.std(0)] + [float(r.std())]}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT}")

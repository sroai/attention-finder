#!/usr/bin/env python
"""Build the window-finder dataset: per-frame [F, 88] feature images + t0 targets.

Usage (from the repo root):
    uv run python scripts/130_build_window_features_1d.py

Reads the phase-1 event-split folds (so train/val/test frames are identical
across all models — no leakage) and writes, per fold:
    <dataset_dir>/windows_v1/<fold>.npz
        frames  float32 [n_frames, F, 88]   feature image
        target  int64   [n_frames]          bin containing true t0
        t0      float32 [n_frames]          true t0 [ns]
        gevt    uint32  [n_frames]          global event id
Frames without any signal hit (no t0 definable) are dropped and counted in
meta.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.dataset import load_fold, load_meta
from trkbkg.windows import N_FEATURES, frame_features, targets_from_t0

DATASET_DIR = (CFG.dataset_dir / "18x275_v3")
# windows_v1 is abandoned: a killed writer left a file there that hangs the
# /data mount on unlink (see notes/lesson-data-mount-poisoned-file.md)
OUT_DIR = (CFG.dataset_dir / "windows_v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

raw_ids = np.array(load_meta(DATASET_DIR)["system_ids"])
meta = {"bin_ns": 25.0, "n_features": N_FEATURES, "dropped_no_signal": {}}
for fold in ("train", "val", "test"):
    out_npz = OUT_DIR / f"{fold}.npz"
    if out_npz.exists():
        try:  # skip folds that already built successfully (resume support)
            np.load(out_npz)["frames"]
            print(f"[{fold}] exists, skipping")
            continue
        except Exception:
            out_npz.unlink()  # truncated file from an interrupted run
    X, sysi, y, evt = load_fold(DATASET_DIR, fold)
    uevt, evt_index = np.unique(evt, return_inverse=True)
    n_ev = len(uevt)

    # true t0 = earliest signal (y==0) hit time per frame
    t0 = np.full(n_ev, np.inf, dtype=np.float64)
    sig = y == 0
    np.minimum.at(t0, evt_index[sig], X[sig, 3].astype(np.float64))

    frames = frame_features(X, evt_index, n_ev, system_id=raw_ids[sysi])
    ok = np.isfinite(t0)
    meta["dropped_no_signal"][fold] = int((~ok).sum())

    np.savez(OUT_DIR / f"{fold}.npz",
             frames=frames[ok], target=targets_from_t0(t0[ok]),
             t0=t0[ok].astype(np.float32), gevt=uevt[ok])
    print(f"[{fold}] {ok.sum()} frames "
          f"({(~ok).sum()} dropped: no signal hits)", flush=True)

(OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT_DIR}")

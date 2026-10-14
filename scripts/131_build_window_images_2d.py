#!/usr/bin/env python
"""Phase 6: build the multi-preset 2D-image dataset for the window finder.

Usage (from the repo root):
    python scripts/131_build_window_images_2d.py

Scans every *_v3 parquet preset dir, renders each frame as a [2, 64, 88]
image (trkbkg/windows2d.py) and computes the true-t0 bin target. Split:
  * 18x275 frames keep their EXACT phase-3 fold assignment (gevts from
    dataset/windows_v4) so the frozen holdout stays comparable to wf_v2_r3.
  * every other preset is split 80/10/10 by frame (seed 42).
Per-fold npz also records a per-frame preset id so evaluation can report
PER-PROCESS metrics (cross-process generalization is a headline result).

Output: <dataset_dir>/images2d_v1/{fold}.npz + meta.json
Idempotent per preset: per-preset staging npz are cached, safe to re-run as
conversions finish.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.windows2d import N_BINS, frame_images
from trkbkg.windows import targets_from_t0

PARQUET_ROOT = CFG.parquet_dir
WIN_V4 = (CFG.dataset_dir / "windows_v4")
OUT = (CFG.dataset_dir / "images2d_v1")
STAGE = OUT / "staging"
STAGE.mkdir(parents=True, exist_ok=True)

PRESETS = ["18x275_v3", "10x130_v3", "background_dvmp-10x275-25-10_v3",
           "background-dis-nc-10x275_v3"]
COLS = ["evt", "prt_origin", "trk_hit_pos_x", "trk_hit_pos_y",
        "trk_hit_pos_z", "trk_hit_time"]


def build_preset(pdir: Path, preset_idx: int):
    """Images+targets for every frame of every file in one preset dir."""
    stage_f = STAGE / f"{pdir.name}.npz"
    if stage_f.exists():
        d = np.load(stage_f)
        return {k: d[k] for k in d.files}
    imgs, tgts, t0s, gevts = [], [], [], []
    files = sorted(pdir.glob("*.trk_hits.parquet"))
    for fi, pq in enumerate(files):
        df = pl.scan_parquet(pq).select(COLS).collect()
        evt = df["evt"].to_numpy()
        uevt, evt_index = np.unique(evt, return_inverse=True)
        X = np.stack([df[c].to_numpy() for c in COLS[2:]], axis=1
                     ).astype(np.float32)
        origin = df["prt_origin"].to_numpy()
        img = frame_images(X, evt_index, len(uevt))

        t0 = np.full(len(uevt), np.inf)
        sig = origin <= 2  # albedo (5) excluded automatically
        np.minimum.at(t0, evt_index[sig], X[sig, 3].astype(np.float64))
        ok = np.isfinite(t0)
        imgs.append(img[ok])
        tgts.append(targets_from_t0(t0[ok]))
        t0s.append(t0[ok].astype(np.float32))
        # global id: preset*1e8 + file*1e6 + evt (18x275 keeps its old ids)
        base = fi * 1_000_000 if preset_idx == 0 else \
            preset_idx * 100_000_000 + fi * 1_000_000
        gevts.append(base + uevt[ok].astype(np.int64))
        print(f"  [{pdir.name}] {pq.name}: {ok.sum()} frames", flush=True)
    out = {"img": np.concatenate(imgs), "target": np.concatenate(tgts),
           "t0": np.concatenate(t0s), "gevt": np.concatenate(gevts),
           "preset": np.full(sum(len(g) for g in gevts), preset_idx,
                             np.int8)}
    np.savez(stage_f, **out)
    return out


rng = np.random.default_rng(42)
folds = {f: {k: [] for k in ("img", "target", "t0", "gevt", "preset")}
         for f in ("train", "val", "test")}
# frozen 18x275 fold assignment from phase 3
assign18 = {}
for f in ("train", "val", "test"):
    for g in np.load(WIN_V4 / f"{f}.npz")["gevt"]:
        assign18[int(g)] = f

meta = {"presets": PRESETS, "frames": {}}
for pi, name in enumerate(PRESETS):
    pdir = PARQUET_ROOT / name
    if not pdir.exists():
        print(f"[skip] {name} not converted yet")
        continue
    d = build_preset(pdir, pi)
    if pi == 0:  # 18x275: frozen assignment
        fold_of = np.array([assign18.get(int(g), "train")
                            for g in d["gevt"]])
    else:
        r = rng.random(len(d["gevt"]))
        fold_of = np.where(r < 0.8, "train", np.where(r < 0.9, "val", "test"))
    for f in ("train", "val", "test"):
        m = fold_of == f
        for k in folds[f]:
            folds[f][k].append(d[k][m])
    meta["frames"][name] = int(len(d["gevt"]))

for f, d in folds.items():
    arrs = {k: np.concatenate(v) for k, v in d.items()}
    np.savez(OUT / f"{f}.npz", **arrs)
    meta.setdefault("fold_frames", {})[f] = int(len(arrs["gevt"]))
    print(f"[{f}] {len(arrs['gevt'])} frames", flush=True)
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print("wrote", OUT)

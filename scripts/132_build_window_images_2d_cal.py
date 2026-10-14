#!/usr/bin/env python
"""STEP 132 — phase 10: build the 4-channel (tracker + calorimeter) image dataset.

Same picture as step 131 with two calorimeter channels appended:

    channel 0: (z, t) tracker hit counts     } frame_images, phase-6 grid
    channel 1: (r, t) tracker hit counts     }
    channel 2: (z, t) calorimeter energy     } frame_images_cal, cal ranges,
    channel 3: (r, t) calorimeter energy     } 10 MeV units, log1p

The baseline of the phase-10 A/B trains on channels 0-1 of the SAME frames
(the trainer slices with --channels 2), so one dataset serves both sides.

Targets and frame selection are identical to 131: t0 = earliest TRACKER
origin<=2 hit (albedo 5 excluded), frames without signal tracker hits are
dropped. Calorimeter hits only add channels; a frame with no cal hits gets
zero cal channels.

Inputs : <parquet_dir>/<sample>_v3/<stem>.trk_hits.parquet      (step 110, v3 albedo rules)
         <parquet_dir>/<sample>_cal_v1/<stem>.cal_hits.parquet  (step 111)
Output : <dataset_dir>/images2d_p10/{train,val,test}.npz + meta.json
         (80/10/10 by frame, seed 42)

Usage:
    python scripts/132_build_window_images_2d_cal.py 10x130_p10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.windows2d import frame_images, frame_images_cal
from trkbkg.windows import targets_from_t0

TRK_COLS = ["evt", "prt_origin", "trk_hit_pos_x", "trk_hit_pos_y",
            "trk_hit_pos_z", "trk_hit_time"]
CAL_COLS = ["evt", "cal_hit_pos_x", "cal_hit_pos_y", "cal_hit_pos_z",
            "cal_hit_time", "cal_hit_energy"]

parser = argparse.ArgumentParser()
parser.add_argument("sample", help="e.g. 10x130_p10: reads <sample>_v3 and <sample>_cal_v1")
parser.add_argument("--out-name", default="images2d_p10")
args = parser.parse_args()

trk_dir = CFG.parquet_dir / f"{args.sample}_v3"
cal_dir = CFG.parquet_dir / f"{args.sample}_cal_v1"
OUT = CFG.dataset_dir / args.out_name
OUT.mkdir(parents=True, exist_ok=True)

imgs, tgts, t0s, gevts = [], [], [], []
trk_files = sorted(trk_dir.glob("*.trk_hits.parquet"))
if not trk_files:
    raise SystemExit(f"no trk parquet in {trk_dir} — run 110 first")
for fi, trk_pq in enumerate(trk_files):
    stem = trk_pq.name.split(".trk_hits")[0]
    cal_pq = cal_dir / f"{stem}.cal_hits.parquet"
    if not cal_pq.exists():
        raise SystemExit(f"{cal_pq} missing — run 111 first (frames must pair)")

    trk = pl.scan_parquet(trk_pq).select(TRK_COLS).collect()
    evt = trk["evt"].to_numpy()
    uevt, evt_index = np.unique(evt, return_inverse=True)
    X = np.stack([trk[c].to_numpy() for c in TRK_COLS[2:]], axis=1).astype(np.float32)
    origin = trk["prt_origin"].to_numpy()
    img_trk = frame_images(X, evt_index, len(uevt))

    cal = pl.scan_parquet(cal_pq).select(CAL_COLS).collect()
    cal_evt = cal["evt"].to_numpy()
    # map cal hits onto the tracker frame index; frames with tracker hits but
    # no cal hits keep zero cal channels, cal hits of dropped frames are cut
    pos = np.searchsorted(uevt, cal_evt)
    in_range = (pos < len(uevt))
    keep = in_range.copy()
    keep[in_range] &= (uevt[pos[in_range]] == cal_evt[in_range])
    Xc = np.stack([cal[c].to_numpy() for c in CAL_COLS[1:]], axis=1).astype(np.float32)[keep]
    img_cal = frame_images_cal(Xc, pos[keep], len(uevt))

    img = np.concatenate([img_trk, img_cal], axis=1)  # [n, 4, 64, 88]

    t0 = np.full(len(uevt), np.inf)
    sig = origin <= 2
    np.minimum.at(t0, evt_index[sig], X[sig, 3].astype(np.float64))
    ok = np.isfinite(t0)
    imgs.append(img[ok])
    tgts.append(targets_from_t0(t0[ok]))
    t0s.append(t0[ok].astype(np.float32))
    gevts.append(fi * 1_000_000 + uevt[ok].astype(np.int64))
    print(f"  {stem}: {ok.sum()} frames, cal hits used {int(keep.sum())}/{len(cal_evt)}",
          flush=True)

img = np.concatenate(imgs)
data = {"img": img, "target": np.concatenate(tgts), "t0": np.concatenate(t0s),
        "gevt": np.concatenate(gevts),
        "preset": np.zeros(len(img), np.int8)}

rng = np.random.default_rng(42)
r = rng.random(len(data["gevt"]))
fold_of = np.where(r < 0.8, "train", np.where(r < 0.9, "val", "test"))
meta = {"sample": args.sample, "channels":
        ["trk_zt_count", "trk_rt_count", "cal_zt_energy", "cal_rt_energy"],
        "fold_frames": {}}
for f in ("train", "val", "test"):
    m = fold_of == f
    np.savez(OUT / f"{f}.npz", **{k: v[m] for k, v in data.items()})
    meta["fold_frames"][f] = int(m.sum())
    print(f"[{f}] {int(m.sum())} frames")
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print("wrote", OUT)

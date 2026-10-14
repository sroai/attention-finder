#!/usr/bin/env python
"""STEP 133 — phase 12: build the two-class (k-Λ vs DIS/NC) image dataset.

Same 4-channel frame images as step 132, over several samples at once, with
per-frame class and kinematics labels appended:

    cls        0 = k-Λ (meson structure), 1 = DIS/NC
    sample_id  index into meta["samples"] (k-Λ, or the DIS minQ2 slice)
    q2, x      truth kinematics (InclusiveKinematicsTruth via step 112);
               NaN where the kinematics parquet has no row for the frame

Targets and frame selection are identical to 131/132 for BOTH classes:
t0 = earliest tracker origin<=2 hit, frames without signal tracker hits are
dropped. gevt = global_file_index * 1_000_000 + evt; meta["files"] maps the
file index back to sample + stem.

Inputs : <parquet_dir>/<sample>_{v3,cal_v1,kin_v1}/ for every sample
Output : <dataset_dir>/images2d_p12/{train,val,test}.npz + meta.json
         (80/10/10 by frame, seed 42, over the combined set)

Usage:
    python scripts/133_build_event_type_dataset.py \
        --klambda 10x100 --dis 10x100_minQ2_1 10x100_minQ2_10 10x100_minQ2_100
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
parser.add_argument("--klambda", nargs="+", default=["10x100"],
                    help="class-0 samples")
parser.add_argument("--dis", nargs="+",
                    default=["10x100_minQ2_1", "10x100_minQ2_10", "10x100_minQ2_100"],
                    help="class-1 samples")
parser.add_argument("--out-name", default="images2d_p12")
args = parser.parse_args()

samples = [(s, 0) for s in args.klambda] + [(s, 1) for s in args.dis]
OUT = CFG.dataset_dir / args.out_name
OUT.mkdir(parents=True, exist_ok=True)

arrays = {k: [] for k in ("img", "target", "t0", "gevt", "cls", "sample_id", "q2", "x")}
files_meta = []

for si, (sample, cls) in enumerate(samples):
    trk_dir = CFG.parquet_dir / f"{sample}_v3"
    cal_dir = CFG.parquet_dir / f"{sample}_cal_v1"
    kin_dir = CFG.parquet_dir / f"{sample}_kin_v1"
    trk_files = sorted(trk_dir.glob("*.trk_hits.parquet"))
    if not trk_files:
        raise SystemExit(f"no trk parquet in {trk_dir} — run 110 first")

    for trk_pq in trk_files:
        stem = trk_pq.name.split(".trk_hits")[0]
        gfi = len(files_meta)
        files_meta.append({"sample": sample, "stem": stem, "cls": cls})

        trk = pl.scan_parquet(trk_pq).select(TRK_COLS).collect()
        evt = trk["evt"].to_numpy()
        uevt, evt_index = np.unique(evt, return_inverse=True)
        X = np.stack([trk[c].to_numpy() for c in TRK_COLS[2:]], axis=1).astype(np.float32)
        origin = trk["prt_origin"].to_numpy()
        img_trk = frame_images(X, evt_index, len(uevt))

        cal = pl.scan_parquet(cal_dir / f"{stem}.cal_hits.parquet").select(CAL_COLS).collect()
        cal_evt = cal["evt"].to_numpy()
        pos = np.searchsorted(uevt, cal_evt)
        in_range = pos < len(uevt)
        keep = in_range.copy()
        keep[in_range] &= (uevt[pos[in_range]] == cal_evt[in_range])
        Xc = np.stack([cal[c].to_numpy() for c in CAL_COLS[1:]], axis=1).astype(np.float32)[keep]
        img = np.concatenate([img_trk, frame_images_cal(Xc, pos[keep], len(uevt))], axis=1)

        # per-frame truth kinematics; frames absent from the kin parquet get NaN
        q2 = np.full(len(uevt), np.nan, np.float32)
        xbj = np.full(len(uevt), np.nan, np.float32)
        kin_pq = kin_dir / f"{stem}.reco_dis.parquet"
        if kin_pq.exists():
            kin = pl.scan_parquet(kin_pq).select(["evt", "truth_q2", "truth_x"]).collect()
            kpos = np.searchsorted(uevt, kin["evt"].to_numpy())
            kin_ok = kpos < len(uevt)
            kin_ok[kin_ok] &= (uevt[kpos[kin_ok]] == kin["evt"].to_numpy()[kin_ok])
            q2[kpos[kin_ok]] = kin["truth_q2"].to_numpy()[kin_ok]
            xbj[kpos[kin_ok]] = kin["truth_x"].to_numpy()[kin_ok]

        t0 = np.full(len(uevt), np.inf)
        sig = origin <= 2
        np.minimum.at(t0, evt_index[sig], X[sig, 3].astype(np.float64))
        ok = np.isfinite(t0)
        arrays["img"].append(img[ok])
        arrays["target"].append(targets_from_t0(t0[ok]))
        arrays["t0"].append(t0[ok].astype(np.float32))
        arrays["gevt"].append(gfi * 1_000_000 + uevt[ok].astype(np.int64))
        arrays["cls"].append(np.full(ok.sum(), cls, np.int8))
        arrays["sample_id"].append(np.full(ok.sum(), si, np.int16))
        arrays["q2"].append(q2[ok])
        arrays["x"].append(xbj[ok])
        print(f"  {sample}/{stem}: {int(ok.sum())} frames "
              f"(q2 filled {int(np.isfinite(q2[ok]).sum())})", flush=True)

data = {k: np.concatenate(v) for k, v in arrays.items()}
n = len(data["img"])
rng = np.random.default_rng(42)
r = rng.random(n)
fold_of = np.where(r < 0.8, "train", np.where(r < 0.9, "val", "test"))

meta = {"samples": [{"name": s, "cls": c} for s, c in samples],
        "files": files_meta,
        "channels": ["trk_zt_count", "trk_rt_count", "cal_zt_energy", "cal_rt_energy"],
        "classes": {"0": "k_lambda", "1": "dis_nc"},
        "fold_frames": {}, "class_frames": {}}
for f in ("train", "val", "test"):
    m = fold_of == f
    np.savez(OUT / f"{f}.npz", **{k: v[m] for k, v in data.items()})
    meta["fold_frames"][f] = int(m.sum())
    meta["class_frames"][f] = [int((data["cls"][m] == c).sum()) for c in (0, 1)]
    print(f"[{f}] {int(m.sum())} frames "
          f"(k-Λ {meta['class_frames'][f][0]}, DIS {meta['class_frames'][f][1]})")
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print("wrote", OUT)

#!/usr/bin/env python
"""STEP 134 — phase 12: background-EXCHANGED two-class dataset.

Step 133's classifier turned out to fingerprint the production campaign:
pooling on a pure-background band still separated k-Λ from DIS at 99.8%,
because the two classes carry different background cocktails (26.06 msf vs
26.07 ePIC). This builder makes background USELESS for classification by
construction: every output frame is

    signal-only image of frame i   (hits with prt_origin <= 2)
  + background-only image of a frame j drawn RANDOMLY from EITHER class
    (hits with prt_origin > 2; j sampled within the same fold — no
     train/test background leakage)

Images add in count/energy space (expm1 -> add -> log1p; the channels are
log1p-binned). t0, target, cls, q2, x come from the SIGNAL frame; the
window target stays valid because the exchanged background has no signal
comb (its donor's signal hits are excluded; its albedo tail remains, which
any real frame also has). Fold split is identical to 133 (seed 42 on the
same gevt), so results compare frame-for-frame.

Output : <dataset_dir>/images2d_p12bgx/{train,val,test}.npz + meta.json
Usage  : python scripts/134_build_event_type_dataset_bgx.py \
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
CAL_COLS = ["evt", "prt_origin", "cal_hit_pos_x", "cal_hit_pos_y",
            "cal_hit_pos_z", "cal_hit_time", "cal_hit_energy"]

parser = argparse.ArgumentParser()
parser.add_argument("--klambda", nargs="+", default=["10x100"])
parser.add_argument("--dis", nargs="+",
                    default=["10x100_minQ2_1", "10x100_minQ2_10", "10x100_minQ2_100"])
parser.add_argument("--out-name", default="images2d_p12bgx")
args = parser.parse_args()

samples = [(s, 0) for s in args.klambda] + [(s, 1) for s in args.dis]
OUT = CFG.dataset_dir / args.out_name
OUT.mkdir(parents=True, exist_ok=True)


def split_images(trk, cal, uevt, evt_index, mask_trk, mask_cal):
    """4-channel image for the selected hits, in count/energy space.

    Stored float16 to fit two full copies (signal and background) of the
    80k-frame set in the 32 GB job — counts and 10 MeV energy units stay
    far below float16 range; composition upcasts to float32.
    """
    X = np.stack([trk[c].to_numpy() for c in TRK_COLS[2:]], axis=1).astype(np.float32)
    img_trk = frame_images(X[mask_trk], evt_index[mask_trk], len(uevt))
    Xc = np.stack([cal[c].to_numpy() for c in CAL_COLS[2:]], axis=1).astype(np.float32)
    img_cal = frame_images_cal(Xc[mask_cal[0]], mask_cal[1], len(uevt))
    return np.expm1(np.concatenate([img_trk, img_cal], axis=1)).astype(np.float16)


arrays = {k: [] for k in ("sig", "bkg", "target", "t0", "gevt", "cls",
                          "sample_id", "q2", "x")}
files_meta = []

for si, (sample, cls) in enumerate(samples):
    trk_files = sorted((CFG.parquet_dir / f"{sample}_v3").glob("*.trk_hits.parquet"))
    if not trk_files:
        raise SystemExit(f"no trk parquet for {sample}")
    for trk_pq in trk_files:
        stem = trk_pq.name.split(".trk_hits")[0]
        gfi = len(files_meta)
        files_meta.append({"sample": sample, "stem": stem, "cls": cls})

        trk = pl.scan_parquet(trk_pq).select(TRK_COLS).collect()
        evt = trk["evt"].to_numpy()
        uevt, evt_index = np.unique(evt, return_inverse=True)
        origin = trk["prt_origin"].to_numpy()
        cal = pl.scan_parquet(CFG.parquet_dir / f"{sample}_cal_v1" /
                              f"{stem}.cal_hits.parquet").select(CAL_COLS).collect()
        cal_evt = cal["evt"].to_numpy()
        pos = np.searchsorted(uevt, cal_evt)
        ok_cal = pos < len(uevt)
        ok_cal[ok_cal] &= (uevt[pos[ok_cal]] == cal_evt[ok_cal])
        cal_origin = cal["prt_origin"].to_numpy()

        sig = split_images(trk, cal, uevt, evt_index, origin <= 2,
                           ((cal_origin <= 2) & ok_cal, pos[(cal_origin <= 2) & ok_cal]))
        bkg = split_images(trk, cal, uevt, evt_index, origin > 2,
                           ((cal_origin > 2) & ok_cal, pos[(cal_origin > 2) & ok_cal]))

        q2 = np.full(len(uevt), np.nan, np.float32)
        xbj = np.full(len(uevt), np.nan, np.float32)
        kin_pq = CFG.parquet_dir / f"{sample}_kin_v1" / f"{stem}.reco_dis.parquet"
        if kin_pq.exists():
            kin = pl.scan_parquet(kin_pq).select(["evt", "truth_q2", "truth_x"]).collect()
            kpos = np.searchsorted(uevt, kin["evt"].to_numpy())
            k_ok = kpos < len(uevt)
            k_ok[k_ok] &= (uevt[kpos[k_ok]] == kin["evt"].to_numpy()[k_ok])
            q2[kpos[k_ok]] = kin["truth_q2"].to_numpy()[k_ok]
            xbj[kpos[k_ok]] = kin["truth_x"].to_numpy()[k_ok]

        X = np.stack([trk[c].to_numpy() for c in TRK_COLS[2:]], axis=1)
        t0 = np.full(len(uevt), np.inf)
        s = origin <= 2
        np.minimum.at(t0, evt_index[s], X[s, 3].astype(np.float64))
        ok = np.isfinite(t0)
        arrays["sig"].append(sig[ok])
        arrays["bkg"].append(bkg[ok])
        arrays["target"].append(targets_from_t0(t0[ok]))
        arrays["t0"].append(t0[ok].astype(np.float32))
        arrays["gevt"].append(gfi * 1_000_000 + uevt[ok].astype(np.int64))
        arrays["cls"].append(np.full(ok.sum(), cls, np.int8))
        arrays["sample_id"].append(np.full(ok.sum(), si, np.int16))
        arrays["q2"].append(q2[ok])
        arrays["x"].append(xbj[ok])
        print(f"  {sample}/{stem}: {int(ok.sum())} frames", flush=True)

data = {k: np.concatenate(v) for k, v in arrays.items()}
for k in arrays:
    arrays[k] = None
n = len(data["cls"])
rng = np.random.default_rng(42)
fold_r = rng.random(n)                       # same split as 133
fold_of = np.where(fold_r < 0.8, "train", np.where(fold_r < 0.9, "val", "test"))

mix_rng = np.random.default_rng(1234)
meta = {"samples": [{"name": s, "cls": c} for s, c in samples],
        "files": files_meta, "background_exchange": True,
        "channels": ["trk_zt_count", "trk_rt_count", "cal_zt_energy", "cal_rt_energy"],
        "classes": {"0": "k_lambda", "1": "dis_nc"},
        "fold_frames": {}, "class_frames": {}, "bkg_donor_class_frac": {}}
for f in ("train", "val", "test"):
    m = np.flatnonzero(fold_of == f)
    donors = m[mix_rng.integers(0, len(m), len(m))]   # background from own fold, any class
    img = np.log1p(data["sig"][m].astype(np.float32)
                   + data["bkg"][donors].astype(np.float32))
    np.savez(OUT / f"{f}.npz", img=img,
             **{k: data[k][m] for k in ("target", "t0", "gevt", "cls", "sample_id", "q2", "x")},
             bkg_donor_cls=data["cls"][donors])
    meta["fold_frames"][f] = int(len(m))
    meta["class_frames"][f] = [int((data["cls"][m] == c).sum()) for c in (0, 1)]
    meta["bkg_donor_class_frac"][f] = float((data["cls"][donors] == 1).mean())
    print(f"[{f}] {len(m)} frames (k-Λ {meta['class_frames'][f][0]}, "
          f"DIS {meta['class_frames'][f][1]}; DIS-background fraction "
          f"{meta['bkg_donor_class_frac'][f]:.3f})")
    del img
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print("wrote", OUT)

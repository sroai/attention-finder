#!/usr/bin/env python
"""Phase 8: build the ALL-PRESET in-window hit dataset for retraining watt.

Why (expectation recorded per protocol): watt_r2 was trained on 18x275 only
and transfers imperfectly (in-window sig/bkg AUC 0.99 home vs 0.94 DVMP /
0.95 DIS-NC — scratchpad watt_transfer_check.py). Its sigmoid score is the
strongest input of BOTH candidate-selector paths (11 hand features + the
verifier's 6th per-hit feature), and DVMP/DIS are exactly the presets where
recall lags (84/86% vs 93/96%). Expected: +0.02-0.05 in-window AUC there,
worth +1-3 pts of preset recall through the selector.

Two window types per frame:
  * TRUE window: [t0-FRONT, t0+BACK] around the true t0 (train: jittered
    U(-25,+25) ns — same deployment-tolerance trick as phase 4);
  * TRAIN ONLY, one WRONG window per frame: a random 25 ns bin start
    >= 100 ns away from t0 — hard negatives so the scores stay meaningful
    in wrong candidate windows (phase-5 lesson: watt was never trained
    there and its scores were unreliable, which capped the verifier).
Labels stay per-hit (signal/background/albedo) in both window types.

Frames/folds/t0 come from images2d_v1 (frozen 18x275 test fold preserved);
system indexing is the same 22-id dense map as everywhere else.

Usage (from the repo root):
    python scripts/141_build_window_hits_all_presets.py

Output: dataset/window_hits_all_v1/{fold}.npz  (X[x,y,z,dt], sys, y, evt =
sequential window id, preset per window in wpreset) + meta.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.dataset import load_meta
from trkbkg.windows import BIN_NS, N_BINS, T_MIN

IMG_DATA = (CFG.dataset_dir / "images2d_v1")
PARQUET = CFG.parquet_dir
HIT_DATA = (CFG.dataset_dir / "18x275_v3")
OUT = (CFG.dataset_dir / "window_hits_all_v1")
OUT.mkdir(parents=True, exist_ok=True)

FRONT, BACK = 50.0, 300.0
JITTER = 25.0
WRONG_MIN_DIST = 100.0

rng = np.random.default_rng(11)
sys_ids = load_meta(HIT_DATA)["system_ids"]
sys_index = {s: i for i, s in enumerate(sys_ids)}
meta = json.loads((IMG_DATA / "meta.json").read_text())
PRESETS = meta["presets"]
COLS = ["evt", "prt_origin", "trk_hit_system_id", "trk_hit_pos_x",
        "trk_hit_pos_y", "trk_hit_pos_z", "trk_hit_time"]
Y_OF_ORIGIN = {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 2.0}


def decode_gevt(gevt, preset):
    rem = gevt - (int(preset) * 100_000_000 if preset > 0 else 0)
    return rem // 1_000_000, rem % 1_000_000


fold_stats = {}
for fold in ("train", "val", "test"):
    ffile = OUT / f"{fold}.npz"
    if ffile.exists():
        print(f"[{fold}] exists, skipping", flush=True)
        continue
    d = np.load(IMG_DATA / f"{fold}.npz")
    n = len(d["gevt"])
    t0_true = d["t0"]
    fidx = np.zeros(n, np.int64)
    evt_of = np.zeros(n, np.int64)
    for i in range(n):
        fidx[i], evt_of[i] = decode_gevt(int(d["gevt"][i]),
                                         int(d["preset"][i]))

    Xc, sc, yc, wc, pc = [], [], [], [], []   # chunks
    wid = 0
    sig_total = sig_inside = 0
    t_start = time.time()
    for pi, preset in enumerate(PRESETS):
        files = sorted((PARQUET / preset).glob("*.trk_hits.parquet"))
        pm = d["preset"] == pi
        for fi in np.unique(fidx[pm]):
            rows = np.where(pm & (fidx == fi))[0]
            df = pl.scan_parquet(files[fi]).select(COLS).collect()
            evt = df["evt"].to_numpy()
            origin = df["prt_origin"].to_numpy()
            sysid = df["trk_hit_system_id"].to_numpy()
            X = np.stack([df[c].to_numpy() for c in COLS[3:]], 1
                         ).astype(np.float32)
            si_all = np.array([sys_index.get(int(s), 0) for s in sysid],
                              np.int64)
            y_all = np.array([Y_OF_ORIGIN.get(int(o), 1.0) for o in origin],
                             np.float32)
            order = np.argsort(evt, kind="stable")
            evt_s = evt[order]
            starts = np.searchsorted(evt_s, np.arange(evt_s.max() + 2))
            for i in rows:
                e = evt_of[i]
                sl = order[starts[e]:starts[e + 1]]
                t = X[sl, 3]
                y_e = y_all[sl]
                sig_total += int((y_e == 0).sum())

                # window list: (t0w, is_true)
                t0w = float(t0_true[i])
                if fold == "train":
                    t0w += float(rng.uniform(-JITTER, JITTER))
                wins = [(t0w, True)]
                if fold == "train":
                    while True:
                        b = int(rng.integers(0, N_BINS))
                        t0c = b * BIN_NS + T_MIN
                        if abs(t0c - t0_true[i]) >= WRONG_MIN_DIST:
                            break
                    wins.append((float(t0c), False))

                for t0x, is_true in wins:
                    m = (t >= t0x - FRONT) & (t <= t0x + BACK)
                    nh = int(m.sum())
                    if is_true:
                        sig_inside += int((y_e[m] == 0).sum())
                    if nh == 0:
                        continue
                    Xw = X[sl][m].copy()
                    Xw[:, 3] = t[m] - t0x
                    Xc.append(Xw)
                    sc.append(si_all[sl][m])
                    yc.append(y_e[m])
                    wc.append(np.full(nh, wid, np.uint32))
                    pc.append(np.full(nh, pi, np.uint8))
                    wid += 1
        print(f"[{fold}] {preset} done ({time.time() - t_start:.0f}s)",
              flush=True)

    Xf = np.concatenate(Xc)
    np.savez(ffile, X=Xf, sys=np.concatenate(sc), y=np.concatenate(yc),
             evt=np.concatenate(wc), wpreset=np.concatenate(pc))
    fold_stats[fold] = {
        "windows": int(wid), "hits": int(len(Xf)),
        "signal_total": int(sig_total), "signal_inside": int(sig_inside),
        "signal_containment": float(sig_inside / max(sig_total, 1)),
    }
    print(f"[{fold}] {wid} windows, {len(Xf)} hits, containment "
          f"{fold_stats[fold]['signal_containment']:.4f}", flush=True)

if fold_stats:
    tr = np.load(OUT / "train.npz")["X"]
    r = np.hypot(tr[:, 0], tr[:, 1])
    Xn = np.concatenate([tr, r[:, None]], 1)
    meta_out = {
        "source": "images2d_v1 folds, true t0 (+wrong-window hard negatives "
                  "in train)", "front_ns": FRONT, "back_ns": BACK,
        "train_jitter_ns": JITTER, "wrong_min_dist_ns": WRONG_MIN_DIST,
        "presets": PRESETS, "folds": fold_stats,
        "norm": {"mean": Xn.mean(0).tolist(), "std": Xn.std(0).tolist()},
    }
    (OUT / "meta.json").write_text(json.dumps(meta_out, indent=2))
print(f"wrote {OUT}")

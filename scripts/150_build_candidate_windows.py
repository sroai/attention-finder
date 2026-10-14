#!/usr/bin/env python
"""Phase 6b: build the candidate-window HIT dataset for the learned verifier.

Same parquet pass as script 31, but instead of 11 pooled features it saves
the window's raw hits: x, y, z, dt, r, watt_r2 score, system index —
subsampled to MAX_HITS (=512) per window with a fixed seed; the true hit
count is kept as a window scalar. Candidates: finder2d_v1 top-k per frame,
folds/identity from images2d_v1 (frozen 18x275 test preserved).

Output (dataset/cand_windows_v1/): {fold}.npz with
    hits   [total_hits, 7]  (x, y, z, dt, r, watt, sys_idx)
    offs   [n_windows+1]    window i owns hits[offs[i]:offs[i+1]]
    n_hits, label, rank, logit, cand_t0  [n_windows]
    frame, preset, gevt, t0_true         (frame = window -> frame index)
    meta.json: normalization (train fold), k, finder, sizes

Usage (from the repo root):
    python scripts/150_build_candidate_windows.py \
        [--finder finder2d_v1] [--k 10]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.attention import WindowNet
from trkbkg.dataset import load_meta
from trkbkg.reranker import BACK, FRONT
from trkbkg.verifier import MAX_HITS
from trkbkg.windows import BIN_NS, T_MIN
from trkbkg.windows2d import WindowFinder2D

IMG_DATA = (CFG.dataset_dir / "images2d_v1")
PARQUET = CFG.parquet_dir
HIT_DATA = (CFG.dataset_dir / "18x275_v3")
WHITS = (CFG.dataset_dir / "window_hits_v3")
RUNS = CFG.runs_dir
parser = argparse.ArgumentParser()
parser.add_argument("--finder", default="finder2d_v1")
parser.add_argument("--win-run", default="watt_r2")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--out-name", default="cand_windows_v1")
args = parser.parse_args()

OUT = CFG.dataset_dir / args.out_name
OUT.mkdir(parents=True, exist_ok=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(7)

fck = torch.load(CFG.runs_local_dir / args.finder /
                 "best.pt", weights_only=False)
finder = WindowFinder2D(width=fck["config"]["width"],
                        n_blocks=fck["config"].get("blocks", 3),
                        batchnorm=fck["config"].get("batchnorm", False)
                        ).to(dev).eval()
finder.load_state_dict(fck["model_state"])

wmeta = json.loads((WHITS / "meta.json").read_text())
sys_ids = load_meta(HIT_DATA)["system_ids"]
sys_index = {s: i for i, s in enumerate(sys_ids)}
wck = torch.load(RUNS / args.win_run / "best.pt", weights_only=False)
wnet = WindowNet(wmeta["norm"]["mean"], wmeta["norm"]["std"], len(sys_ids),
                 d=wck["config"]["dim"], n_blocks=wck["config"]["blocks"]
                 ).to(dev).eval()
wnet.load_state_dict(wck["model_state"])

meta = json.loads((IMG_DATA / "meta.json").read_text())
PRESETS = meta["presets"]
COLS = ["evt", "trk_hit_system_id", "trk_hit_pos_x", "trk_hit_pos_y",
        "trk_hit_pos_z", "trk_hit_time"]


def decode_gevt(gevt, preset):
    rem = gevt - (int(preset) * 100_000_000 if preset > 0 else 0)
    return rem // 1_000_000, rem % 1_000_000


for fold in ("train", "val", "test"):
    ffile = OUT / f"{fold}.npz"
    if ffile.exists():
        print(f"[{fold}] exists, skipping", flush=True)
        continue
    d = np.load(IMG_DATA / f"{fold}.npz")
    n = len(d["gevt"])
    lg = []
    with torch.no_grad():
        for i in range(0, n, 512):
            lg.append(finder(torch.from_numpy(d["img"][i:i + 512]).to(dev))
                      .cpu().numpy())
    logits = np.concatenate(lg)
    top = np.argsort(-logits, axis=1)[:, :args.k]
    cand_t0 = (top * BIN_NS + T_MIN).astype(np.float32)

    fidx = np.zeros(n, np.int64)
    evt_of = np.zeros(n, np.int64)
    for i in range(n):
        fidx[i], evt_of[i] = decode_gevt(int(d["gevt"][i]),
                                         int(d["preset"][i]))

    hits_chunks = []           # [n_i, 7] per window, in window order
    n_hits = np.zeros(n * args.k, np.int32)
    t_start = time.time()
    done = 0
    for pi, preset in enumerate(PRESETS):
        files = sorted((PARQUET / preset).glob("*.trk_hits.parquet"))
        pm = d["preset"] == pi
        for fi in np.unique(fidx[pm]):
            rows = np.where(pm & (fidx == fi))[0]
            df = pl.scan_parquet(files[fi]).select(COLS).collect()
            evt = df["evt"].to_numpy()
            sysid = df["trk_hit_system_id"].to_numpy()
            X = np.stack([df[c].to_numpy() for c in COLS[2:]], 1
                         ).astype(np.float32)
            si_all = np.array([sys_index.get(int(s), 0) for s in sysid],
                              np.float32)
            order = np.argsort(evt, kind="stable")
            evt_s = evt[order]
            starts = np.searchsorted(evt_s, np.arange(evt_s.max() + 2))
            with torch.no_grad():
                for i in rows:
                    e = evt_of[i]
                    sl = order[starts[e]:starts[e + 1]]
                    t = X[sl, 3]
                    for c_i in range(args.k):
                        wi = i * args.k + c_i
                        t0c = cand_t0[i, c_i]
                        m = (t >= t0c - FRONT) & (t <= t0c + BACK)
                        nh = int(m.sum())
                        n_hits[wi] = nh
                        if nh == 0:
                            hits_chunks.append(np.zeros((0, 7), np.float32))
                            continue
                        Xw = X[sl][m].copy()
                        Xw[:, 3] = t[m] - t0c
                        siw = si_all[sl][m]
                        if nh > MAX_HITS:
                            pick = rng.choice(nh, MAX_HITS, replace=False)
                            Xw, siw = Xw[pick], siw[pick]
                        sc = torch.sigmoid(wnet(
                            torch.from_numpy(Xw).to(dev),
                            torch.from_numpy(siw.astype(np.int64)).to(dev))
                        ).cpu().numpy()
                        r = np.hypot(Xw[:, 0], Xw[:, 1])
                        hits_chunks.append(np.column_stack(
                            [Xw[:, 0], Xw[:, 1], Xw[:, 2], Xw[:, 3],
                             r, sc, siw]).astype(np.float32))
            done += len(rows)
            print(f"[{fold}] {preset} file {fi}: {done}/{n} frames "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    # windows appear in (frame-row, candidate) order only within each file
    # pass; rebuild global order: we appended in iteration order of rows,
    # which is NOT 0..n-1 — fix by tracking explicit window ids.
    # (rows come from np.where per file; concatenate order == append order)
    hits = np.concatenate(hits_chunks) if hits_chunks else \
        np.zeros((0, 7), np.float32)
    lens = np.array([len(h) for h in hits_chunks], np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)])
    # explicit window ids in append order:
    wid = []
    for pi, preset in enumerate(PRESETS):
        pm = d["preset"] == pi
        for fi in np.unique(fidx[pm]):
            for i in np.where(pm & (fidx == fi))[0]:
                wid.extend(i * args.k + c_i for c_i in range(args.k))
    wid = np.array(wid, np.int64)

    rows_i = np.repeat(np.arange(n), args.k)          # frame of window w
    label = (np.abs(cand_t0 + BIN_NS / 2 -
                    d["t0"][:, None]) <= 50).reshape(-1)
    cand_logit = logits[rows_i, top.reshape(-1)].astype(np.float32)
    np.savez(ffile,
             hits=hits, offs=offs, window_id=wid,
             n_hits=n_hits[wid],
             label=label[wid],
             rank=np.tile(np.arange(args.k), n)[wid].astype(np.int8),
             logit=cand_logit[wid],
             cand_t0=cand_t0.reshape(-1)[wid],
             frame=rows_i[wid],
             preset=d["preset"][rows_i[wid]],
             gevt=d["gevt"][rows_i[wid]],
             t0_true=d["t0"][rows_i[wid]])
    print(f"[{fold}] saved {len(wid)} windows, {len(hits)} hits", flush=True)

# normalization from train hits
tr = np.load(OUT / "train.npz")
norm = {"mean": tr["hits"][:, :6].mean(0).tolist(),
        "std": (tr["hits"][:, :6].std(0) + 1e-6).tolist(),
        "order": ["x", "y", "z", "dt", "r", "watt"]}
(OUT / "meta.json").write_text(json.dumps(
    {"finder": args.finder, "win_run": args.win_run, "k": args.k,
     "max_hits": MAX_HITS, "front": FRONT, "back": BACK,
     "system_ids": sys_ids, "norm": norm}, indent=2))
print("meta written", flush=True)

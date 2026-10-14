#!/usr/bin/env python
"""Phase 6b: hit-based reranker on ALL presets, hits read from parquet.

The 18x275-only reranker (script 23) lifts recall 88.3 -> 92.3% on that
holdout, and its selector is data-limited (7k training frames). This script
removes the limit: candidate features are built for EVERY preset's frames
(~22k train), reading hits directly from the *_v3 parquet (no per-hit fold
npz needed), scored by watt_r2 — verified to transfer (in-window AUC
0.94-0.99 per preset; reports/08_finder2d.md).

Frame identity and fold assignment come from images2d_v1 (gevt encodes
preset/file/evt exactly as script 25 wrote it), so the frozen 18x275 test
fold stays comparable with every previous number.

Usage (from the repo root):
    python scripts/350_rerank_candidates_hand_features.py \
        [--finder finder2d_v1] [--k 10] [--out-name t0_rerank_all_v1]

Outputs:
    dataset/<out-name>/cand_cache/{fold}.npz     features (selector is free)
    dataset/<out-name>/{fold}.npz                refined t0 estimates
    <runs_local_dir>/<out-name>/metrics.json           per-preset recall table
    plots/phase6/<out-name>_recall.png
    reports/phase6_<out-name>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.attention import WindowNet
from trkbkg.dataset import load_meta
from trkbkg.reranker import (BACK, FEAT_NAMES, FRONT, apply_selector,
                             cand_features, train_selector, within_frame_std)
from trkbkg.windows import BIN_NS, T_MIN
from trkbkg.windows2d import WindowFinder2D

IMG_DATA = (CFG.dataset_dir / "images2d_v1")
PARQUET = CFG.parquet_dir
HIT_DATA = (CFG.dataset_dir / "18x275_v3")
WHITS = (CFG.dataset_dir / "window_hits_v3")
RUNS = CFG.runs_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase6"
PLOTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--finder", default="finder2d_v1")
parser.add_argument("--win-run", default="watt_r2")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--out-name", default="t0_rerank_all_v1")
args = parser.parse_args()
OUT = CFG.dataset_dir / args.out_name
CACHE = OUT / "cand_cache"
CACHE.mkdir(parents=True, exist_ok=True)
run_dir = CFG.runs_local_dir / args.out_name
run_dir.mkdir(parents=True, exist_ok=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- models ----------------------------------------------------------------
fdir = CFG.runs_local_dir / args.finder
fck = torch.load(fdir / "best.pt", weights_only=False)
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
    """(file idx, evt) from script 25's encoding."""
    rem = gevt - (int(preset) * 100_000_000 if preset > 0 else 0)
    return rem // 1_000_000, rem % 1_000_000


def build_fold(fold):
    d = np.load(IMG_DATA / f"{fold}.npz")
    n = len(d["gevt"])
    lg = []
    with torch.no_grad():
        for i in range(0, n, 512):
            lg.append(finder(torch.from_numpy(d["img"][i:i + 512]).to(dev))
                      .cpu().numpy())
    logits = np.concatenate(lg)
    top = np.argsort(-logits, axis=1)[:, :args.k]
    cand_t0 = top * BIN_NS + T_MIN

    F = np.zeros((n, args.k, len(FEAT_NAMES)), np.float32)
    ok = np.zeros((n, args.k), bool)

    fidx, evt_of = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for i in range(n):
        fidx[i], evt_of[i] = decode_gevt(int(d["gevt"][i]), int(d["preset"][i]))

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
            si_all = np.array([sys_index.get(int(s), 0) for s in sysid])
            order = np.argsort(evt, kind="stable")
            evt_s = evt[order]
            starts = np.searchsorted(evt_s, np.arange(evt_s.max() + 2))
            with torch.no_grad():
                for i in rows:
                    e = evt_of[i]
                    sl = order[starts[e]:starts[e + 1]]
                    t = X[sl, 3]
                    for c_i in range(args.k):
                        t0c = cand_t0[i, c_i]
                        m = (t >= t0c - FRONT) & (t <= t0c + BACK)
                        if m.sum() < 5:
                            continue
                        Xw = X[sl][m].copy()
                        Xw[:, 3] = t[m] - t0c
                        sc = torch.sigmoid(wnet(
                            torch.from_numpy(Xw).to(dev),
                            torch.from_numpy(si_all[sl][m]).to(dev))
                        ).cpu().numpy()
                        F[i, c_i] = cand_features(
                            logits[i, top[i, c_i]], c_i, Xw, sc,
                            sysid[sl][m])
                        ok[i, c_i] = True
            done += len(rows)
            print(f"[{fold}] {preset} file {fi}: {done}/{n} frames "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    lab = np.abs(cand_t0 + BIN_NS / 2 - d["t0"][:, None]) <= 50
    return {"gevt": d["gevt"], "preset": d["preset"], "t0_true": d["t0"],
            "cand_t0": cand_t0, "F": F, "ok": ok, "label": lab}


cands = {}
for fold in ("train", "val", "test"):
    cfile = CACHE / f"{fold}.npz"
    if cfile.exists():
        cands[fold] = dict(np.load(cfile))
        print(f"[{fold}] candidate cache loaded", flush=True)
    else:
        cands[fold] = build_fold(fold)
        np.savez(cfile, **cands[fold])
        print(f"[{fold}] candidates built + cached", flush=True)

# ---- selector ---------------------------------------------------------------
# within-frame standardized features + margin-gated override; margin chosen
# on val (global scalar, deployable). See reranker.py for the why of both.
tr = cands["train"]
net, norm = train_selector(within_frame_std(tr["F"], tr["ok"]), tr["ok"],
                           tr["label"], hidden=32)


def recall50(t0_est, t0_true, m=None):
    hit = np.abs(t0_est + BIN_NS / 2 - t0_true) <= 50
    return float(hit.mean() if m is None else hit[m].mean())


va = cands["val"]
Fva = within_frame_std(va["F"], va["ok"])
margin, best_v = 0.0, -1.0
for m_try in (0.0, 0.25, 0.5, 0.75, 1.0):
    t0_m, _ = apply_selector(net, norm, Fva, va["ok"], va["cand_t0"],
                             margin=m_try)
    v = recall50(t0_m, va["t0_true"])
    if v > best_v:
        best_v, margin = v, m_try
print(f"val-chosen margin: {margin}", flush=True)

metrics = {"finder": args.finder, "win_run": args.win_run, "k": args.k,
           "selector": "bce32 within-frame-std", "margin": margin}
for fold in ("train", "val", "test"):
    c = cands[fold]
    t0_ref, sel = apply_selector(net, norm,
                                 within_frame_std(c["F"], c["ok"]),
                                 c["ok"], c["cand_t0"], margin=margin)
    rec = {"frames": int(len(sel)),
           "recall50_argmax": recall50(c["cand_t0"][:, 0], c["t0_true"]),
           "recall50_reranked": recall50(t0_ref, c["t0_true"]),
           "topk_ceiling": float(c["label"].any(1).mean()),
           "per_preset": {}}
    for pi, name in enumerate(PRESETS):
        m = c["preset"] == pi
        if m.any():
            rec["per_preset"][name] = {
                "argmax": recall50(c["cand_t0"][:, 0], c["t0_true"], m),
                "reranked": recall50(t0_ref, c["t0_true"], m),
                "ceiling": float(c["label"][m].any(1).mean())}
    metrics[fold] = rec
    print(fold, json.dumps(rec), flush=True)
    np.savez(OUT / f"{fold}.npz", gevt=c["gevt"], preset=c["preset"],
             t0_est=t0_ref, t0_true=c["t0_true"])

(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
stats_f = Path(__file__).resolve().parents[1] / "reports" / \
    f"phase6_{args.out_name}.json"
stats_f.write_text(json.dumps(metrics, indent=2))

# ---- plot -------------------------------------------------------------------
te = metrics["test"]
names = ["overall"] + list(te["per_preset"])
arg = [te["recall50_argmax"]] + [v["argmax"]
                                 for v in te["per_preset"].values()]
rer = [te["recall50_reranked"]] + [v["reranked"]
                                   for v in te["per_preset"].values()]
ceil = [te["topk_ceiling"]] + [v["ceiling"]
                               for v in te["per_preset"].values()]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9.5, 4.5))
ax.bar(x - 0.2, arg, 0.4, label="finder argmax", color="#999999")
ax.bar(x + 0.2, rer, 0.4, label="+ hit-based reranker", color="#1a7f37")
ax.plot(x, ceil, "kv", ms=6, label=f"top-{args.k} ceiling")
for i in range(len(names)):
    ax.text(i + 0.2, rer[i] + 0.012, f"{rer[i]:.1%}", ha="center", fontsize=8)
ax.set_xticks(x, [n.replace("background", "bg") for n in names],
              fontsize=8, rotation=10)
ax.set_ylabel("test recall within 50 ns"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="y")
ax.set_title(f"all-preset hit-based reranker ({args.finder} top-{args.k}, "
             f"{args.win_run} scores)")
fig.tight_layout()
fig.savefig(PLOTS / f"{args.out_name}_recall.png", dpi=130)
print(f"wrote {PLOTS / f'{args.out_name}_recall.png'}")

#!/usr/bin/env python
"""STEP 428 — full autopsy of the frames a 2D finder run misses (>50 ns).

For every missed test frame: which file/event it is, where the model looked
instead, how deep the true bin ranks in the logits, and a two-part figure —

  top: the model's view (image channels, true and predicted time columns
       marked, per-column tracker-count and calorimeter-energy profiles);
  bottom: the physicist's view (z vs t of the frame's tracker and cal hits
       from the parquet, colored by prt_origin, t0/pred lines).

Usage:
    python scripts/428_autopsy_missed_frames.py p10_cal \
        --dataset images2d_p10 --sample 10x130

Outputs: plots/phase10/<run>_miss_<gevt>.png + a printed table.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.windows import BIN_NS, T_MIN
from trkbkg.windows2d import N_BINS, WindowFinder2D

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--dataset", default="images2d_p10")
parser.add_argument("--sample", default="10x130",
                    help="parquet sample: <sample>_v3 (trk) + <sample>_cal_v1")
parser.add_argument("--channels", type=int, default=0)
args = parser.parse_args()

DATA = CFG.dataset_dir / args.dataset
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase10"
PLOTS.mkdir(parents=True, exist_ok=True)

run_dir = CFG.runs_local_dir / args.run_name
if not (run_dir / "best.pt").exists():
    run_dir = CFG.runs_dir / args.run_name
ckpt = torch.load(run_dir / "best.pt", weights_only=False, map_location="cpu")
cfg = ckpt["config"]
n_ch = args.channels or cfg.get("channels", 2)
model = WindowFinder2D(n_channels=n_ch, width=cfg["width"],
                       n_blocks=cfg.get("blocks", 3),
                       batchnorm=cfg.get("batchnorm", False)).eval()
model.load_state_dict(ckpt["model_state"])

te = np.load(DATA / "test.npz")
X = torch.from_numpy(te["img"][:, :n_ch])
with torch.no_grad():
    logits = np.concatenate([model(X[i:i + 1024]).numpy()
                             for i in range(0, len(X), 1024)])
pred = logits.argmax(1)
pred_t = pred * BIN_NS + T_MIN + BIN_NS / 2
err = pred_t - te["t0"]
missed = np.flatnonzero(np.abs(err) > 50)
print(f"{args.run_name}: {len(missed)} missed of {len(err)} test frames")

trk_files = sorted((CFG.parquet_dir / f"{args.sample}_v3").glob("*.trk_hits.parquet"))

for mi in missed:
    gevt = int(te["gevt"][mi])
    fi, evt = gevt // 1_000_000, gevt % 1_000_000
    stem = trk_files[fi].name.split(".trk_hits")[0]
    order = np.argsort(-logits[mi])
    true_bin = int(te["target"][mi])
    rank = int(np.flatnonzero(order == true_bin)[0])
    print(f"  gevt {gevt}: {stem} evt {evt} | t0 {te['t0'][mi]:.0f} ns, "
          f"predicted {pred_t[mi]:.0f} ns (err {err[mi]:+.0f}) | "
          f"true-bin rank {rank} of {N_BINS}, "
          f"logit gap {logits[mi, pred[mi]] - logits[mi, true_bin]:.2f}")

    img = te["img"][mi]                      # [4, 64, 88]
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.7, 1.4])
    titles = ["trk (z,t) counts", "trk (r,t) counts",
              "cal (z,t) energy", "cal (r,t) energy"]
    for ci in range(min(4, img.shape[0])):
        ax = fig.add_subplot(gs[ci // 2, ci % 2])
        ax.imshow(img[ci], aspect="auto", origin="lower", cmap="viridis")
        ax.axvline(true_bin, color="#1a7f37", lw=1.5)
        ax.axvline(int(pred[mi]), color="#d62728", lw=1.5, ls="--")
        ax.set_title(titles[ci], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # per-column profiles + logits
    axp = fig.add_subplot(gs[1, :]) if img.shape[0] <= 2 else fig.add_subplot(gs[2, 0])
    axp.plot(np.expm1(img[0]).sum(0), label="trk hits / column", lw=1.2)
    if img.shape[0] > 2:
        axp.plot(np.expm1(img[2]).sum(0), label="cal energy [10 MeV] / column", lw=1.2)
    axp.axvline(true_bin, color="#1a7f37", lw=1.5, label="true t0 bin")
    axp.axvline(int(pred[mi]), color="#d62728", lw=1.5, ls="--", label="predicted")
    axp.set_xlabel("time bin (25 ns)"); axp.legend(fontsize=8); axp.grid(alpha=0.3)

    # hit-level view from parquet
    axz = fig.add_subplot(gs[2, 1])
    trk = (pl.scan_parquet(trk_files[fi])
           .filter(pl.col("evt") == evt)
           .select(["trk_hit_pos_z", "trk_hit_time", "prt_origin"]).collect())
    cal_pq = CFG.parquet_dir / f"{args.sample}_cal_v1" / f"{stem}.cal_hits.parquet"
    cal = (pl.scan_parquet(cal_pq).filter(pl.col("evt") == evt)
           .select(["cal_hit_pos_z", "cal_hit_time", "cal_hit_energy",
                    "prt_origin"]).collect())
    for df, zc, tc, mk, lab in ((trk, "trk_hit_pos_z", "trk_hit_time", ".", "trk"),
                                (cal, "cal_hit_pos_z", "cal_hit_time", "o", "cal")):
        o = df["prt_origin"].to_numpy()
        colors = np.where(o <= 2, "#1a7f37", "#d62728")
        axz.scatter(df[tc].to_numpy(), df[zc].to_numpy(), s=6 if mk == "." else 14,
                    c=colors, marker=mk, linewidths=0, alpha=0.6, label=lab)
    axz.axvline(te["t0"][mi], color="#1a7f37", lw=1.5)
    axz.axvline(pred_t[mi], color="#d62728", lw=1.5, ls="--")
    axz.set_xlabel("time [ns]"); axz.set_ylabel("z [mm]")
    axz.legend(fontsize=8, loc="upper right"); axz.grid(alpha=0.3)
    axz.set_title("hits: green=signal-descended, red=background", fontsize=9)

    n_sig_trk = int((trk["prt_origin"].to_numpy() <= 2).sum())
    sig_cal = cal.filter(pl.col("prt_origin") <= 2)
    fig.suptitle(f"{args.run_name} miss: {stem} evt {evt} — t0 {te['t0'][mi]:.0f} ns, "
                 f"pred {pred_t[mi]:.0f} ns, true-bin rank {rank}; "
                 f"signal: {n_sig_trk} trk hits, {len(sig_cal)} cal hits "
                 f"({float(sig_cal['cal_hit_energy'].sum()):.2f} GeV)", fontsize=10)
    fig.tight_layout()
    out = PLOTS / f"{args.run_name}_miss_{gevt}.png"
    fig.savefig(out, dpi=110)
    print(f"  wrote {out}")

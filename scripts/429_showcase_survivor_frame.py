#!/usr/bin/env python
"""STEP 429 — showcase a frame the finder gets RIGHT despite a heavy burst.

Companion to 428: the autopsy shows the one frame lost to a p-beam-gas
burst; this finds the strongest bursts the model SURVIVED. Among the
correctly-predicted test frames it ranks the out-of-window calorimeter
energy column (the competing comb) against the signal column, prints the
top candidates, and renders the same two-part figure as 428 for the pick,
plus a JSON with the numbers for the slides.

Usage:
    python scripts/429_showcase_survivor_frame.py p10_cal \
        --dataset images2d_p10 --sample 10x130 [--gevt N]

Outputs: plots/phase10/<run>_survivor_<gevt>.png/.json + a printed table.
"""

import argparse
import json
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
parser.add_argument("--gevt", type=int, default=None,
                    help="plot this frame instead of the top-ranked one")
parser.add_argument("--top", type=int, default=10)
parser.add_argument("--min-sig", type=float, default=5.0,
                    help="min signal-column cal energy [GeV] — keeps the "
                         "showcase analogous to the missed frame")
args = parser.parse_args()

DATA = CFG.dataset_dir / args.dataset
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase10"
PLOTS.mkdir(parents=True, exist_ok=True)

run_dir = CFG.runs_local_dir / args.run_name
if not (run_dir / "best.pt").exists():
    run_dir = CFG.runs_dir / args.run_name
ckpt = torch.load(run_dir / "best.pt", weights_only=False, map_location="cpu")
cfg = ckpt["config"]
n_ch = cfg.get("channels", 2)
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
correct = np.abs(err) <= 25

# competing burst: strongest calorimeter energy column >4 bins from truth
cal_prof = np.expm1(te["img"][:, 2]).sum(1) * 0.01        # [N, 88] in GeV
cols = np.arange(N_BINS)[None, :]
away = np.abs(cols - te["target"][:, None]) > 4
burst_e = np.where(away, cal_prof, 0).max(1)
burst_bin = np.where(away, cal_prof, 0).argmax(1)
sig_e = cal_prof[np.arange(len(cal_prof)), te["target"].astype(int)]
ratio = burst_e / np.maximum(sig_e, 1e-6)

cand = np.flatnonzero(correct & (sig_e > args.min_sig) & (burst_e > sig_e))
cand = cand[np.argsort(-ratio[cand])][:args.top]
print(f"{args.run_name}: top surviving-burst frames "
      f"({correct.sum()} correct of {len(err)}):")
print("  gevt      burst[GeV] @bin   signal[GeV] @bin   ratio  margin")
for i in cand:
    order = np.argsort(-logits[i])
    margin = logits[i, pred[i]] - logits[i, order[1]]
    print(f"  {int(te['gevt'][i]):8d}  {burst_e[i]:8.1f} {burst_bin[i]:5d}  "
          f"{sig_e[i]:10.1f} {int(te['target'][i]):5d}  {ratio[i]:5.2f}  "
          f"{margin:5.2f}")

mi = int(np.flatnonzero(te["gevt"] == args.gevt)[0]) if args.gevt else int(cand[0])
gevt = int(te["gevt"][mi])
fi, evt = gevt // 1_000_000, gevt % 1_000_000
trk_files = sorted((CFG.parquet_dir / f"{args.sample}_v3").glob("*.trk_hits.parquet"))
stem = trk_files[fi].name.split(".trk_hits")[0]
true_bin = int(te["target"][mi])
order = np.argsort(-logits[mi])
runner = int(order[1])
margin = float(logits[mi, pred[mi]] - logits[mi, runner])

img = te["img"][mi]                      # [4, 64, 88]
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.7, 1.4])
titles = ["trk (z,t) counts", "trk (r,t) counts",
          "cal (z,t) energy", "cal (r,t) energy"]
for ci in range(min(4, img.shape[0])):
    ax = fig.add_subplot(gs[ci // 2, ci % 2])
    ax.imshow(img[ci], aspect="auto", origin="lower", cmap="viridis")
    ax.axvline(true_bin, color="#1a7f37", lw=1.5)
    ax.axvline(int(burst_bin[mi]), color="#b06400", lw=1.5, ls=":")
    ax.set_title(titles[ci], fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

axp = fig.add_subplot(gs[2, 0])
axp.plot(np.expm1(img[0]).sum(0), label="trk hits / column", lw=1.2)
if img.shape[0] > 2:
    axp.plot(np.expm1(img[2]).sum(0), label="cal energy [10 MeV] / column", lw=1.2)
axp.axvline(true_bin, color="#1a7f37", lw=1.5, label="true t0 = predicted")
axp.axvline(int(burst_bin[mi]), color="#b06400", lw=1.5, ls=":", label="burst")
axp.set_xlabel("time bin (25 ns)"); axp.legend(fontsize=8); axp.grid(alpha=0.3)

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
axz.set_xlabel("time [ns]"); axz.set_ylabel("z [mm]")
axz.legend(fontsize=8, loc="upper right"); axz.grid(alpha=0.3)
axz.set_title("hits: green=signal-descended, red=background", fontsize=9)

n_sig_trk = int((trk["prt_origin"].to_numpy() <= 2).sum())
sig_cal = cal.filter(pl.col("prt_origin") <= 2)
fig.suptitle(f"{args.run_name} survivor: {stem} evt {evt} — t0 {te['t0'][mi]:.0f} ns "
             f"found correctly over a {burst_e[mi]:.0f} GeV burst at "
             f"{int(burst_bin[mi]) * BIN_NS + T_MIN + BIN_NS / 2:.0f} ns; "
             f"signal: {n_sig_trk} trk hits, {len(sig_cal)} cal hits "
             f"({float(sig_cal['cal_hit_energy'].sum()):.2f} GeV); "
             f"logit margin {margin:.2f}", fontsize=10)
fig.tight_layout()
out = PLOTS / f"{args.run_name}_survivor_{gevt}.png"
fig.savefig(out, dpi=110)
(out.with_suffix(".json")).write_text(json.dumps({
    "gevt": gevt, "stem": stem, "evt": evt, "t0_ns": float(te["t0"][mi]),
    "pred_ns": float(pred_t[mi]), "true_bin": true_bin,
    "burst_bin": int(burst_bin[mi]),
    "burst_t_ns": float(int(burst_bin[mi]) * BIN_NS + T_MIN + BIN_NS / 2),
    "burst_gev": float(burst_e[mi]), "signal_col_gev": float(sig_e[mi]),
    "ratio": float(ratio[mi]), "logit_margin": margin,
    "runner_up_bin": runner,
    "n_sig_trk_hits": n_sig_trk, "n_sig_cal_hits": len(sig_cal),
    "sig_cal_gev": float(sig_cal["cal_hit_energy"].sum()),
}, indent=2))
print(f"wrote {out} and .json")

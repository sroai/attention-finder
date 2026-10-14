#!/usr/bin/env python
"""Phase 8 (slides): event displays that SHOW candidate-window selection.

One page per frame: hit z vs time (signal red / background gray / albedo
purple), the finder's top-k candidate windows drawn as vertical bands
(numbers = finder rank), the selector's pick in green, the true t0 dashed.
Frames are chosen to tell the story:
  * rescued  — finder argmax wrong, selector picks the right candidate;
  * argmax-ok — the easy case (argmax right, selector agrees);
  * still-missed — no candidate within 50 ns or selector picks wrong.

Usage (from the repo root):
    python scripts/720_candidate_event_displays.py \
        [--t0 t0_final_v1] [--finder finder2d_v1] [--k 10] \
        [--n-rescued 6] [--n-ok 2] [--n-missed 4] [--cpu]

Outputs: plots/phase8/events/{category}_{preset}_{gevt}.png
         + plots/phase8/events/index.json
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
from trkbkg.windows import BIN_NS, N_BINS, T_MIN
from trkbkg.windows2d import WindowFinder2D

IMG_DATA = (CFG.dataset_dir / "images2d_v1")
PARQUET = CFG.parquet_dir
DSETS = CFG.dataset_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase8" / "events"
PLOTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--t0", default="t0_final_v1")
parser.add_argument("--finder", default="finder2d_v1")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--n-rescued", type=int, default=6)
parser.add_argument("--n-ok", type=int, default=2)
parser.add_argument("--n-missed", type=int, default=4)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
dev = torch.device("cpu" if args.cpu else
                   ("cuda" if torch.cuda.is_available() else "cpu"))

meta = json.loads((IMG_DATA / "meta.json").read_text())
PRESETS = meta["presets"]
SHORT = {"18x275_v3": "18x275", "10x130_v3": "10x130",
         "background_dvmp-10x275-25-10_v3": "DVMP",
         "background-dis-nc-10x275_v3": "DIS-NC"}

fck = torch.load(CFG.runs_local_dir / args.finder /
                 "best.pt", weights_only=False)
cfg = fck["config"]
finder = WindowFinder2D(width=cfg["width"], n_blocks=cfg.get("blocks", 3),
                        batchnorm=cfg.get("batchnorm", False)).to(dev).eval()
finder.load_state_dict(fck["model_state"])

te = np.load(IMG_DATA / "test.npz")
est = np.load(DSETS / args.t0 / "test.npz")
assert np.array_equal(te["gevt"], est["gevt"]), "frame order mismatch"
t0_true, t0_est, preset = est["t0_true"], est["t0_est"], est["preset"]

lg = []
with torch.no_grad():
    for i in range(0, len(te["img"]), 512):
        lg.append(finder(torch.from_numpy(te["img"][i:i + 512]).to(dev))
                  .cpu().numpy())
logits = np.concatenate(lg)
top = np.argsort(-logits, axis=1)[:, :args.k]
cand_t0 = top * BIN_NS + T_MIN                       # candidate window starts
argmax_t = logits.argmax(1) * BIN_NS + T_MIN + BIN_NS / 2

argmax_ok = np.abs(argmax_t - t0_true) <= 50
sel_ok = np.abs(t0_est + BIN_NS / 2 - t0_true) <= 50  # est is a bin START
rng = np.random.default_rng(3)


def pick(mask, n):
    """Round-robin across presets, then top up from whatever is left."""
    idx = np.where(mask)[0]
    queues = [list(idx[preset[idx] == pi]) for pi in range(len(PRESETS))]
    out = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


cases = [("rescued", pick(~argmax_ok & sel_ok, args.n_rescued)),
         ("argmax_ok", pick(argmax_ok & sel_ok, args.n_ok)),
         ("still_missed", pick(~sel_ok, args.n_missed))]


def decode_gevt(gevt, pi):
    rem = gevt - (int(pi) * 100_000_000 if pi > 0 else 0)
    return rem // 1_000_000, rem % 1_000_000


COLS = ["evt", "prt_origin", "trk_hit_pos_z", "trk_hit_time"]
index = []
for cat, rows in cases:
    for i in rows:
        pi = int(preset[i])
        fi, e = decode_gevt(int(te["gevt"][i]), pi)
        files = sorted((PARQUET / PRESETS[pi]).glob("*.trk_hits.parquet"))
        df = (pl.scan_parquet(files[fi]).select(COLS)
              .filter(pl.col("evt") == e).collect())
        z = df["trk_hit_pos_z"].to_numpy()
        t = df["trk_hit_time"].to_numpy()
        origin = df["prt_origin"].to_numpy()
        is_sig = (origin == 1) | (origin == 2)
        is_alb = origin == 5

        fig, ax = plt.subplots(figsize=(11, 4.6))
        bkg = ~is_sig & ~is_alb
        ax.scatter(t[bkg], z[bkg] / 1000, s=2, c="#b0b0b0", alpha=0.35,
                   label=f"background ({bkg.sum()})", rasterized=True)
        ax.scatter(t[is_alb], z[is_alb] / 1000, s=8, c="#8e44ad",
                   label=f"albedo ({is_alb.sum()})")
        ax.scatter(t[is_sig], z[is_sig] / 1000, s=8, c="#c0392b",
                   label=f"signal ({is_sig.sum()})")

        chosen_start = float(t0_est[i])
        for rank in range(args.k - 1, -1, -1):     # draw rank 0 last (on top)
            c0 = float(cand_t0[i, rank])
            chosen = abs(c0 - chosen_start) < 1e-6
            color = "#27ae60" if chosen else "#5b8ec4"
            alpha = 0.30 if chosen else 0.10
            ax.axvspan(c0, c0 + BIN_NS, color=color, alpha=alpha, lw=0)
            y_lab = 0.975 - 0.035 * (rank % 3)     # stagger overlapping labels
            ax.annotate(str(rank), (c0 + BIN_NS / 2, y_lab), size=7,
                        xycoords=("data", "axes fraction"), ha="center",
                        color="#1e8449" if chosen else "#31597e",
                        weight="bold" if chosen else "normal")
        ax.axvline(t0_true[i], color="k", ls="--", lw=1.2,
                   label="true interaction t0")
        # legend proxies for the bands
        ax.axvspan(np.nan, np.nan, color="#5b8ec4", alpha=0.25,
                   label=f"finder top-{args.k} candidates")
        ax.axvspan(np.nan, np.nan, color="#27ae60", alpha=0.45,
                   label="selector's pick")

        err = chosen_start + BIN_NS / 2 - t0_true[i]
        aerr = argmax_t[i] - t0_true[i]
        ax.set_title(f"{SHORT[PRESETS[pi]]} file {fi} frame {e} — "
                     f"argmax err {aerr:+.0f} ns, selector err {err:+.0f} ns"
                     f"  [{cat.replace('_', ' ')}]", size=10)
        ax.set_xlabel("hit time in frame [ns]")
        ax.set_ylabel("hit z [m]")
        ax.set_xlim(-BIN_NS, 2000 + BIN_NS)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        fig.tight_layout()
        name = f"{cat}_{SHORT[PRESETS[pi]]}_{int(te['gevt'][i])}.png"
        fig.savefig(PLOTS / name, dpi=140)
        plt.close(fig)
        index.append({"file": name, "category": cat,
                      "preset": SHORT[PRESETS[pi]], "gevt": int(te["gevt"][i]),
                      "argmax_err_ns": float(aerr),
                      "selector_err_ns": float(err)})
        print(f"wrote {name}", flush=True)

(PLOTS / "index.json").write_text(json.dumps(index, indent=2))
print(f"{len(index)} displays in {PLOTS}")

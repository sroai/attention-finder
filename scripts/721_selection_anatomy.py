#!/usr/bin/env python
"""Phase 8 (slides): anatomy of ONE frame's window selection, top to bottom.

Four stacked panels, shared time axis — the whole two-stage story in one
figure:
  1. the (z,t) occupancy image the 2D finder sees (channel 1);
  2. the (r,t) occupancy image (channel 2);
  3. the finder's per-column logits, top-k candidates dotted, argmax marked;
  4. the learned verifier's score for each candidate window (bars), the
     selected candidate green, the true t0 dashed across all panels.

Default frames: one argmax-trap rescue and one easy frame, chosen from the
script-53 index if present (else first rescued/argmax-ok in fold order).

Usage (from the repo root):
    python scripts/721_selection_anatomy.py \
        [--t0 t0_final_v1] [--wver wver_v1] [--cwin cand_windows_v1] \
        [--finder finder2d_v1] [--gevts 229 ...] [--cpu]

Outputs: plots/phase8/selection_anatomy_<preset>_<gevt>.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.windows import BIN_NS, N_BINS, T_MIN
from trkbkg.windows2d import WindowFinder2D

IMG_DATA = (CFG.dataset_dir / "images2d_v1")
DSETS = CFG.dataset_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase8"
PLOTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--t0", default="t0_final_v1")
parser.add_argument("--wver", default="wver_v1")
parser.add_argument("--cwin", default="cand_windows_v1")
parser.add_argument("--finder", default="finder2d_v1")
parser.add_argument("--gevts", type=int, nargs="*", default=None,
                    help="explicit test-fold gevt ids; default: one rescued"
                         " + one easy frame")
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
dev = torch.device("cpu" if args.cpu else
                   ("cuda" if torch.cuda.is_available() else "cpu"))

meta = json.loads((IMG_DATA / "meta.json").read_text())
PRESETS = meta["presets"]
SHORT = {"18x275_v3": "18x275", "10x130_v3": "10x130",
         "background_dvmp-10x275-25-10_v3": "DVMP",
         "background-dis-nc-10x275_v3": "DIS-NC"}

te = np.load(IMG_DATA / "test.npz")
est = np.load(DSETS / args.t0 / "test.npz")
assert np.array_equal(te["gevt"], est["gevt"])

cw = np.load(DSETS / args.cwin / "test.npz")
K = json.loads((DSETS / args.cwin / "meta.json").read_text())["k"]
wver = np.load(DSETS / args.cwin / f"{args.wver}_scores_test.npy")
# candidate arrays are in window order = frame * K + rank
cand_t0 = cw["cand_t0"].reshape(-1, K)
wver_grid = wver.reshape(-1, K)

fck = torch.load(CFG.runs_local_dir / args.finder /
                 "best.pt", weights_only=False)
cfg = fck["config"]
finder = WindowFinder2D(width=cfg["width"], n_blocks=cfg.get("blocks", 3),
                        batchnorm=cfg.get("batchnorm", False)).to(dev).eval()
finder.load_state_dict(fck["model_state"])

if args.gevts:
    frames = [int(np.where(te["gevt"] == g)[0][0]) for g in args.gevts]
else:
    idx_file = PLOTS / "events" / "index.json"
    frames = []
    if idx_file.exists():
        entries = json.loads(idx_file.read_text())
        want = {"rescued": 1, "argmax_ok": 1}
        for e in entries:
            if want.get(e["category"], 0) > 0:
                frames.append(int(np.where(te["gevt"] == e["gevt"])[0][0]))
                want[e["category"]] -= 1
    if not frames:
        frames = [0]

t_edges = np.arange(N_BINS + 1) * BIN_NS + T_MIN
for i in frames:
    img = te["img"][i]
    t0_true = float(est["t0_true"][i])
    t0_est = float(est["t0_est"][i])
    with torch.no_grad():
        lg = finder(torch.from_numpy(img[None]).to(dev))[0].cpu().numpy()
    top = np.argsort(-lg)[:K]

    fig, axes = plt.subplots(
        4, 1, figsize=(11, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.5, 1.2, 1.2], "hspace": 0.38})
    for ax, ch, lab in [(axes[0], 0, "hit z (binned)"),
                        (axes[1], 1, "hit r (binned)")]:
        ax.imshow(img[ch], aspect="auto", origin="lower", cmap="Greys",
                  extent=[t_edges[0], t_edges[-1], 0, img[ch].shape[0]],
                  interpolation="nearest")
        ax.set_ylabel(lab, size=9)
        ax.set_yticks([])
    axes[0].set_title("what the finder sees: log(1+count) occupancy images",
                      size=10, loc="left")

    ax = axes[2]
    centers = t_edges[:-1] + BIN_NS / 2
    ax.margins(y=0.3)
    ax.step(centers, lg, where="mid", color="#2c5aa0", lw=1.2)
    ax.plot(centers[top], lg[top], "o", ms=5, mfc="none", mec="#c0392b",
            mew=1.4, label=f"top-{K} candidates")
    am = int(lg.argmax())
    ax.annotate("argmax", (centers[am], lg[am]), xytext=(0, 14), size=8,
                textcoords="offset points", ha="center", color="#c0392b",
                arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8))
    ax.set_ylabel("finder logit", size=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("stage 1: one score per 25 ns column — the burst is "
                 "visible but can be outranked", size=10, loc="left")

    ax = axes[3]
    sc = 1 / (1 + np.exp(-wver_grid[i]))
    chosen = np.argmin(np.abs(cand_t0[i] - t0_est))
    colors = ["#27ae60" if r == chosen else "#5b8ec4" for r in range(K)]
    ax.bar(cand_t0[i] + BIN_NS / 2, sc, width=BIN_NS * 0.9, color=colors)
    for r in range(K):
        ax.annotate(str(r), (cand_t0[i, r] + BIN_NS / 2, sc[r]), size=7,
                    xytext=(0, 2), textcoords="offset points", ha="center",
                    color="#1e8449" if r == chosen else "#31597e",
                    weight="bold" if r == chosen else "normal")
    ax.set_ylabel("verifier score", size=9)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("time in frame [ns]", size=9)
    ax.set_title("stage 2: each candidate window re-judged from its raw "
                 "hits (numbers = finder rank)", size=10, loc="left")

    for ax in axes:
        ax.axvline(t0_true, color="k", ls="--", lw=1.1)
    axes[0].annotate(" true t0", (t0_true, 0.86), size=8, ha="left",
                     xycoords=("data", "axes fraction"),
                     bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    pi = int(est["preset"][i])
    err = t0_est + BIN_NS / 2 - t0_true
    aerr = centers[am] - t0_true
    fig.suptitle(f"{SHORT[PRESETS[pi]]} frame {int(te['gevt'][i])}: "
                 f"argmax err {aerr:+.0f} ns, selected err {err:+.0f} ns",
                 size=11, y=0.995)
    fig.align_ylabels(axes)
    name = f"selection_anatomy_{SHORT[PRESETS[pi]]}_{int(te['gevt'][i])}.png"
    fig.savefig(PLOTS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}", flush=True)

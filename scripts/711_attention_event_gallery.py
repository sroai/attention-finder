"""Attention gallery: 50 real windows, four views each (plots/window/attention_events/).

Companion to script 45 (which makes the one summary figure). Here every one of
the first N test windows gets its own page, so the behaviour can be checked
event by event instead of trusted from an average.

Each page shows the SAME window in the (z, r) plane four times:

  (a) truth + what a SIGNAL hit listens to. The circled hit is the signal hit
      the model is most confident about; lines go to the hits it attends to,
      thickness/opacity ∝ a_ij (rows of the attention matrix sum to 1).
  (b) truth + what a BACKGROUND hit listens to (the most confidently rejected
      one). Same drawing rules. Note where the lines end up: on the burst.
  (c) attention RECEIVED per hit, in units of uniform attention (1.0 = an
      average hit that nobody looks at). This is the network's own saliency
      map, and nobody ever told it which hits are signal.
  (d) the model's per-hit background score, with the deployed in-window
      threshold. Hits above it are filtered. Compare (c) and (d): the hits the
      window STARES at are the hits that survive.

The header of each page carries the numbers for that window: hits, signal
hits, signal efficiency and background rejection at the deployed threshold,
and the mean attention-on-burst for each class.

Run: python scripts/711_attention_event_gallery.py [--n 50]
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
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.attention import WindowNet
from trkbkg.dataset import load_meta

DATA = (CFG.dataset_dir / "window_hits_v2")
HIT_META = (CFG.dataset_dir / "18x275_v2")
RUN = (CFG.runs_dir / "watt_r2")
OUT = (CFG.plots_dir / "window/attention_events")
TOP_K_LINES = 40

SIG, BKG, ALB = "#e8833a", "#c9ccd1", "#8f6fbf"   # signal / background / albedo
LINK = "#1f4e79"

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=50, help="windows to draw")
args = parser.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta = json.loads((DATA / "meta.json").read_text())
n_systems = len(load_meta(HIT_META)["system_ids"])
thr_in = json.loads((RUN / "metrics_end_to_end.json").read_text())["thresholds"]["inside"]

model = WindowNet(meta["norm"]["mean"], meta["norm"]["std"], n_systems,
                  d=64, n_blocks=2).to(dev)
model.load_state_dict(torch.load(RUN / "best.pt", map_location=dev)["model_state"])
model.eval()

d = np.load(DATA / "test.npz")
X, SYS, Y, EVT = d["X"], d["sys"].astype(np.int64), d["y"], d["evt"]
bounds = np.flatnonzero(np.diff(EVT.astype(np.int64))) + 1
starts, ends = np.concatenate([[0], bounds]), np.concatenate([bounds, [len(EVT)]])

OUT.mkdir(parents=True, exist_ok=True)
summary = []
drawn = 0
for w in range(len(starts)):
    s, e = starts[w], ends[w]
    n = e - s
    y = Y[s:e]
    if n < 150 or n > 2500 or (y == 0).sum() < 8:
        continue                       # too small / too heavy / no burst to see

    with torch.no_grad():
        logits, atts = model.forward_with_attention(
            torch.from_numpy(X[s:e]).to(dev), torch.from_numpy(SYS[s:e]).to(dev))
    score = torch.sigmoid(logits).cpu().numpy()      # P(background)
    a = np.mean([t.cpu().numpy() for t in atts], axis=0)   # avg of the 2 blocks

    x, yy_, z, dt = X[s:e].T
    r = np.hypot(x, yy_)
    is_sig, is_bkg, is_alb = y == 0, y == 1, y == 2
    recv = a.mean(axis=0) * n                       # attention received [x uniform]
    to_burst = a[:, is_sig].sum(axis=1)             # attention spent on the burst

    kept = score < thr_in
    eff = kept[is_sig].mean() if is_sig.any() else np.nan
    rej = (~kept[is_bkg]).mean() if is_bkg.any() else np.nan

    central = (np.abs(z) < 1600) & (r < 900)
    c_sig = np.flatnonzero(is_sig & central)
    c_bkg = np.flatnonzero(is_bkg & central)
    if not len(c_sig) or not len(c_bkg):
        continue
    q_sig = c_sig[np.argmin(score[c_sig])]          # most confident signal hit
    q_bkg = c_bkg[np.argmax(score[c_bkg])]          # most confident background hit

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.6))
    ZLIM = (-1900, 1900)

    def truth_scatter(ax):
        ax.scatter(z[is_bkg], r[is_bkg], s=8, c=BKG, linewidths=0, zorder=1,
                   label="background")
        if is_alb.any():
            ax.scatter(z[is_alb], r[is_alb], s=12, c=ALB, linewidths=0,
                       zorder=2, label="albedo (ignored)")
        ax.scatter(z[is_sig], r[is_sig], s=16, c=SIG, linewidths=0, zorder=3,
                   label="signal")

    def links(ax, q, title):
        truth_scatter(ax)
        wq = a[q]
        top = np.argsort(wq)[::-1][:TOP_K_LINES]
        wmax = wq[top].max()
        for j in top:
            if j == q:
                continue
            rel = float(np.clip(wq[j] / wmax, 0.0, 1.0))
            ax.plot([z[q], z[j]], [r[q], r[j]], lw=0.6 + 2.2 * rel,
                    color=LINK, alpha=0.25 + 0.65 * rel, zorder=4)
        ax.scatter([z[q]], [r[q]], s=170, facecolor="none", edgecolor=LINK,
                   linewidths=2.2, zorder=5, label="query hit")
        ax.set_title(f"{title}\nspends {wq[is_sig].sum():.0%} of its attention "
                     f"on the burst  ·  its score = {score[q]:.2f}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    links(axes[0, 0], q_sig, "(a) what the most confident SIGNAL hit listens to")
    links(axes[0, 1], q_bkg, "(b) what the most confident BACKGROUND hit listens to")

    ax = axes[1, 0]
    sc = ax.scatter(z, r, c=np.clip(recv, 0.01, None), s=14, cmap="magma",
                    norm=LogNorm(vmin=0.02, vmax=max(3.0, recv.max())),
                    linewidths=0)
    fig.colorbar(sc, ax=ax, label="attention received [× uniform]")
    # circle the true signal so the claim is checkable: the bright hits ARE
    # the signal, and the attention layer was never told that
    ax.scatter(z[is_sig], r[is_sig], s=42, facecolor="none", edgecolor="black",
               linewidths=0.6, zorder=3, label="true signal")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("(c) who the window LOOKS AT — the burst lights up,\n"
                 "and no label was ever given to the attention layer",
                 fontsize=10)

    ax = axes[1, 1]
    sc = ax.scatter(z, r, c=score, s=14, cmap="coolwarm", vmin=0, vmax=1,
                    linewidths=0)
    fig.colorbar(sc, ax=ax, label="model score = P(background)")
    ax.scatter(z[is_sig], r[is_sig], s=42, facecolor="none", edgecolor="black",
               linewidths=0.6, zorder=3, label="true signal")
    ax.set_title(f"(d) what the model DECIDES — filter if score > {thr_in:.2f}\n"
                 f"this window: {eff:.0%} of signal kept, "
                 f"{rej:.0%} of background rejected", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for ax in axes.ravel():
        ax.set_xlim(*ZLIM)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("r [mm]")
        ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle(
        f"window {int(EVT[s])} — {n} hits, {int(is_sig.sum())} signal "
        f"({is_sig.mean():.1%})\n"
        f"attention spent on the burst: {to_burst[is_sig].mean():.0%} by signal "
        f"hits, {to_burst[is_bkg].mean():.0%} by background hits   ·   "
        f"the burst RECEIVES {np.median(recv[is_sig]):.0f}× uniform attention, "
        f"background {np.median(recv[is_bkg]):.2f}×",
        fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / f"window_{int(EVT[s]):05d}.png", dpi=110)
    plt.close(fig)

    summary.append({"event": int(EVT[s]), "hits": int(n),
                    "signal_hits": int(is_sig.sum()),
                    "signal_eff_at_threshold": float(eff),
                    "bkg_rejection_at_threshold": float(rej),
                    "attention_on_burst_signal": float(to_burst[is_sig].mean()),
                    "attention_on_burst_background": float(to_burst[is_bkg].mean()),
                    "recv_median_signal": float(np.median(recv[is_sig])),
                    "recv_median_background": float(np.median(recv[is_bkg]))})
    drawn += 1
    if drawn >= args.n:
        break

(OUT / "index.json").write_text(json.dumps(
    {"threshold_inside": thr_in, "windows": summary}, indent=2))
eff_all = np.mean([w["signal_eff_at_threshold"] for w in summary])
rej_all = np.mean([w["bkg_rejection_at_threshold"] for w in summary])
print(f"drew {drawn} windows -> {OUT}")
print(f"mean over these windows: signal eff {eff_all:.1%}, bkg rejection "
      f"{rej_all:.1%} at threshold {thr_in:.3f}")

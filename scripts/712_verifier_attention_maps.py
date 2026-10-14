#!/usr/bin/env python
"""STEP 712 — what the WINDOW VERIFIER's attention looks at (slide figure).

The 2D finder proposes k candidate time windows; the verifier (trkbkg/
verifier.py, 2 self-attention blocks over the window's hits) scores each
one. This visualizes a RESCUE: a test frame where the finder's top-1
candidate is a background burst and the verifier picks the true window.

For every candidate window the verifier score is recomputed, and for the
two windows that matter (finder's wrong top-1, the true one) the attention
matrices a_ij are recomputed by hand (same maths as VerifierBlock.forward,
softmax kept). Plotted per hit: attention RECEIVED, mean over queries,
heads and blocks, in units of uniform attention — the landmark the window
is judged by.

Usage:
    python scripts/712_verifier_attention_maps.py \
        [--run wver_v3] [--data cand_windows_v2] [--frame N]

Outputs: plots/phase10/verifier_attention_f<frame>.png + .json
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
from trkbkg.verifier import WindowVerifier, pack_windows

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="wver_v3")
parser.add_argument("--data", default="cand_windows_v2")
parser.add_argument("--frame", type=int, default=None,
                    help="test-fold frame index; default: best rescue "
                         "(largest verifier margin over the finder's top-1)")
args = parser.parse_args()

DATA = CFG.dataset_dir / args.data
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase10"
PLOTS.mkdir(parents=True, exist_ok=True)

ckpt = torch.load(CFG.runs_local_dir / args.run / "best.pt",
                  weights_only=False, map_location="cpu")
cfg = ckpt["config"]
state = ckpt["model_state"]
n_systems, sys_dim = state["sys_emb.weight"].shape   # mu/sd come from the state dict
model = WindowVerifier(np.zeros(6), np.ones(6), n_systems,
                       d=cfg["dim"], n_blocks=cfg["blocks"], sys_dim=sys_dim).eval()
model.load_state_dict(state)

d = np.load(DATA / "test.npz")
K = json.loads((DATA / "meta.json").read_text())["k"]
scores2d = np.load(DATA / f"{args.run}_scores_test.npy")   # [n_frames, K]
scores = scores2d[d["frame"], d["rank"]]                   # per-window, flat
lens = np.diff(d["offs"])

# ---- choose the frame: finder top-1 wrong, verifier right, biggest margin
frames = d["frame"]
n_frames = int(frames.max()) + 1
rank = d["rank"]
label = d["label"]

if args.frame is None:
    best, best_margin = None, -1.0
    for f in range(n_frames):
        wi = np.flatnonzero(frames == f)
        if len(wi) < 2:
            continue
        top1 = wi[np.argmin(rank[wi])]
        pick = wi[np.argmax(scores[wi])]
        if label[top1] == 0 and label[pick] == 1:
            margin = scores[pick] - scores[top1]
            if margin > best_margin:
                best, best_margin = f, margin
    args.frame = int(best)
    print(f"picked frame {args.frame} (verifier margin {best_margin:.2f})")

wi = np.flatnonzero(frames == args.frame)
wi = wi[np.argsort(rank[wi])]
print(f"frame {args.frame} (preset {d['preset'][wi[0]]}): {len(wi)} candidates, "
      f"t0_true {d['t0_true'][wi[0]]:.0f} ns")


def window_hits(i):
    return d["hits"][d["offs"][i]:d["offs"][i + 1]]


def attention_of(i):
    """(hits, received[n] in units of uniform, A1[n,n] block-1 head-mean,
    verifier score) for window i — same maths as VerifierBlock.forward."""
    h = window_hits(i)
    X, S, pad = pack_windows([h])
    scalars = torch.tensor([[np.log1p(d["n_hits"][i]), d["logit"][i],
                             float(d["rank"][i])]], dtype=torch.float32)
    with torch.no_grad():
        score = float(model(X, S, pad, scalars))
        x = model.enc(torch.cat([(X - model.mu) / model.sd,
                                 model.sys_emb(S)], dim=-1))
        received, A1 = [], None
        for bi, blk in enumerate(model.blocks):
            B, N, dd = x.shape
            q, k, v = blk.qkv(blk.n1(x)).chunk(3, dim=-1)
            hs = dd // blk.h
            def split(t):
                return t.view(B, N, blk.h, hs).transpose(1, 2)
            att = (split(q) @ split(k).transpose(-2, -1)) / hs ** 0.5
            att = att.masked_fill(pad[:, None, None, :], -1e9).softmax(-1)
            if bi == 0:
                A1 = att[0].mean(dim=0).numpy()                # head-mean [N,N]
            received.append(att[0].mean(dim=(0, 1)).numpy())   # mean heads+queries
            out = (att @ split(v)).transpose(1, 2).reshape(B, N, dd)
            x = x + blk.proj(out)
            x = x + blk.ff(blk.n2(x))
    n = len(h)
    return h, np.mean(received, axis=0)[:n] * n, A1[:n, :n], score


# ---- figure: query lines on top (710 style), received maps below ----------
sc = [float(scores[i]) for i in wi]
top1_i = wi[0]
true_i = wi[np.flatnonzero(label[wi] == 1)[0]]
pick_i = wi[int(np.argmax(sc))]

QUERY_LINES = 40     # strongest links drawn per query hit
panels = ((top1_i, "the finder's top-1: a background burst",
           "#d62728"),
          (true_i, "the TRUE window (verifier's pick)", "#1a7f37"))

fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)

stats = {}
for ci, (i, name, edge) in enumerate(panels):
    h, att, A1, score = attention_of(i)
    t, z = h[:, 3], h[:, 2] / 1000
    watt = h[:, 5]
    comb_dt = float(d["t0_true"][i] - d["cand_t0"][i])
    in_comb = (t > comb_dt - 10) & (t < comb_dt + 35)
    stats[name] = {"score": score, "comb_received": float(att[in_comb].mean())
                   if in_comb.any() else 0.0}

    comb_in_window = bool(in_comb.any())
    xlim = (t.min() - 15, t.max() + 15)

    # (top) what one hit LISTENS TO: query = the most signal-like hit (watt)
    ax = axes[0, ci]
    qi = int(np.argmax(watt))
    ax.scatter(t, z, s=5, c="#b5b5b5", linewidths=0, label="window hits")
    row = A1[qi].copy()
    top = np.argsort(-row)[:QUERY_LINES]
    for j in top:
        a = row[j] / row[top[0]]
        ax.plot([t[qi], t[j]], [z[qi], z[j]], color="#2c5aa0",
                lw=0.4 + 2.8 * a, alpha=0.15 + 0.8 * a, zorder=2)
    ax.scatter(t[top], z[top], s=14, c="#2c5aa0", linewidths=0, zorder=3,
               label="what it listens to")
    ax.scatter([t[qi]], [z[qi]], s=140, facecolors="none",
               edgecolors="#1a3a6b", linewidths=2.5, label="query hit", zorder=5)
    spent = float(row[in_comb].sum()) if comb_in_window else 0.0
    line2 = (f"its most signal-like hit spends {100 * spent:.0f}% of its "
             "attention at the collision comb" if comb_in_window else
             "no collision in this window — attention finds no landmark")
    ax.set_title(f"{name}\nverifier score {score:+.1f} — {line2}",
                 fontsize=10, color=edge)
    ax.set_xlim(*xlim)
    ax.set_xlabel("t − window start [ns]")
    ax.set_ylabel("z [m]" if ci == 0 else "")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)

    # (bottom) who the window LOOKS AT: received attention
    ax = axes[1, ci]
    order = np.argsort(att)
    s = ax.scatter(t[order], z[order], c=att[order], s=12, cmap="inferno",
                   linewidths=0,
                   norm=matplotlib.colors.LogNorm(vmin=0.2, vmax=max(att.max(), 5)))
    if comb_in_window:
        ax.axvline(comb_dt, color="#1a7f37", lw=1.4, ls="--")
        ax.annotate("true collision time", (comb_dt, ax.get_ylim()[0]),
                    xytext=(8, 12), textcoords="offset points",
                    fontsize=8.5, color="#1a7f37")
        ax.set_title(f"attention RECEIVED per hit — comb hits get "
                     f"{stats[name]['comb_received']:.0f}× uniform", fontsize=10)
    else:
        ax.set_title("attention RECEIVED per hit — diffuse, no comb to "
                     "agree on", fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_xlabel("t − window start [ns]")
    ax.set_ylabel("z [m]" if ci == 0 else "")
    ax.grid(alpha=0.25)
    fig.colorbar(s, ax=ax, pad=0.01, label="attention received (1 = uniform)")

fig.suptitle(
    f"Verifier attention, test frame {args.frame} — the conv finder picks a "
    f"burst ({sc[0]:+.1f}); attention rescues the true window "
    f"({float(scores[true_i]):+.1f})", fontsize=12.5)
out = PLOTS / f"verifier_attention_f{args.frame}.png"
fig.savefig(out, dpi=115)
(out.with_suffix(".json")).write_text(json.dumps({
    "frame": args.frame, "preset": str(d["preset"][wi[0]]),
    "k": len(wi), "t0_true_ns": float(d["t0_true"][wi[0]]),
    "scores": sc, "ranks": [int(rank[i]) for i in wi],
    "labels": [int(label[i]) for i in wi],
    "picked_rank": int(rank[pick_i]), "true_rank": int(rank[true_i]),
    "margin": float(scores[pick_i] - scores[top1_i]),
}, indent=2))
print(f"wrote {out} and .json")

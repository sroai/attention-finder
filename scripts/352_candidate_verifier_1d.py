#!/usr/bin/env python
"""Phase 5 step 2: top-k candidate windows, verified by the attention model.

Usage (from the repo root):
    python scripts/352_candidate_verifier_1d.py \
        [--wf-run wf_v2_r3] [--win-run watt_r3] [--k 5]

Why: the finder's argmax finds 78% of frames, but the true bin is in its
top-5 for 94% (script 22) — the burst is visible, just out-ranked by
beam-gas bursts. The per-bin aggregates can't tell them apart; the attention
model CAN (it sees actual hit geometry). So: build a window around each of
the finder's top-k bins and let the attention model vote — the candidate
whose hits contain the largest expected signal count, sum(1 - score), wins.
No new training: pure composition of phase-3 + phase-4 models.

Outputs:
    dataset/t0_refined_v1/{fold}.npz    gevt + refined t0_est (feeds 14/16)
    plots/phase5/verifier_recall.png    recall & error hist, argmax vs verified
    reports/phase5_stats.json           (merged) recall/containment numbers
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
from trkbkg.attention import WindowNet
from trkbkg.dataset import load_fold, load_meta
from trkbkg.windows import BIN_NS, T_MIN, WindowFinderCNN

HIT_DATA = (CFG.dataset_dir / "18x275_v3")
WIN_DATA = (CFG.dataset_dir / "windows_v4")
WHITS = (CFG.dataset_dir / "window_hits_v3")
RUNS = CFG.runs_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase5"
PLOTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--wf-run", default="wf_v2_r3")
parser.add_argument("--win-run", default="watt_r3")
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--finder2d", default=None,
                    help="run name of a phase-6 2D finder; candidates then "
                         "come from its logits (18x275 frames of images2d)")
parser.add_argument("--out-name", default="t0_refined_v1")
args = parser.parse_args()
OUT = CFG.dataset_dir / args.out_name
OUT.mkdir(parents=True, exist_ok=True)

FRONT, BACK = 50.0, 300.0
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.finder2d:  # phase 6: candidates from the 2D-image finder
    from trkbkg.windows2d import WindowFinder2D
    IMG_DATA = (CFG.dataset_dir / "images2d_v1")
    ck2 = torch.load(CFG.runs_local_dir / args.finder2d
                     / "best.pt", weights_only=False)
    f2d = WindowFinder2D(width=ck2["config"]["width"],
                         n_blocks=ck2["config"].get("blocks", 3),
                         batchnorm=ck2["config"].get("batchnorm", False)
                         ).to(dev).eval()
    f2d.load_state_dict(ck2["model_state"])

    def frame_logits(fold, gevts_needed):
        d = np.load(IMG_DATA / f"{fold}.npz")
        m = d["preset"] == 0  # 18x275 frames only (hit folds exist for them)
        idx = {int(g): i for i, g in enumerate(d["gevt"][m])}
        img = d["img"][m]
        out = []
        with torch.no_grad():
            for i in range(0, len(img), 1024):
                out.append(f2d(torch.from_numpy(img[i:i + 1024]).to(dev))
                           .cpu().numpy())
        L = np.concatenate(out)
        rows = np.array([idx[int(g)] for g in gevts_needed])
        return L[rows]
else:
    wf_ckpt = torch.load(RUNS / args.wf_run / "best.pt", weights_only=False)
    wf = WindowFinderCNN(width=wf_ckpt["config"]["width"]).to(dev).eval()
    wf.load_state_dict(wf_ckpt["model_state"])

wmeta = json.loads((WHITS / "meta.json").read_text())
win_ckpt = torch.load(RUNS / args.win_run / "best.pt", weights_only=False)
wnet = WindowNet(wmeta["norm"]["mean"], wmeta["norm"]["std"],
                 len(load_meta(HIT_DATA)["system_ids"]),
                 d=win_ckpt["config"]["dim"],
                 n_blocks=win_ckpt["config"]["blocks"]).to(dev).eval()
wnet.load_state_dict(win_ckpt["model_state"])


# Selection strategy history (kept for the record):
#   sum(1-score)          -> rewards occupancy, recall 78% -> 32%. Dead.
#   count/coherence rules -> 78% -> 81%. Weak: watt never saw wrong windows
#                            in training, so its scores there are unreliable
#                            as a lone signal.
# Current: extract a FEATURE VECTOR per candidate window (verifier counts +
# geometry of confident hits + finder logit/rank) and train a tiny logistic
# candidate-classifier on the train fold — it learns from real negative
# candidates, which the hand rules and watt never did.
FEAT_NAMES = ["logit", "rank", "log_n_hits", "log_n_conf05", "log_n_conf02",
              "coherent02", "coherent05", "log_conf02_outer",
              "log_conf02_farfwd", "log_conf02_vertex", "log_sum_sig"]


def cand_features(logit, rank, Xw, sc, sys_of_hits):
    r = np.hypot(Xw[:, 0], Xw[:, 1])
    z = Xw[:, 2]
    c05, c02 = sc < 0.5, sc < 0.2
    return [
        float(logit), float(rank), np.log1p(len(sc)),
        np.log1p(c05.sum()), np.log1p(c02.sum()),
        float(len(np.unique(sys_of_hits[c02]))),
        float(len(np.unique(sys_of_hits[c05]))),
        np.log1p((c02 & (r > 550) & (np.abs(z) < 1600)).sum()),
        np.log1p((c02 & (z > 4000)).sum()),
        np.log1p((c02 & (r < 150)).sum()),
        np.log1p(float((1 - sc).sum())),
    ]


def build_candidates(fold):
    """One pass: features + labels for every top-k candidate of every frame."""
    win = np.load(WIN_DATA / f"{fold}.npz")
    if args.finder2d:
        logits = frame_logits(fold, win["gevt"])
    else:
        with torch.no_grad():
            logits = wf(torch.from_numpy(win["frames"]).to(dev)).cpu().numpy()
    top = np.argsort(-logits, axis=1)[:, :args.k]
    cand_t0 = top * BIN_NS + T_MIN

    X, sysi, y, evt = load_fold(HIT_DATA, fold)
    uevt, _ = np.unique(evt, return_inverse=True)
    order = dict(zip(uevt.tolist(), range(len(uevt))))
    starts = np.concatenate([[0], np.flatnonzero(np.diff(
        evt.astype(np.int64))) + 1, [len(evt)]])

    F = np.zeros((len(win["gevt"]), args.k, len(FEAT_NAMES)), np.float32)
    ok = np.zeros((len(win["gevt"]), args.k), bool)
    with torch.no_grad():
        for i, g in enumerate(win["gevt"]):
            fi = order.get(int(g))
            if fi is None:
                continue
            s, e = starts[fi], starts[fi + 1]
            t = X[s:e, 3]
            for c_i in range(args.k):
                t0c = cand_t0[i, c_i]
                m = (t >= t0c - FRONT) & (t <= t0c + BACK)
                if m.sum() < 5:
                    continue
                Xw = X[s:e][m].copy()
                Xw[:, 3] = t[m] - t0c
                sc = torch.sigmoid(wnet(
                    torch.from_numpy(Xw).to(dev),
                    torch.from_numpy(sysi[s:e][m]).to(dev))).cpu().numpy()
                F[i, c_i] = cand_features(logits[i, top[i, c_i]], c_i,
                                          Xw, sc, sysi[s:e][m])
                ok[i, c_i] = True
    # candidate label: window start within one bin of the true t0
    lab = np.abs(cand_t0 + BIN_NS / 2 - win["t0"][:, None]) <= 50
    return {"gevt": win["gevt"], "t0_true": win["t0"], "cand_t0": cand_t0,
            "F": F, "ok": ok, "label": lab}


def select_with_classifier(train_c, apply_c, epochs=300):
    """Tiny logistic candidate-classifier (CPU, seconds). Returns refined t0
    per frame of apply_c: argmax classifier prob over valid candidates."""
    def flat(c):
        m = c["ok"].reshape(-1)
        return (c["F"].reshape(-1, len(FEAT_NAMES))[m],
                c["label"].reshape(-1)[m].astype(np.float32))

    Xtr, ytr = flat(train_c)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    lin = torch.nn.Sequential(torch.nn.Linear(len(FEAT_NAMES), 16),
                              torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    Xt = torch.from_numpy((Xtr - mu) / sd)
    yt = torch.from_numpy(ytr)
    w = torch.where(yt == 1, (len(ytr) - ytr.sum()) / ytr.sum(), 1.0)
    for _ in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            lin(Xt).squeeze(1), yt, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        Fa = apply_c["F"]
        p = lin(torch.from_numpy(
            ((Fa.reshape(-1, len(FEAT_NAMES)) - mu) / sd))).squeeze(1)
        p = p.reshape(Fa.shape[0], Fa.shape[1]).numpy()
    p[~apply_c["ok"]] = -1e9
    sel = p.argmax(axis=1)
    return apply_c["cand_t0"][np.arange(len(sel)), sel], lin, (mu, sd)


def recall50(est, true):
    return float((np.abs(est + BIN_NS / 2 - true) <= 50).mean())


# candidate features are cached (one expensive GPU pass per fold) so the
# selector can be iterated for free
CACHE = OUT / f"cand_cache_k{args.k}"
CACHE.mkdir(exist_ok=True)
cands = {}
for fold in ("train", "val", "test"):
    cfile = CACHE / f"{fold}.npz"
    if cfile.exists():
        cands[fold] = dict(np.load(cfile))
        print(f"[{fold}] candidate cache loaded", flush=True)
    else:
        cands[fold] = build_candidates(fold)
        np.savez(cfile, **cands[fold])
        print(f"[{fold}] candidates built + cached", flush=True)

results = {}
best_crit = "classifier"
torch.manual_seed(0)
_, clf, norm = select_with_classifier(cands["train"], cands["train"])


def apply_clf(c):
    with torch.no_grad():
        Fa = c["F"]
        p = clf(torch.from_numpy(
            ((Fa.reshape(-1, len(FEAT_NAMES)) - norm[0]) / norm[1]))
        ).squeeze(1).reshape(Fa.shape[0], Fa.shape[1]).numpy()
    p[~c["ok"]] = -1e9
    sel = p.argmax(axis=1)
    return c["cand_t0"][np.arange(len(sel)), sel]


for fold in ("train", "val", "test"):
    c = cands[fold]
    t0_ref = apply_clf(c)
    rec = {"frames": len(c["t0_true"]),
           "recall50_argmax": recall50(c["cand_t0"][:, 0], c["t0_true"]),
           "recall50_classifier": recall50(t0_ref, c["t0_true"]),
           "topk_ceiling": float(c["label"].any(axis=1).mean())}
    results[fold] = rec
    print(fold, json.dumps(rec), flush=True)
    np.savez(OUT / f"{fold}.npz", gevt=c["gevt"], t0_est=t0_ref,
             t0_true=c["t0_true"])
    if fold == "test":
        te = (c["t0_true"], c["cand_t0"][:, 0], t0_ref)

stats_file = Path(__file__).resolve().parents[1] / "reports" / "phase5_stats.json"
merged = json.loads(stats_file.read_text()) if stats_file.exists() else {}
merged["candidate_verifier"] = {"k": args.k, "win_run": args.win_run,
                                **results}
stats_file.write_text(json.dumps(merged, indent=2))

t0_true, t0_argmax, t0_ref = te
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
bins = np.linspace(-500, 500, 100)
for est, color, lab in [(t0_argmax, "#999999", "argmax (phase 3)"),
                        (t0_ref, "#1a7f37",
                         f"top-{args.k} + verifier ({best_crit})")]:
    axes[0].hist(np.clip(est - t0_true, -500, 500), bins=bins,
                 histtype="step", lw=1.6, color=color, label=lab)
axes[0].set_xlabel("t0_est - t0 [ns] (test, clipped)")
axes[0].set_ylabel("frames"); axes[0].set_yscale("log"); axes[0].legend(fontsize=8)
folds = ["train", "val", "test"]
w = 0.35
xx = np.arange(len(folds))
axes[1].bar(xx - w/2, [results[f]["recall50_argmax"] for f in folds], w,
            color="#999999", label="argmax")
axes[1].bar(xx + w/2, [results[f][f"recall50_{best_crit}"] for f in folds], w,
            color="#1a7f37", label=f"verified ({best_crit})")
axes[1].set_xticks(xx, folds); axes[1].set_ylim(0.5, 1.0)
axes[1].set_ylabel("frames within 50 ns"); axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, axis="y")
fig.suptitle(f"phase 5: candidate verifier ({args.win_run} voting over "
             f"top-{args.k} windows of {args.wf_run})")
fig.tight_layout()
fig.savefig(PLOTS / "verifier_recall.png", dpi=130)
print(f"wrote {OUT} and {PLOTS/'verifier_recall.png'}")

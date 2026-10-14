#!/usr/bin/env python
"""Phase 6b FINAL: combined candidate selector = hand features + learned
verifier score.

Measured complementarity (test recall@50, overall / per preset):
  hand-feature reranker (31):  90.3  [18x275 92.7, 10x130 93.9, DVMP 81.7, DIS 84.2]
  learned verifier (33):       90.7  [91.9, 95.1, 83.0, 86.4]
  COMBINED (this script):      91.9  [93.4, 96.0, 84.2, 86.4]
The wver_v1 logit becomes a 12th raw feature next to the 11 hand features;
within-frame standardization doubles the set; tiny BCE MLP(32) selects.
Margin gate swept on val (lands at 0 — the combined selector never needs to
defer to the finder's argmax).

Inputs: cand_cache of script 31 + wver_v1 scores (computed here if the
.npy grids are absent — one GPU pass over cand_windows_v1, then cached).

Usage (from the repo root):
    python scripts/351_select_windows_combined.py \
        [--wver wver_v1] [--out-name t0_final_v1]

Outputs: dataset/<out-name>/{fold}.npz (refined t0), <runs_local_dir>/<out-name>/
{selector.pt,metrics.json}, plots/phase6/<out-name>_recall.png,
reports/phase6_<out-name>.json.
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
from trkbkg.reranker import apply_selector, train_selector, within_frame_std
from trkbkg.verifier import WindowVerifier
from trkbkg.windows import BIN_NS

PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase6"

parser = argparse.ArgumentParser()
parser.add_argument("--wver", default="wver_v1")
parser.add_argument("--out-name", default="t0_final_v1")
parser.add_argument("--rerank-name", default="t0_rerank_all_v1",
                    help="script-31 output dir holding cand_cache")
parser.add_argument("--cwin-name", default="cand_windows_v1",
                    help="script-32 candidate-window dataset dir")
args = parser.parse_args()
CACHE = (CFG.dataset_dir / args.rerank_name /
         "cand_cache")
CWIN = CFG.dataset_dir / args.cwin_name
OUT = CFG.dataset_dir / args.out_name
OUT.mkdir(parents=True, exist_ok=True)
run_dir = CFG.runs_local_dir / args.out_name
run_dir.mkdir(parents=True, exist_ok=True)

meta = json.loads((CWIN / "meta.json").read_text())
K = meta["k"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wver_scores(fold):
    """Verifier logit on the (frame, rank) grid; cached next to the hits."""
    cfile = CWIN / f"{args.wver}_scores_{fold}.npy"
    if cfile.exists():
        return np.load(cfile)
    ck = torch.load(CFG.runs_local_dir / args.wver /
                    "best.pt", weights_only=False, map_location=dev)
    model = WindowVerifier(meta["norm"]["mean"], meta["norm"]["std"],
                           len(meta["system_ids"]), d=ck["config"]["dim"],
                           n_blocks=ck["config"]["blocks"]).to(dev).eval()
    model.load_state_dict(ck["model_state"])
    d = np.load(CWIN / f"{fold}.npz")
    offs, hits = d["offs"], d["hits"]
    lens = np.diff(offs)
    scalars = np.column_stack([np.log1p(d["n_hits"]).astype(np.float32),
                               d["logit"], d["rank"].astype(np.float32)])
    out = np.zeros(len(lens), np.float32)
    order = np.argsort(lens, kind="stable")
    with torch.no_grad():
        for i in range(0, len(order), 256):
            idx = order[i:i + 256]
            N = max(1, int(lens[idx].max()))
            X = np.zeros((len(idx), N, 6), np.float32)
            S = np.zeros((len(idx), N), np.int64)
            pad = np.ones((len(idx), N), bool)
            for j, w in enumerate(idx):
                nh = lens[w]
                h = hits[offs[w]:offs[w + 1]]
                X[j, :nh] = h[:, :6]
                S[j, :nh] = h[:, 6]
                pad[j, :nh] = False
            out[idx] = model(torch.from_numpy(X).to(dev),
                             torch.from_numpy(S).to(dev),
                             torch.from_numpy(pad).to(dev),
                             torch.from_numpy(scalars[idx]).to(dev)
                             ).cpu().numpy()
    grid = np.zeros((int(d["frame"].max()) + 1, K), np.float32)
    grid[d["frame"], d["rank"].astype(int)] = out
    np.save(cfile, grid)
    return grid


c, feats = {}, {}
for fold in ("train", "val", "test"):
    c[fold] = dict(np.load(CACHE / f"{fold}.npz"))
    F = np.concatenate([c[fold]["F"], wver_scores(fold)[..., None]], axis=2)
    feats[fold] = within_frame_std(F, c[fold]["ok"])
    print(f"[{fold}] features ready", flush=True)

net, norm = train_selector(feats["train"], c["train"]["ok"],
                           c["train"]["label"], hidden=32)
torch.save({"state": net.state_dict(), "mu": norm[0], "sd": norm[1],
            "wver": args.wver}, run_dir / "selector.pt")


def recall50(t0_est, t0_true, m=None):
    hit = np.abs(t0_est + BIN_NS / 2 - t0_true) <= 50
    return float(hit.mean() if m is None else hit[m].mean())


margin, best_v = 0.0, -1.0
for m_try in (0.0, 0.25, 0.5, 0.75, 1.0):
    t0_m, _ = apply_selector(net, norm, feats["val"], c["val"]["ok"],
                             c["val"]["cand_t0"], margin=m_try)
    v = recall50(t0_m, c["val"]["t0_true"])
    if v > best_v:
        best_v, margin = v, m_try
print(f"val-chosen margin {margin}", flush=True)

presets = json.loads((Path(
    str(CFG.dataset_dir / "images2d_v1")) /
    "meta.json").read_text())["presets"]
metrics = {"selector": "combined hand+wver, bce32, within-frame std",
           "wver": args.wver, "margin": margin, "k": K}
for fold in ("train", "val", "test"):
    cf = c[fold]
    t0_ref, _ = apply_selector(net, norm, feats[fold], cf["ok"],
                               cf["cand_t0"], margin=margin)
    rec = {"frames": int(len(t0_ref)),
           "recall50_argmax": recall50(cf["cand_t0"][:, 0], cf["t0_true"]),
           "recall50_combined": recall50(t0_ref, cf["t0_true"]),
           "topk_ceiling": float(cf["label"].any(1).mean()),
           "per_preset": {}}
    for pi, name in enumerate(presets):
        m = cf["preset"] == pi
        if m.any():
            rec["per_preset"][name] = {
                "argmax": recall50(cf["cand_t0"][:, 0], cf["t0_true"], m),
                "combined": recall50(t0_ref, cf["t0_true"], m)}
    metrics[fold] = rec
    print(fold, json.dumps(rec), flush=True)
    np.savez(OUT / f"{fold}.npz", gevt=cf["gevt"], preset=cf["preset"],
             t0_est=t0_ref, t0_true=cf["t0_true"])
(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
(Path(__file__).resolve().parents[1] / "reports" /
 f"phase6_{args.out_name}.json").write_text(json.dumps(metrics, indent=2))

te = metrics["test"]
names = ["overall"] + list(te["per_preset"])
series = {
    "finder argmax": [te["recall50_argmax"]] +
    [v["argmax"] for v in te["per_preset"].values()],
    "combined selector": [te["recall50_combined"]] +
    [v["combined"] for v in te["per_preset"].values()],
}
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9.5, 4.5))
ax.bar(x - 0.2, series["finder argmax"], 0.4, label="finder argmax",
       color="#999999")
ax.bar(x + 0.2, series["combined selector"], 0.4,
       label="hand features + learned verifier", color="#b2182b")
for i in range(len(names)):
    ax.text(i + 0.2, series["combined selector"][i] + 0.012,
            f"{series['combined selector'][i]:.1%}", ha="center", fontsize=8)
ax.axhline(te["topk_ceiling"], color="k", ls=":", lw=1,
           label=f"top-{K} ceiling (overall)")
ax.set_xticks(x, [n.replace("background", "bg") for n in names],
              fontsize=8, rotation=10)
ax.set_ylabel("test recall within 50 ns"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="y")
ax.set_title(f"{args.out_name}: combined candidate selector (final)")
fig.tight_layout()
fig.savefig(PLOTS / f"{args.out_name}_recall.png", dpi=130)
print(f"wrote {PLOTS / f'{args.out_name}_recall.png'}")

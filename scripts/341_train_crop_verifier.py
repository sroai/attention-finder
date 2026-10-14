#!/usr/bin/env python
"""Phase 6b: train the crop-verifier — a second look at each candidate bin.

Motivation (measured, this run):
  * finder2d_v1 argmax recall@50ns: 87.9% overall; top-10 ceiling 98.7%.
  * hit-based reranker (script 23, 11 hand features + logistic): 92.3% on
    the 18x275 holdout — but it is stuck: deeper selectors and listwise
    training don't move val, because 11 numbers are 11 numbers and only
    18x275 has the per-hit folds it needs.
  * crop-verifier: a small CNN looks at the IMAGE CROP around each of the
    finder's top-k bins (shape: the vertical comb, the radial ladder —
    what the hand features cannot see) and trains on ALL presets
    (~22k frames x k candidates). Expectation recorded per protocol:
    +2-4 pts over the hit-based reranker on the same holdout, because it
    fixes both of that reranker's limits (features and training data).

Usage (from the repo root):
    python scripts/341_train_crop_verifier.py \
        cropver_v1 [--finder finder2d_v1] [--k 10] [--epochs 15]

Outputs: <runs_local_dir>/<run>/{best.pt,history.jsonl,config.json,metrics.json},
plots/phase6/<run>_selection.png, reports/phase6_<run>.json.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.windows import BIN_NS, N_BINS, T_MIN
from trkbkg.windows2d import (CROP_BEFORE, CROP_W, CandidateVerifier2D,
                              WindowFinder2D, candidate_crops)

DATA = (CFG.dataset_dir / "images2d_v1")
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase6"
PLOTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--finder", default="finder2d_v1")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch-frames", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

run_dir = CFG.runs_local_dir / args.run_name
run_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = torch.device("cpu" if args.cpu else
                   ("cuda" if torch.cuda.is_available() else "cpu"))

# ---- finder logits -> top-k candidates per fold ----------------------------
fdir = CFG.runs_local_dir / args.finder
fck = torch.load(fdir / "best.pt", weights_only=False)
finder = WindowFinder2D(width=fck["config"]["width"],
                        n_blocks=fck["config"].get("blocks", 3),
                        batchnorm=fck["config"].get("batchnorm", False)
                        ).to(dev).eval()
finder.load_state_dict(fck["model_state"])

meta = json.loads((DATA / "meta.json").read_text())
folds = {}
for fold in ("train", "val", "test"):
    d = np.load(DATA / f"{fold}.npz")
    lg = []
    with torch.no_grad():
        for i in range(0, len(d["img"]), 512):
            lg.append(finder(torch.from_numpy(d["img"][i:i + 512]).to(dev))
                      .cpu().numpy())
    logits = np.concatenate(lg)
    top = np.argsort(-logits, axis=1)[:, :args.k]
    cand_t0 = top * BIN_NS + T_MIN
    label = np.abs(cand_t0 + BIN_NS / 2 - d["t0"][:, None]) <= 50
    p = np.exp(logits - logits.max(1, keepdims=True))
    p = p / p.sum(1, keepdims=True)
    rows = np.arange(len(top))[:, None]
    scalars = np.stack([logits[rows, top],
                        np.broadcast_to(np.arange(args.k, dtype=np.float32),
                                        top.shape),
                        p[rows, top]], axis=2).astype(np.float32)
    folds[fold] = {"img": d["img"], "top": top, "cand_t0": cand_t0,
                   "label": label, "scalars": scalars, "t0": d["t0"],
                   "preset": d["preset"]}
    print(f"[{fold}] {len(top)} frames, argmax recall "
          f"{label[:, 0].mean():.4f}, top-{args.k} ceiling "
          f"{label.any(1).mean():.4f}", flush=True)

config = {"model": "candidate_verifier_2d", "finder": args.finder,
          "k": args.k, "crop": [CROP_BEFORE, CROP_W - CROP_BEFORE],
          "epochs": args.epochs, "batch_frames": args.batch_frames,
          "lr": args.lr}
model = CandidateVerifier2D().to(dev)
config["parameters"] = sum(p.numel() for p in model.parameters())
(run_dir / "config.json").write_text(json.dumps(config, indent=2))
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)


def frame_batch(fold, idx):
    """Crops + scalars + labels for a set of frames, flattened over k."""
    f = folds[fold]
    crops = candidate_crops(f["img"][idx], f["top"][idx])
    B = len(idx) * args.k
    return (torch.from_numpy(crops.reshape(B, *crops.shape[2:])),
            torch.from_numpy(f["scalars"][idx].reshape(B, -1)),
            torch.from_numpy(f["label"][idx].reshape(B).astype(np.float32)))


def select_recall(fold):
    """Recall@50 when the verifier's argmax candidate is taken per frame."""
    f = folds[fold]
    scores = []
    with torch.no_grad():
        for i in range(0, len(f["top"]), 256):
            idx = np.arange(i, min(i + 256, len(f["top"])))
            crop, sca, _ = frame_batch(fold, idx)
            scores.append(model(crop.to(dev), sca.to(dev))
                          .reshape(len(idx), args.k).cpu().numpy())
    sel = np.concatenate(scores).argmax(1)
    t0_ref = f["cand_t0"][np.arange(len(sel)), sel]
    hit = np.abs(t0_ref + BIN_NS / 2 - f["t0"]) <= 50
    return hit, t0_ref


pos_frac = folds["train"]["label"].mean()
pos_w = torch.tensor((1 - pos_frac) / pos_frac, device=dev) ** 0.5  # soften
print(f"train positives {pos_frac:.3f}, pos_weight {float(pos_w):.2f}",
      flush=True)

gen = torch.Generator().manual_seed(1)
best = -1.0
hist = open(run_dir / "history.jsonl", "a")
n_tr = len(folds["train"]["top"])
for epoch in range(1, args.epochs + 1):
    model.train()
    t_start = time.time()
    perm = torch.randperm(n_tr, generator=gen).numpy()
    loss_sum, nb = 0.0, 0
    for i in range(0, n_tr, args.batch_frames):
        crop, sca, y = frame_batch("train", perm[i:i + args.batch_frames])
        out = model(crop.to(dev), sca.to(dev))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out, y.to(dev), pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item(); nb += 1
    sched.step()

    model.eval()
    hit_va, _ = select_recall("val")
    rec = {"epoch": epoch, "train_loss": loss_sum / nb,
           "val_recall50": float(hit_va.mean()),
           "sec": round(time.time() - t_start, 1)}
    hist.write(json.dumps(rec) + "\n"); hist.flush()
    print(rec, flush=True)
    ckpt = {"model_state": model.state_dict(), "config": config,
            "epoch": epoch, "val_recall50": rec["val_recall50"]}
    torch.save(ckpt, run_dir / "last.pt")
    if rec["val_recall50"] > best:
        best = rec["val_recall50"]
        torch.save(ckpt, run_dir / "best.pt")
hist.close()

# ---- final evaluation with the best checkpoint -----------------------------
model.load_state_dict(torch.load(run_dir / "best.pt",
                                 weights_only=False)["model_state"])
model.eval()
metrics = {"run": args.run_name, "finder": args.finder, "k": args.k}
for fold in ("val", "test"):
    hit, t0_ref = select_recall(fold)
    f = folds[fold]
    m = {"frames": int(len(hit)),
         "recall50_argmax": float(f["label"][:, 0].mean()),
         "recall50_verifier": float(hit.mean()),
         "topk_ceiling": float(f["label"].any(1).mean()),
         "per_preset": {}}
    for pi, name in enumerate(meta["presets"]):
        pm = f["preset"] == pi
        if pm.any():
            m["per_preset"][name] = {
                "argmax": float(f["label"][pm, 0].mean()),
                "verifier": float(hit[pm].mean())}
    metrics[fold] = m
    if fold == "test":
        np.savez(CFG.dataset_dir /
                 f"t0_cropver_{args.run_name}_test.npz",
                 t0_est=t0_ref, t0_true=f["t0"], preset=f["preset"])
(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
stats_f = Path(__file__).resolve().parents[1] / "reports" / \
    f"phase6_{args.run_name}.json"
stats_f.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

# ---- plot: argmax vs verifier vs ceiling, overall + per preset -------------
te = metrics["test"]
names = ["overall"] + list(te["per_preset"])
arg = [te["recall50_argmax"]] + [v["argmax"] for v in te["per_preset"].values()]
ver = [te["recall50_verifier"]] + [v["verifier"]
                                   for v in te["per_preset"].values()]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x - 0.2, arg, 0.4, label="finder argmax", color="#999999")
ax.bar(x + 0.2, ver, 0.4, label="+ crop verifier", color="#1a7f37")
ax.axhline(te["topk_ceiling"], color="k", ls=":", lw=1,
           label=f"top-{args.k} ceiling (overall)")
ax.set_xticks(x, [n.replace("background", "bg") for n in names],
              fontsize=8, rotation=10)
ax.set_ylabel("test recall within 50 ns"); ax.set_ylim(0, 1.02)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
ax.set_title(f"{args.run_name}: crop-verifier candidate selection")
fig.tight_layout()
fig.savefig(PLOTS / f"{args.run_name}_selection.png", dpi=130)
print(f"wrote {PLOTS / f'{args.run_name}_selection.png'}")

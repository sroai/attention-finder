#!/usr/bin/env python
"""STEP 310 — train the per-hit signal/background classifier (the baseline).

WHAT GOES IN  : <dataset_dir>/<dataset>/{train,val}.npz   (from step 120)
WHAT COMES OUT: <runs_dir>/<run_name>/
                    config.json    everything needed to reproduce the run
                    history.jsonl  one line per epoch (loss, val AUC, ...)
                    best.pt        checkpoint with the best val rejection
                    last.pt        latest checkpoint (resume/inspect)

The model (trkbkg/models.py: HitMLP) sees ONE hit at a time — x, y, z,
time, and an embedding of the detector-system id — and outputs a "this is
background" score. Two things a newcomer should understand about the loop
below, both driven by the physics:

* Class imbalance. Only ~0.5% of hits are signal, so the loss weights each
  signal hit by ~n_bkg/n_sig. Training must deliver good RANKING of hits;
  the actual keep/filter threshold is chosen later (step 410) to guarantee
  >=99% signal efficiency.
* Albedo (label 2) is "don't care": physically unreconstructable late hits
  (see step 110). They are dropped from training entirely and excluded
  from the efficiency denominator during evaluation.

Usage (from the repo root):
    python scripts/310_train_hit_classifier.py mlp_v1
    ... 310_train_hit_classifier.py test_run --smoke        # 1-epoch check
    ... 310_train_hit_classifier.py mlp_big --hidden 128,128,64 --epochs 12

Speed note: the whole train fold (~50M hits) fits on the GPU as ~1.5 GB of
tensors, so by default it lives there and an epoch takes seconds. On a
busy/shared GPU pass --cpu-data (folds stay in host RAM, ~4x slower).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.dataset import load_fold, load_meta
from trkbkg.evaluate import rejection_at_efficiency, roc_auc
from trkbkg.models import build_model
from trkbkg.scoring import score_fold

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run_name", help="artifacts go to <runs_dir>/<run_name>")
parser.add_argument("--dataset", default="18x275_v3",
                    help="dataset dir name under <dataset_dir> (step 120)")
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=65_536)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden", default="64,64,32",
                    help="comma-separated MLP trunk widths")
parser.add_argument("--emb-dim", type=int, default=8,
                    help="size of the detector-system embedding")
parser.add_argument("--target-eff", type=float, default=0.99,
                    help="signal efficiency at which val rejection is "
                         "tracked (checkpoint selection metric)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--smoke", action="store_true",
                    help="train a single epoch to verify the pipeline")
parser.add_argument("--cpu-data", action="store_true",
                    help="keep folds in host RAM (shared/low-memory GPU)")
args = parser.parse_args()

epochs = 1 if args.smoke else args.epochs
dataset_dir = CFG.dataset_dir / args.dataset
run_dir = CFG.runs_dir / args.run_name
run_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed)
dev = torch.device(CFG.device if torch.cuda.is_available() else "cpu")

# ---- data -------------------------------------------------------------------
meta = load_meta(dataset_dir)
X_tr, sys_tr, y_tr, _ = load_fold(dataset_dir, "train")
X_va, sys_va, y_va, _ = load_fold(dataset_dir, "val")

if (y_tr == 2).any():                    # drop albedo from training entirely
    keep = y_tr < 2
    X_tr, sys_tr, y_tr = X_tr[keep], sys_tr[keep], y_tr[keep]

n_sig, n_bkg = int((y_tr == 0).sum()), int((y_tr == 1).sum())
signal_weight = n_bkg / n_sig            # fully rebalanced classes

config = {"dataset": str(dataset_dir), "model": "hit_mlp",
          "model_kwargs": {"hidden": tuple(int(w) for w
                                           in args.hidden.split(",")),
                           "emb_dim": args.emb_dim},
          "epochs": epochs, "batch_size": args.batch_size, "lr": args.lr,
          "signal_weight": signal_weight, "target_eff": args.target_eff,
          "seed": args.seed, "train_hits": len(y_tr),
          "config_profile": str(CFG.config_path)}
(run_dir / "config.json").write_text(json.dumps(config, indent=2))
print(json.dumps(config, indent=2))

model = build_model("hit_mlp", meta, **config["model_kwargs"]).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
bce = torch.nn.functional.binary_cross_entropy_with_logits

# folds resident on the GPU when it has room (--cpu-data: host RAM)
ddev = torch.device("cpu") if args.cpu_data else dev
X_tr = torch.from_numpy(X_tr).to(ddev)
sys_tr = torch.from_numpy(sys_tr).to(ddev)
y_tr_t = torch.from_numpy(y_tr).to(ddev)
w_tr = torch.where(y_tr_t == 0, float(signal_weight), 1.0)
X_va_t = torch.from_numpy(X_va).to(ddev)
sys_va_t = torch.from_numpy(sys_va).to(ddev)

# ---- the loop ----------------------------------------------------------------
best_rej = -1.0
hist_f = open(run_dir / "history.jsonl", "a")
for epoch in range(1, epochs + 1):
    model.train()
    t0 = time.time()
    loss_sum, n_batches = 0.0, 0
    perm = torch.randperm(len(y_tr_t), device=ddev)  # hits are independent:
    for i in range(0, len(perm), args.batch_size):   # shuffle them freely
        idx = perm[i:i + args.batch_size]
        logits = model(X_tr[idx].to(dev), sys_tr[idx].to(dev))
        loss = bce(logits, y_tr_t[idx].to(dev), weight=w_tr[idx].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        n_batches += 1

    # score the whole val fold; the metric that picks the checkpoint is
    # BACKGROUND REJECTION at the target signal efficiency — the physics
    # number — not the loss.
    scores_va = score_fold(model, X_va_t, sys_va_t)
    auc = roc_auc(y_va, scores_va)
    rej, thr = rejection_at_efficiency(y_va, scores_va, args.target_eff)
    rec = {"epoch": epoch, "train_loss": loss_sum / n_batches,
           "val_auc": auc, f"val_rej@{args.target_eff:.2f}": rej,
           "threshold": thr, "sec": round(time.time() - t0, 1)}
    hist_f.write(json.dumps(rec) + "\n")
    hist_f.flush()
    print(rec, flush=True)

    ckpt = {"model_state": model.state_dict(), "config": config,
            "epoch": epoch, "val_rejection": rej, "threshold": thr}
    torch.save(ckpt, run_dir / "last.pt")
    if rej > best_rej:
        best_rej = rej
        torch.save(ckpt, run_dir / "best.pt")

hist_f.close()
print(f"best val rejection @ {args.target_eff:.0%} eff: {best_rej:.4f}"
      f" -> {run_dir}/best.pt\nnext: 410_evaluate_hit_classifier.py "
      f"{args.run_name}")

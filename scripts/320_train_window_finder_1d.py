#!/usr/bin/env python
"""Train the window-finder CNN.

Usage (from the repo root):
    uv run python scripts/320_train_window_finder_1d.py <run_name> [--epochs N]

Training is an 88-way classification per frame ("which 25 ns slice holds the
interaction start"), so the loss is plain cross-entropy over bins. The whole
dataset is tiny (7k frames x 8 x 88 floats), so this trains in seconds.
Artifacts in <runs_dir>/<run_name>/ (same layout as
hit-model runs: config.json, history.jsonl, best.pt/last.pt); best checkpoint
is chosen by validation fraction-within-50ns.
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
from trkbkg.windows import BIN_NS, WindowFinderCNN

DATA = (CFG.dataset_dir / "windows_v4")
RUNS = CFG.runs_dir

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--width", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

run_dir = RUNS / args.run_name
run_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

fold = {name: np.load(DATA / f"{name}.npz") for name in ("train", "val")}
Xtr = torch.from_numpy(fold["train"]["frames"]).to(dev)
ytr = torch.from_numpy(fold["train"]["target"]).to(dev)
Xva = torch.from_numpy(fold["val"]["frames"]).to(dev)
t0_va = fold["val"]["t0"]
tgt_va = fold["val"]["target"]

config = {"model": "window_finder_cnn", "width": args.width,
          "epochs": args.epochs, "lr": args.lr, "bin_ns": BIN_NS,
          "train_frames": len(ytr), "val_frames": len(tgt_va)}
(run_dir / "config.json").write_text(json.dumps(config, indent=2))

model = WindowFinderCNN(width=args.width).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
ce = torch.nn.functional.cross_entropy

best = -1.0
hist_f = open(run_dir / "history.jsonl", "a")
for epoch in range(1, args.epochs + 1):
    model.train()
    t_start = time.time()
    perm = torch.randperm(len(ytr), device=dev)
    loss_sum, nb = 0.0, 0
    for i in range(0, len(perm), 512):
        idx = perm[i:i + 512]
        loss = ce(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item(); nb += 1
    sched.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xva).argmax(dim=1).cpu().numpy()
    err_ns = pred * BIN_NS + (-100.0) - t0_va  # bin start - true t0
    within = lambda w: float((np.abs(err_ns + BIN_NS / 2) <= w).mean())
    rec = {"epoch": epoch, "train_loss": loss_sum / nb,
           "val_within_25ns": within(25), "val_within_50ns": within(50),
           "val_within_100ns": within(100),
           "val_exact_bin": float((pred == tgt_va).mean()),
           "sec": round(time.time() - t_start, 2)}
    hist_f.write(json.dumps(rec) + "\n"); hist_f.flush()
    if epoch % 10 == 0 or epoch == 1:
        print(rec, flush=True)

    ckpt = {"model_state": model.state_dict(), "config": config,
            "epoch": epoch, "val_within_50ns": rec["val_within_50ns"]}
    torch.save(ckpt, run_dir / "last.pt")
    if rec["val_within_50ns"] > best:
        best = rec["val_within_50ns"]
        torch.save(ckpt, run_dir / "best.pt")

hist_f.close()
print(f"best val within-50ns: {best:.4f}")

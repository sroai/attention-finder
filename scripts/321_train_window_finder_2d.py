#!/usr/bin/env python
"""Phase 6: train the 2D-image window finder.

Usage (from the repo root):
    python scripts/321_train_window_finder_2d.py <run> \
        [--epochs 40] [--width 32] [--batch 256] [--no-aug]

Cross-entropy over the 88 time columns, cyclic time-shift augmentation on
every training batch (exact, since t0 is uniform — see windows2d.py).
Checkpoint selection by val recall@50ns. Run layout identical to phase 3.
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
from trkbkg.windows import BIN_NS, T_MIN
from trkbkg.windows2d import N_BINS, WindowFinder2D

RUNS = CFG.runs_dir

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--width", type=int, default=32)
parser.add_argument("--blocks", type=int, default=3)
parser.add_argument("--batchnorm", action="store_true")
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--no-aug", action="store_true")
parser.add_argument("--per-sample-roll", action="store_true",
                    help="an independent cyclic shift PER SAMPLE instead of "
                         "one shared shift per batch — more augmentation "
                         "diversity (the sweep showed capacity saturates "
                         "while v3_wide overfits, so diversity is the lever)")
parser.add_argument("--smooth", type=float, default=0.0,
                    help="label smoothing onto the two ADJACENT time bins "
                         "(cyclic): t0 near a bin edge makes the neighbour "
                         "nearly correct, so a hard one-hot target "
                         "over-penalizes near-misses the 50 ns metric accepts")
parser.add_argument("--local-run-dir", action="store_true",
                    help="write checkpoints under /work (the /data mount "
                         "intermittently hangs writers in D-state); copy to "
                         "runs/ afterwards")
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--dataset", default="images2d_v1",
                    help="dataset dir name under <dataset_dir> (phase 10 "
                         "uses images2d_p10, whose images carry 4 channels)")
parser.add_argument("--channels", type=int, default=0,
                    help="train on the FIRST N image channels only (0 = all)."
                         " The phase-10 A/B: --channels 2 = tracker-only "
                         "baseline, default = tracker + calorimeter")
args = parser.parse_args()

run_dir = (CFG.runs_local_dir / args.run_name
           if args.local_run_dir else RUNS / args.run_name)
run_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = torch.device("cpu" if args.cpu else
                   ("cuda" if torch.cuda.is_available() else "cpu"))

DATA = CFG.dataset_dir / args.dataset
tr = np.load(DATA / "train.npz")
va = np.load(DATA / "val.npz")
# images: ~30k x 2 x 64 x 88 x 4B = ~1.4 GB -> host RAM; batches to GPU
Xtr = torch.from_numpy(tr["img"])
ytr = torch.from_numpy(tr["target"].astype(np.int64))
Xva = torch.from_numpy(va["img"])
t0_va, tgt_va = va["t0"], va["target"]
if args.channels:
    Xtr = Xtr[:, :args.channels].contiguous()
    Xva = Xva[:, :args.channels].contiguous()
n_channels = Xtr.shape[1]

config = {"model": "window_finder_2d", "dataset": args.dataset,
          "channels": n_channels, "width": args.width,
          "blocks": args.blocks, "batchnorm": args.batchnorm,
          "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
          "augment": not args.no_aug, "per_sample_roll": args.per_sample_roll,
          "smooth": args.smooth, "train_frames": len(ytr),
          "val_frames": len(tgt_va)}
(run_dir / "config.json").write_text(json.dumps(config, indent=2))

model = WindowFinder2D(n_channels=n_channels, width=args.width,
                       n_blocks=args.blocks, batchnorm=args.batchnorm).to(dev)
config["parameters"] = sum(p.numel() for p in model.parameters())
(run_dir / "config.json").write_text(json.dumps(config, indent=2))
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
gen = torch.Generator().manual_seed(1)

best = -1.0
hist = open(run_dir / "history.jsonl", "a")
for epoch in range(1, args.epochs + 1):
    model.train()
    t_start = time.time()
    perm = torch.randperm(len(ytr), generator=gen)
    loss_sum, nb = 0.0, 0
    for i in range(0, len(perm), args.batch):
        idx = perm[i:i + args.batch]
        xb, yb = Xtr[idx], ytr[idx]
        if not args.no_aug:
            if args.per_sample_roll:  # independent cyclic shift per sample
                shifts = torch.randint(0, N_BINS, (len(idx),), generator=gen)
                cols = (torch.arange(N_BINS)[None, :]
                        - shifts[:, None]) % N_BINS          # out[j]=in[j-s]
                xb = torch.gather(
                    xb, -1, cols[:, None, None, :].expand(xb.shape))
                yb = (yb + shifts) % N_BINS
            else:                     # one shared shift per batch
                shift = int(torch.randint(0, N_BINS, (1,), generator=gen))
                xb = torch.roll(xb, shifts=shift, dims=-1)
                yb = (yb + shift) % N_BINS
        logits = model(xb.to(dev))
        yb = yb.to(dev)
        if args.smooth > 0:  # mass on the true bin's cyclic neighbours
            q = torch.zeros_like(logits)
            q.scatter_(1, yb[:, None], 1.0 - 2.0 * args.smooth)
            for d in (-1, 1):
                q.scatter_add_(1, ((yb + d) % N_BINS)[:, None],
                               torch.full((len(yb), 1), args.smooth,
                                          device=dev))
            loss = -(q * torch.log_softmax(logits, 1)).sum(1).mean()
        else:
            loss = torch.nn.functional.cross_entropy(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item(); nb += 1
    sched.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xva), 1024):
            preds.append(model(Xva[i:i + 1024].to(dev)).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    err = pred * BIN_NS + T_MIN + BIN_NS / 2 - t0_va
    rec = {"epoch": epoch, "train_loss": loss_sum / nb,
           "val_within_25ns": float((np.abs(err) <= 25).mean()),
           "val_within_50ns": float((np.abs(err) <= 50).mean()),
           "val_within_100ns": float((np.abs(err) <= 100).mean()),
           "sec": round(time.time() - t_start, 1)}
    hist.write(json.dumps(rec) + "\n"); hist.flush()
    print(rec, flush=True)
    ckpt = {"model_state": model.state_dict(), "config": config,
            "epoch": epoch, "val_within_50ns": rec["val_within_50ns"]}
    torch.save(ckpt, run_dir / "last.pt")
    if rec["val_within_50ns"] > best:
        best = rec["val_within_50ns"]
        torch.save(ckpt, run_dir / "best.pt")
hist.close()
print(f"best val within-50ns: {best:.4f}")

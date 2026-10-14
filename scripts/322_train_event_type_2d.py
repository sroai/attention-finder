#!/usr/bin/env python
"""STEP 322 — phase 12: train the two-head window + event-type network.

Joint training on the two-class dataset (step 133): the window head learns
the 88-column time softmax exactly as in 321 (cyclic roll augmentation,
neighbour label smoothing); the class head learns k-Λ vs DIS/NC from
features pooled around the TRUE window column (teacher forcing). Validation
runs the deployment path: class prediction at the PREDICTED window.

The phase-10-style A/B is the --channels flag on the same dataset:
    322_train_event_type_2d.py p12_trk --channels 2 --per-sample-roll --smooth 0.05
    322_train_event_type_2d.py p12_cal --per-sample-roll --smooth 0.05

Checkpoint selection: mean of val within-50ns and val class accuracy.
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
from trkbkg.windows2d import N_BINS, WindowTypeNet

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--width", type=int, default=32)
parser.add_argument("--blocks", type=int, default=3)
parser.add_argument("--batchnorm", action="store_true")
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--no-aug", action="store_true")
parser.add_argument("--per-sample-roll", action="store_true")
parser.add_argument("--smooth", type=float, default=0.0)
parser.add_argument("--cls-weight", type=float, default=1.0,
                    help="weight of the class CE next to the window CE")
parser.add_argument("--band-before", type=int, default=1)
parser.add_argument("--band-after", type=int, default=6)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--dataset", default="images2d_p12")
parser.add_argument("--channels", type=int, default=0,
                    help="first N image channels (0 = all 4); 2 = tracker-only")
args = parser.parse_args()

run_dir = CFG.runs_dir / args.run_name
run_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = torch.device("cpu" if args.cpu else
                   ("cuda" if torch.cuda.is_available() else "cpu"))

DATA = CFG.dataset_dir / args.dataset
tr = np.load(DATA / "train.npz")
va = np.load(DATA / "val.npz")
Xtr = torch.from_numpy(tr["img"])
ytr = torch.from_numpy(tr["target"].astype(np.int64))
ctr = torch.from_numpy(tr["cls"].astype(np.int64))
Xva = torch.from_numpy(va["img"])
t0_va, cls_va = va["t0"], va["cls"].astype(np.int64)
if args.channels:
    Xtr = Xtr[:, :args.channels].contiguous()
    Xva = Xva[:, :args.channels].contiguous()
n_channels = Xtr.shape[1]

config = {"model": "window_type_net", "dataset": args.dataset,
          "channels": n_channels, "width": args.width, "blocks": args.blocks,
          "batchnorm": args.batchnorm, "epochs": args.epochs,
          "batch": args.batch, "lr": args.lr, "augment": not args.no_aug,
          "per_sample_roll": args.per_sample_roll, "smooth": args.smooth,
          "cls_weight": args.cls_weight, "band_before": args.band_before,
          "band_after": args.band_after, "n_classes": 2,
          "train_frames": len(ytr), "val_frames": len(cls_va)}
model = WindowTypeNet(n_channels=n_channels, width=args.width,
                      n_blocks=args.blocks, batchnorm=args.batchnorm,
                      band_before=args.band_before,
                      band_after=args.band_after).to(dev)
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
    loss_sum = loss_t_sum = loss_c_sum = 0.0
    nb = 0
    for i in range(0, len(perm), args.batch):
        idx = perm[i:i + args.batch]
        xb, yb, cb = Xtr[idx], ytr[idx], ctr[idx]
        if not args.no_aug:
            if args.per_sample_roll:
                shifts = torch.randint(0, N_BINS, (len(idx),), generator=gen)
                cols = (torch.arange(N_BINS)[None, :]
                        - shifts[:, None]) % N_BINS
                xb = torch.gather(
                    xb, -1, cols[:, None, None, :].expand(xb.shape))
                yb = (yb + shifts) % N_BINS
            else:
                shift = int(torch.randint(0, N_BINS, (1,), generator=gen))
                xb = torch.roll(xb, shifts=shift, dims=-1)
                yb = (yb + shift) % N_BINS
        yb, cb = yb.to(dev), cb.to(dev)
        logits_t, logits_c = model(xb.to(dev), col=yb)   # teacher forcing
        if args.smooth > 0:
            q = torch.zeros_like(logits_t)
            q.scatter_(1, yb[:, None], 1.0 - 2.0 * args.smooth)
            for dshift in (-1, 1):
                q.scatter_add_(1, ((yb + dshift) % N_BINS)[:, None],
                               torch.full((len(yb), 1), args.smooth,
                                          device=dev))
            loss_t = -(q * torch.log_softmax(logits_t, 1)).sum(1).mean()
        else:
            loss_t = torch.nn.functional.cross_entropy(logits_t, yb)
        loss_c = torch.nn.functional.cross_entropy(logits_c, cb)
        loss = loss_t + args.cls_weight * loss_c
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item(); loss_t_sum += loss_t.item()
        loss_c_sum += loss_c.item(); nb += 1
    sched.step()

    model.eval()
    preds_t, preds_c = [], []
    with torch.no_grad():
        for i in range(0, len(Xva), 1024):
            lt, lc = model(Xva[i:i + 1024].to(dev))      # predicted window
            preds_t.append(lt.argmax(1).cpu().numpy())
            preds_c.append(lc.argmax(1).cpu().numpy())
    pred_t = np.concatenate(preds_t)
    pred_c = np.concatenate(preds_c)
    err = pred_t * BIN_NS + T_MIN + BIN_NS / 2 - t0_va
    rec = {"epoch": epoch, "train_loss": loss_sum / nb,
           "train_loss_window": loss_t_sum / nb,
           "train_loss_cls": loss_c_sum / nb,
           "val_within_50ns": float((np.abs(err) <= 50).mean()),
           "val_cls_acc": float((pred_c == cls_va).mean()),
           "sec": round(time.time() - t_start, 1)}
    rec["val_select"] = (rec["val_within_50ns"] + rec["val_cls_acc"]) / 2
    hist.write(json.dumps(rec) + "\n"); hist.flush()
    print(rec, flush=True)
    ckpt = {"model_state": model.state_dict(), "config": config,
            "epoch": epoch, "val_within_50ns": rec["val_within_50ns"],
            "val_cls_acc": rec["val_cls_acc"]}
    torch.save(ckpt, run_dir / "last.pt")
    if rec["val_select"] > best:
        best = rec["val_select"]
        torch.save(ckpt, run_dir / "best.pt")
hist.close()
print(f"best val select (mean of within-50ns and cls acc): {best:.4f}")

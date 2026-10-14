#!/usr/bin/env python
"""Train the phase-4 in-window hit classifier (attention or MLP baseline).

Usage (from the repo root):
    python scripts/330_train_inwindow_attention.py <run> \
        [--blocks 2] [--epochs 15]

--blocks 0 trains the no-attention baseline ("is dt alone enough?");
--blocks 2 (default) trains the attention model. Everything else is equal,
so the delta measures what hit-to-hit attention itself buys.

Batching: windows have variable hit counts, so each batch is a group of
windows padded to the longest one, with a padding mask that (a) blocks
attention to pad positions and (b) removes pad hits from the loss. Windows
are length-sorted before grouping to keep padding waste small.

Loss/metric protocol matches phase 1 (weighted BCE; per-epoch val AUC and
rejection at the IN-WINDOW efficiency that corresponds to 99% end-to-end
given the window containment from meta.json).
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
from trkbkg.attention import WindowNet
from trkbkg.dataset import load_meta
from trkbkg.evaluate import rejection_at_efficiency, roc_auc

HIT_META = (CFG.dataset_dir / "18x275_v3")
RUNS = CFG.runs_dir

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--data", default="window_hits_v3",
                    help="window-hit dataset dir name (phase 8: "
                         "window_hits_all_v1)")
parser.add_argument("--blocks", type=int, default=2)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--windows-per-batch", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

DATA = CFG.dataset_dir / args.data

run_dir = CFG.runs_local_dir / args.run_name  # /data (NTFS!) wedges writers in D-state
run_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meta = json.loads((DATA / "meta.json").read_text())
n_systems = len(load_meta(HIT_META)["system_ids"])


def load(fold):
    d = np.load(DATA / f"{fold}.npz")
    return (d["X"], d["sys"].astype(np.int64), d["y"].astype(np.float32),
            d["evt"])


X_tr, sys_tr, y_tr, evt_tr = load("train")
X_va, sys_va, y_va, evt_va = load("val")

n_sig, n_bkg = int((y_tr == 0).sum()), int((y_tr == 1).sum())
signal_weight = n_bkg / n_sig
# in-window efficiency needed for 99% end-to-end, given window containment
target_eff_inwin = min(0.999, 0.99 / meta["folds"]["val"]["signal_containment"])

config = {"model": "window_net", "blocks": args.blocks, "dim": args.dim,
          "epochs": args.epochs, "lr": args.lr,
          "windows_per_batch": args.windows_per_batch,
          "signal_weight": signal_weight,
          "target_eff_inwindow": target_eff_inwin,
          "window_hits_meta": meta, "train_hits": len(y_tr)}
(run_dir / "config.json").write_text(json.dumps(config, indent=2))

model = WindowNet(meta["norm"]["mean"], meta["norm"]["std"], n_systems,
                  d=args.dim, n_blocks=args.blocks).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
bce = torch.nn.functional.binary_cross_entropy_with_logits


def window_slices(evt):
    """(start, end) per window; hits are stored contiguously per event."""
    change = np.flatnonzero(np.diff(evt.astype(np.int64))) + 1
    starts = np.concatenate([[0], change])
    return list(zip(starts, np.concatenate([change, [len(evt)]])))


def make_groups(sorted_idx, lens, k, attn_budget=14_000_000):
    """Groups of up to k windows, shrunk for long windows so the padded
    attention matrix B*L_max^2 stays under attn_budget elements — the
    all-preset data has windows up to ~6700 hits, and a fixed batch of
    those OOMs a shared 12 GB GPU (sorted_idx is ascending in length, so
    L of the newest member is the group max)."""
    groups, cur = [], []
    for j in sorted_idx:
        cur.append(j)
        cap = max(1, min(k, attn_budget // max(1, int(lens[j]) ** 2)))
        if len(cur) >= cap:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def batches(slices, groups):
    """Yield padded (X [B,L,4], sys [B,L], y [B,L], pad [B,L]) batches."""
    for g in groups:
        group = [slices[j] for j in g]
        L = max(e - s for s, e in group)
        B = len(group)
        Xb = np.zeros((B, L, 4), np.float32)
        sb = np.zeros((B, L), np.int64)
        yb = np.zeros((B, L), np.float32)
        pad = np.ones((B, L), bool)
        for b, (s, e) in enumerate(group):
            n = e - s
            Xb[b, :n], sb[b, :n], yb[b, :n] = X_cur[s:e], sys_cur[s:e], y_cur[s:e]
            pad[b, :n] = False
        yield (torch.from_numpy(Xb).to(dev), torch.from_numpy(sb).to(dev),
               torch.from_numpy(yb).to(dev), torch.from_numpy(pad).to(dev))


tr_slices = window_slices(evt_tr)
va_slices = window_slices(evt_va)
# length-sorted order minimizes padding; reshuffle groups each epoch
tr_lens = np.array([e - s for s, e in tr_slices])
tr_sorted = np.argsort(tr_lens)
tr_groups = make_groups(tr_sorted, tr_lens, args.windows_per_batch)

best_rej = -1.0
hist_f = open(run_dir / "history.jsonl", "a")
for epoch in range(1, args.epochs + 1):
    model.train()
    t0 = time.time()
    X_cur, sys_cur, y_cur = X_tr, sys_tr, y_tr
    groups = list(tr_groups)
    rng = np.random.default_rng(epoch)
    rng.shuffle(groups)
    loss_sum, nb = 0.0, 0
    for Xb, sb, yb, pad in batches(tr_slices, groups):
        logits = model.forward_padded(Xb, sb, pad)
        # albedo hits (y==2) get weight 0: don't-care in the loss
        w = torch.where(yb == 0, signal_weight, 1.0) * (~pad) * (yb < 2)
        loss = (bce(logits, yb.clamp(max=1.0), weight=w, reduction="sum")
                / w.sum())
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item(); nb += 1
    sched.step()

    model.eval()
    X_cur, sys_cur, y_cur = X_va, sys_va, y_va
    scores = np.empty(len(y_va), np.float32)
    with torch.no_grad():
        pos = 0
        for s, e in va_slices:
            out = torch.sigmoid(model(
                torch.from_numpy(X_va[s:e]).to(dev),
                torch.from_numpy(sys_va[s:e]).to(dev))).cpu().numpy()
            scores[s:e] = out
            pos += e - s
    auc = roc_auc(y_va, scores)
    rej, thr = rejection_at_efficiency(y_va, scores, target_eff_inwin)
    rec = {"epoch": epoch, "train_loss": loss_sum / nb, "val_auc": auc,
           f"val_rej@{target_eff_inwin:.4f}": rej, "threshold": thr,
           "sec": round(time.time() - t0, 1)}
    hist_f.write(json.dumps(rec) + "\n"); hist_f.flush()
    print(rec, flush=True)

    ckpt = {"model_state": model.state_dict(), "config": config,
            "epoch": epoch, "val_rejection": rej, "threshold": thr}
    torch.save(ckpt, run_dir / "last.pt")
    if rej > best_rej:
        best_rej = rej
        torch.save(ckpt, run_dir / "best.pt")

hist_f.close()
print(f"best val in-window rejection: {best_rej:.4f}")

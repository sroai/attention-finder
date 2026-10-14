#!/usr/bin/env python
"""Phase 6b: train the LEARNED window verifier on candidate-window hits.

Replaces the 11 hand features + tiny MLP (script 31, test 90.3% overall)
with learned pooling over raw hits (trkbkg/verifier.py). Expectation
(recorded per protocol): +1.5-4 pts overall — the hand-feature selector is
feature-limited (measured), and this model sees hard negatives watt never
saw; biggest expected gain on DVMP (dim signal the pooled counts miss).

Frame-level selection: verifier score per candidate, argmax with a margin
gate over the finder's rank-0 (margin tuned on val, same policy as 31).

Usage (from the repo root):
    python scripts/340_train_window_verifier.py \
        wver_v1 [--blocks 2] [--epochs 20] [--batch 96]

Outputs: <runs_local_dir>/<run>/{best.pt,history.jsonl,config.json,metrics.json},
dataset/t0_verifier_<run>/{fold}.npz, plots/phase6/<run>_recall.png,
reports/phase6_<run>.json.
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
from trkbkg.verifier import WindowVerifier
from trkbkg.windows import BIN_NS

PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase6"
PLOTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--data", default="cand_windows_v1",
                    help="candidate-window dataset dir name")
parser.add_argument("--blocks", type=int, default=2,
                    help="attention blocks; 0 = DeepSets baseline")
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch", type=int, default=96)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

run_dir = CFG.runs_local_dir / args.run_name
run_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = torch.device("cpu" if args.cpu else
                   ("cuda" if torch.cuda.is_available() else "cpu"))

DATA = CFG.dataset_dir / args.data
meta = json.loads((DATA / "meta.json").read_text())
K = meta["k"]


class Fold:
    """Bucketed batches over variable-length windows, GPU-ready tensors."""

    def __init__(self, fold):
        d = np.load(DATA / f"{fold}.npz")
        self.hits = d["hits"]
        self.offs = d["offs"]
        self.wid = d["window_id"]          # global window id = frame*K+rank
        self.n_hits = d["n_hits"]
        self.label = d["label"]
        self.scalars = np.column_stack([
            np.log1p(d["n_hits"]).astype(np.float32),
            d["logit"], d["rank"].astype(np.float32)])
        self.cand_t0 = d["cand_t0"]
        self.frame = d["frame"]
        self.preset = d["preset"]
        self.t0_true = d["t0_true"]
        self.n_frames = int(self.frame.max()) + 1
        self.lens = np.diff(self.offs)

    def batches(self, batch, shuffle, gen=None):
        """Yield index arrays, bucketed by window length to limit padding."""
        order = np.argsort(self.lens, kind="stable")
        chunks = [order[i:i + batch] for i in range(0, len(order), batch)]
        if shuffle:
            perm = torch.randperm(len(chunks), generator=gen).numpy()
            chunks = [chunks[i] for i in perm]
        return chunks

    def pack(self, idx):
        N = max(1, int(self.lens[idx].max()))
        B = len(idx)
        X = np.zeros((B, N, 6), np.float32)
        S = np.zeros((B, N), np.int64)
        pad = np.ones((B, N), bool)
        for j, w in enumerate(idx):
            n = self.lens[w]
            h = self.hits[self.offs[w]:self.offs[w + 1]]
            X[j, :n] = h[:, :6]
            S[j, :n] = h[:, 6]
            pad[j, :n] = False
        return (torch.from_numpy(X).to(dev), torch.from_numpy(S).to(dev),
                torch.from_numpy(pad).to(dev),
                torch.from_numpy(self.scalars[idx]).to(dev),
                torch.from_numpy(self.label[idx].astype(np.float32)).to(dev))


folds = {f: Fold(f) for f in ("train", "val", "test")}
print({f: (folds[f].n_frames, len(folds[f].label)) for f in folds},
      flush=True)

model = WindowVerifier(meta["norm"]["mean"], meta["norm"]["std"],
                       len(meta["system_ids"]), d=args.dim,
                       n_blocks=args.blocks).to(dev)
config = {"model": "window_verifier", "blocks": args.blocks, "dim": args.dim,
          "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
          "k": K, "parameters": sum(p.numel() for p in model.parameters())}
(run_dir / "config.json").write_text(json.dumps(config, indent=2))
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
pos = folds["train"].label.mean()
pos_w = torch.tensor(float(((1 - pos) / pos) ** 0.5), device=dev)


def window_scores(fold):
    f = folds[fold]
    out = np.zeros(len(f.label), np.float32)
    model.eval()
    with torch.no_grad():
        for idx in f.batches(512, shuffle=False):
            X, S, pad, sca, _ = f.pack(idx)
            out[idx] = model(X, S, pad, sca).cpu().numpy()
    return out


def selection_recall(fold, scores, margin=0.0):
    """Frame-level: argmax candidate (margin-gated vs rank 0)."""
    f = folds[fold]
    s = np.full((f.n_frames, K), -1e9, np.float32)
    t0c = np.zeros((f.n_frames, K), np.float32)
    s[f.frame, f.wid % K] = scores
    t0c[f.frame, f.wid % K] = f.cand_t0
    sel = s.argmax(1)
    if margin > 0:
        sel = np.where(s[np.arange(len(sel)), sel] - s[:, 0] > margin,
                       sel, 0)
    t0_est = t0c[np.arange(len(sel)), sel]
    t0_true = np.zeros(f.n_frames, np.float32)
    t0_true[f.frame] = f.t0_true
    preset = np.zeros(f.n_frames, np.int8)
    preset[f.frame] = f.preset
    hit = np.abs(t0_est + BIN_NS / 2 - t0_true) <= 50
    return hit, preset, t0_est, t0_true


gen = torch.Generator().manual_seed(1)
best = -1.0
hist = open(run_dir / "history.jsonl", "a")
for epoch in range(1, args.epochs + 1):
    model.train()
    t_start = time.time()
    loss_sum, nb = 0.0, 0
    for idx in folds["train"].batches(args.batch, shuffle=True, gen=gen):
        X, S, pad, sca, y = folds["train"].pack(idx)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(X, S, pad, sca), y, pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item(); nb += 1
    sched.step()

    hit_va, _, _, _ = selection_recall("val", window_scores("val"))
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

# ---- final eval: margin on val, then test ----------------------------------
model.load_state_dict(torch.load(run_dir / "best.pt",
                                 weights_only=False)["model_state"])
sc_va = window_scores("val")
margin, best_v = 0.0, -1.0
for m_try in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
    v = float(selection_recall("val", sc_va, m_try)[0].mean())
    if v > best_v:
        best_v, margin = v, m_try
print(f"val-chosen margin {margin}", flush=True)

metrics = {"run": args.run_name, "blocks": args.blocks, "margin": margin,
           "baseline_reranker_test_overall": 0.9033}
presets = json.loads((Path(
    str(CFG.dataset_dir / "images2d_v1")) /
    "meta.json").read_text())["presets"]
for fold in ("val", "test"):
    sc = sc_va if fold == "val" else window_scores(fold)
    hit, preset, t0_est, t0_true = selection_recall(fold, sc, margin)
    hit0, _, _, _ = selection_recall(fold, sc * 0)  # rank-0 = argmax
    m = {"frames": int(len(hit)),
         "recall50_argmax": float(hit0.mean()),
         "recall50_verifier": float(hit.mean()), "per_preset": {}}
    for pi, name in enumerate(presets):
        pm = preset == pi
        if pm.any():
            m["per_preset"][name] = {"argmax": float(hit0[pm].mean()),
                                     "verifier": float(hit[pm].mean())}
    metrics[fold] = m
    f = folds[fold]
    frame_gevt = np.zeros(f.n_frames, np.int64)
    frame_gevt[f.frame] = np.load(DATA / f"{fold}.npz")["gevt"]
    out_dir = CFG.dataset_dir / \
        f"t0_verifier_{args.run_name}"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / f"{fold}.npz", gevt=frame_gevt, preset=preset,
             t0_est=t0_est, t0_true=t0_true)
(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
stats_f = Path(__file__).resolve().parents[1] / "reports" / \
    f"phase6_{args.run_name}.json"
stats_f.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

te = metrics["test"]
names = ["overall"] + list(te["per_preset"])
arg = [te["recall50_argmax"]] + [v["argmax"]
                                 for v in te["per_preset"].values()]
ver = [te["recall50_verifier"]] + [v["verifier"]
                                   for v in te["per_preset"].values()]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9.5, 4.5))
ax.bar(x - 0.2, arg, 0.4, label="finder argmax", color="#999999")
ax.bar(x + 0.2, ver, 0.4, label="+ learned verifier", color="#7b3294")
for i in range(len(names)):
    ax.text(i + 0.2, ver[i] + 0.012, f"{ver[i]:.1%}", ha="center", fontsize=8)
ax.set_xticks(x, [n.replace("background", "bg") for n in names],
              fontsize=8, rotation=10)
ax.set_ylabel("test recall within 50 ns"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="y")
ax.set_title(f"{args.run_name}: learned window verifier "
             f"(blocks={args.blocks}, margin={margin})")
fig.tight_layout()
fig.savefig(PLOTS / f"{args.run_name}_recall.png", dpi=130)
print(f"wrote {PLOTS / f'{args.run_name}_recall.png'}")

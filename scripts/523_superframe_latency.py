#!/usr/bin/env python
"""STEP 523 — inference latency on STACKED frames (EIC super-frames).

Streaming readout can concatenate k consecutive 2 us frames into one
super-frame. The finder is fully convolutional along time (space-only
pooling, 1D-conv head), so a [C, 64, 88*k] image runs through the SAME
weights and yields 88*k per-column logits — no retraining, no reshaping.

This measures how batch-1 latency scales with k on one CPU thread (the
production shape) and on the GPU, and reports the amortized cost per 2 us
frame. Super-frames are built by concatenating k distinct test frames
along the time axis — compute-wise identical to a real merged window.

Usage:
    python scripts/523_superframe_latency.py p10_cal \
        [--dataset images2d_p10] [--stacks 1 2 3 5 10] [--repeats 200]

Output: printed table + <results>/plots-phase10/superframe_latency.png/json.
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
from trkbkg.windows2d import WindowFinder2D

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--dataset", default="images2d_p10")
parser.add_argument("--stacks", type=int, nargs="+", default=[1, 2, 3, 5, 10])
parser.add_argument("--repeats", type=int, default=200)
args = parser.parse_args()

te = np.load(CFG.dataset_dir / args.dataset / "test.npz")
run_dir = CFG.runs_local_dir / args.run_name
if not (run_dir / "best.pt").exists():
    run_dir = CFG.runs_dir / args.run_name
ckpt = torch.load(run_dir / "best.pt", weights_only=False, map_location="cpu")
cfg = ckpt["config"]
n_ch = cfg.get("channels", 2)

out_dir = Path(CFG.results_root) / "plots-phase10" if hasattr(CFG, "results_root") \
    else Path("plots-phase10")
out_dir.mkdir(parents=True, exist_ok=True)

devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
results = {}

for dev_name in devices:
    dev = torch.device(dev_name)
    if dev_name == "cpu":
        torch.set_num_threads(1)
    model = WindowFinder2D(n_channels=n_ch, width=cfg["width"],
                           n_blocks=cfg.get("blocks", 3),
                           batchnorm=cfg.get("batchnorm", False)).to(dev).eval()
    model.load_state_dict(ckpt["model_state"])

    for k in args.stacks:
        n = min(args.repeats, len(te["img"]) // k)
        stacked = torch.from_numpy(np.concatenate(
            [np.concatenate(te["img"][i * k:(i + 1) * k, :n_ch], axis=-1)[None]
             for i in range(n)]))
        with torch.no_grad():
            for i in range(5):
                model(stacked[i:i + 1].to(dev))
            if dev_name == "cuda":
                torch.cuda.synchronize()
            times = np.empty(n)
            for i in range(n):
                x = stacked[i:i + 1].to(dev)
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(x)
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                times[i] = (time.perf_counter() - t0) * 1e3
        med = float(np.median(times))
        results[f"{dev_name}/k{k}"] = {
            "median_ms": med, "p99_ms": float(np.percentile(times, 99)),
            "per_frame_ms": med / k, "logit_bins": 88 * k}
        print(f"{dev_name:4s} k={k:2d}  super-frame {med:7.2f} ms  "
              f"per 2us frame {med / k:6.2f} ms", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for dev_name, marker in zip(devices, ("o", "s")):
    ks = args.stacks
    tot = [results[f"{dev_name}/k{k}"]["median_ms"] for k in ks]
    per = [results[f"{dev_name}/k{k}"]["per_frame_ms"] for k in ks]
    label = "1 CPU thread" if dev_name == "cpu" else "GPU"
    ax[0].plot(ks, tot, marker=marker, label=label)
    ax[1].plot(ks, per, marker=marker, label=label)
ax[0].set_xlabel("frames stacked (k)"); ax[0].set_ylabel("super-frame latency [ms]")
ax[1].set_xlabel("frames stacked (k)"); ax[1].set_ylabel("amortized per 2 us frame [ms]")
for a in ax:
    a.grid(alpha=0.3); a.legend(); a.set_xticks(args.stacks)
ax[0].set_ylim(bottom=0); ax[1].set_ylim(bottom=0)
fig.suptitle(f"WindowFinder2D on stacked frames ({args.run_name}, {n_ch} channels, "
             "same weights, 88k output columns)")
fig.tight_layout()
fig.savefig(out_dir / "superframe_latency.png", dpi=110)
(out_dir / "superframe_latency.json").write_text(json.dumps(results, indent=2))
print("wrote", out_dir / "superframe_latency.png")

#!/usr/bin/env python
"""STEP 522 — per-frame inference latency of the 2D window finder.

Times SINGLE-FRAME (batch=1) forward passes — the deployment shape: EICrecon
calls the finder once per frame, on one CPU in production (ONNX), so the CPU
numbers use torch.set_num_threads(1). GPU numbers time the same calls on the
available CUDA device with proper synchronization.

Output: <runs>/../plots-phase10/finder2d_latency.png + latency.json next to
it (mean/median/p95/p99 per model x device), printed summary.

Usage:
    python scripts/522_finder2d_latency.py \
        [--runs p10_trk p10_cal] [--dataset images2d_p10] [--frames 500]
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
parser.add_argument("--runs", nargs="+", default=["p10_trk", "p10_cal"])
parser.add_argument("--dataset", default="images2d_p10")
parser.add_argument("--frames", type=int, default=500)
parser.add_argument("--out-dir", type=Path, default=None)
args = parser.parse_args()

te = np.load(CFG.dataset_dir / args.dataset / "test.npz")
out_dir = args.out_dir or (CFG.results_root / "plots-phase10"
                           if hasattr(CFG, "results_root") else Path("plots-phase10"))
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
results = {}
fig, axes = plt.subplots(1, len(devices), figsize=(6.5 * len(devices), 4.4),
                         squeeze=False)

for run_name in args.runs:
    run_dir = CFG.runs_local_dir / run_name
    if not (run_dir / "best.pt").exists():
        run_dir = CFG.runs_dir / run_name
    ckpt = torch.load(run_dir / "best.pt", weights_only=False,
                      map_location="cpu")
    cfg = ckpt["config"]
    n_ch = cfg.get("channels", 2)
    frames = torch.from_numpy(te["img"][: args.frames, :n_ch]).contiguous()

    for di, dev_name in enumerate(devices):
        dev = torch.device(dev_name)
        if dev_name == "cpu":
            torch.set_num_threads(1)   # production: one CPU per frame
        model = WindowFinder2D(n_channels=n_ch, width=cfg["width"],
                               n_blocks=cfg.get("blocks", 3),
                               batchnorm=cfg.get("batchnorm", False)).to(dev)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        with torch.no_grad():
            for i in range(10):                      # warmup
                model(frames[i:i + 1].to(dev))
            if dev_name == "cuda":
                torch.cuda.synchronize()
            times_ms = np.empty(len(frames))
            for i in range(len(frames)):
                x = frames[i:i + 1].to(dev)
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(x)
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                times_ms[i] = (time.perf_counter() - t0) * 1e3

        stats = {"mean_ms": float(times_ms.mean()),
                 "median_ms": float(np.median(times_ms)),
                 "p95_ms": float(np.percentile(times_ms, 95)),
                 "p99_ms": float(np.percentile(times_ms, 99))}
        results[f"{run_name}/{dev_name}"] = stats
        print(f"{run_name:8s} {dev_name:4s}  median {stats['median_ms']:6.2f} ms  "
              f"mean {stats['mean_ms']:6.2f}  p99 {stats['p99_ms']:6.2f}", flush=True)

        ax = axes[0][di]
        ax.hist(times_ms, bins=60, histtype="step", lw=1.6,
                label=f"{run_name} ({n_ch} ch), median {stats['median_ms']:.2f} ms")

for di, dev_name in enumerate(devices):
    ax = axes[0][di]
    label = "1 CPU thread" if dev_name == "cpu" else torch.cuda.get_device_name(0)
    ax.set_title(f"per-frame inference on {label}")
    ax.set_xlabel("latency [ms]")
    ax.set_ylabel("frames")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(f"WindowFinder2D single-frame latency ({args.frames} test frames)")
fig.tight_layout()
fig.savefig(out_dir / "finder2d_latency.png", dpi=110)
(out_dir / "latency.json").write_text(json.dumps(results, indent=2))
print("wrote", out_dir / "finder2d_latency.png")

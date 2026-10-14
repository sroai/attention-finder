#!/usr/bin/env python
"""Export a trained run to ONNX, verify CPU parity, benchmark, write deploy info.

Usage (from the repo root):
    uv run python scripts/510_export_hit_model_onnx.py <run_name>

Outputs in runs/<run_name>/: model.onnx, deploy_info.json.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.dataset import load_fold, load_meta
from trkbkg.export import export_onnx, parity_and_benchmark, write_deploy_info
from trkbkg.models import build_model

DATASET_DIR = (CFG.dataset_dir / "18x275_v3")
RUNS_DIR = CFG.runs_dir

run_name = sys.argv[1]
run_dir = RUNS_DIR / run_name

ckpt = torch.load(run_dir / "best.pt", weights_only=False)
meta = load_meta(DATASET_DIR)
model = build_model(ckpt["config"]["model"], meta, **ckpt["config"]["model_kwargs"])
model.load_state_dict(ckpt["model_state"])

onnx_path = export_onnx(model, run_dir / "model.onnx")

# parity + speed on a 200k-hit slice of val (unseen in training)
X, sysi, _, _ = load_fold(DATASET_DIR, "val")
stats = parity_and_benchmark(model, onnx_path, X[:200_000], sysi[:200_000])

# threshold: prefer the evaluated one; fall back to checkpoint's val threshold
metrics_file = run_dir / "metrics.json"
threshold = (json.loads(metrics_file.read_text())["threshold_from_val"]
             if metrics_file.exists() else ckpt["threshold"])

info_path = write_deploy_info(onnx_path, threshold, meta, extra=stats)
print(json.dumps(stats, indent=2))
print(f"wrote {onnx_path} and {info_path}")

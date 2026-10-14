#!/usr/bin/env python
"""Export the phase-4 window model (attention or baseline) to ONNX.

Usage (from the repo root):
    python scripts/512_export_inwindow_model_onnx.py <run>

Graph contract (one WINDOW per call — i.e. the hits inside
[t0_est-50, t0_est+300] of one frame):
    inputs : hits   float32 [N, 4]  (x, y, z [mm], dt [ns], dt = t - t0_est)
             system int64 [N]       dense system index (same map as phase 1)
    output : score  float32 [N]     background probability
Full deployment pipeline = window_finder.onnx (script 13) -> this model for
inside hits + phase-1 model.onnx for outside hits + the two thresholds from
metrics_end_to_end.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.attention import WindowNet
from trkbkg.dataset import load_meta
from trkbkg.export import export_onnx, parity_and_benchmark

HIT_DATA = (CFG.dataset_dir / "18x275_v3")
WHITS = (CFG.dataset_dir / "window_hits_v3")
RUNS = CFG.runs_dir

run_name = sys.argv[1]
run_dir = RUNS / run_name
ckpt = torch.load(run_dir / "best.pt", weights_only=False)
wmeta = ckpt["config"]["window_hits_meta"]
model = WindowNet(wmeta["norm"]["mean"], wmeta["norm"]["std"],
                  len(load_meta(HIT_DATA)["system_ids"]),
                  d=ckpt["config"]["dim"], n_blocks=ckpt["config"]["blocks"])
model.load_state_dict(ckpt["model_state"])

onnx_path = export_onnx(model, run_dir / "window_model.onnx")

va = np.load(WHITS / "val.npz")
stats = parity_and_benchmark(model, onnx_path, va["X"][:100_000],
                             va["sys"][:100_000].astype(np.int64),
                             event_size=1200)  # ~hits per window

e2e_file = run_dir / "metrics_end_to_end.json"
info = {
    "onnx_file": onnx_path.name,
    "inputs": {"hits": "float32 [N,4] (x,y,z mm, dt ns; dt = t - t0_est)",
               "system": "int64 [N] dense index (see phase-1 deploy_info)"},
    "window_ns": [-wmeta["front_ns"], wmeta["back_ns"]],
    "pipeline": ("window_finder.onnx -> t0_est; inside hits -> this model; "
                 "outside hits -> phase-1 model.onnx; thresholds in "
                 "metrics_end_to_end.json"),
    "thresholds": (json.loads(e2e_file.read_text())["thresholds"]
                   if e2e_file.exists() else None),
    **stats,
}
(run_dir / "window_model_deploy_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps(stats, indent=2))
print(f"wrote {onnx_path}")

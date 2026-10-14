#!/usr/bin/env python
"""Export the window finder to ONNX + CPU parity/latency check + deploy info.

Usage (from the repo root):
    uv run python scripts/511_export_window_finder_onnx.py <run_name>

The graph input is the per-frame feature image [1, F, 88] (float32); output
is [1, 88] per-bin logits — argmax bin gives t0_est = -100 + 25*bin ns.
Feature computation (simple counting, documented in deploy_info.json) runs
on the C++ side before the session call.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.windows import (BIN_NS, FEATURE_NAMES, N_BINS, N_FEATURES, T_MIN,
                            WindowFinderCNN)

RUNS = CFG.runs_dir
WIN_DATA = (CFG.dataset_dir / "windows_v4")

run_name = sys.argv[1]
run_dir = RUNS / run_name
ckpt = torch.load(run_dir / "best.pt", weights_only=False)
model = WindowFinderCNN(width=ckpt["config"]["width"]).cpu().eval()
model.load_state_dict(ckpt["model_state"])

onnx_path = run_dir / "window_finder.onnx"
torch.onnx.export(model, (torch.zeros(1, N_FEATURES, N_BINS),),
                  str(onnx_path), input_names=["frame_features"],
                  output_names=["bin_logits"],
                  dynamic_axes={"frame_features": {0: "batch"},
                                "bin_logits": {0: "batch"}},
                  opset_version=17, dynamo=False)

import onnxruntime as ort
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
te = np.load(WIN_DATA / "test.npz")
sample = te["frames"][:256]
with torch.no_grad():
    ref = model(torch.from_numpy(sample)).numpy()
got = sess.run(["bin_logits"], {"frame_features": sample})[0]

one = sample[:1]
for _ in range(10):
    sess.run(["bin_logits"], {"frame_features": one})
t0 = time.perf_counter()
for _ in range(200):
    sess.run(["bin_logits"], {"frame_features": one})
stats = {
    "max_abs_diff": float(np.abs(ref - got).max()),
    "argmax_agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
    "onnx_file_bytes": onnx_path.stat().st_size,
    "cpu_ms_per_frame": round((time.perf_counter() - t0) / 200 * 1e3, 4),
}

info = {
    "onnx_file": onnx_path.name,
    "input": f"frame_features float32 [1, {N_FEATURES}, {N_BINS}]",
    "output": f"bin_logits float32 [1, {N_BINS}]; "
              f"t0_est_ns = {T_MIN} + {BIN_NS} * argmax(bin_logits)",
    "bin_ns": BIN_NS, "t_min_ns": T_MIN,
    "features_per_bin (hits with t in [edge, edge+25ns))": {
        str(i): name for i, name in enumerate(FEATURE_NAMES)},
    "feature_definitions": {
        "log1p_n_hits": "log(1 + count of all hits in bin)",
        "log1p_n_outer": "log1p count with sqrt(x^2+y^2)>550 && |z|<1600 [mm]",
        "log1p_n_farfwd": "log1p count with z>4000",
        "log1p_n_backward": "log1p count with z<-1200",
        "log1p_n_vertex": "log1p count with r<150",
        "r_std": "stddev of r in bin / 100",
        "z_std": "stddev of z in bin / 1000",
        "occupancy_contrast": "bin count / (frame hits / 88)",
    },
    **stats,
}
(run_dir / "window_deploy_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps(stats, indent=2))
print(f"wrote {onnx_path} and window_deploy_info.json")

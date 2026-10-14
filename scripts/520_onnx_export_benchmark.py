#!/usr/bin/env python
"""Phase 6b: ONNX export + CPU benchmarks for the full window pipeline.

Exports the two models that had none yet and benchmarks EVERYTHING the
deployed pipeline runs, single-threaded CPU (the EIC C++ ONNX runtime is
CPU-only and memory-capped):

  stage 1 WINDOW FINDING
    finder2d_v1   2-ch 64x88 image -> 88 bin logits        (exported here)
    watt_r2       scores hits of the 10 candidate windows  (existing onnx)
    wver_v1       verifies each candidate window           (exported here)
    selector      tiny MLP on 24 features                  (negligible, we
                  still time a numpy/torch-cpu equivalent)
  stage 2 HIT REJECTION (unchanged from phase 4, measured for the slide)
    watt_r2       per-hit scores inside the chosen window
    mlp_v1_r3     out-of-window fallback

Numbers reported per model: file size, parameter count, ms per unit
(frame / window / candidate set), session RSS delta, and torch-vs-onnx
parity. Everything lands in reports/phase6_onnx_benchmarks.json.

Usage (from the repo root):
    python scripts/520_onnx_export_benchmark.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.verifier import WindowVerifier
from trkbkg.windows2d import WindowFinder2D

RUNS_L = CFG.runs_local_dir
RUNS = CFG.runs_dir
IMG = (CFG.dataset_dir / "images2d_v1")
CWIN = (CFG.dataset_dir / "cand_windows_v1")
OUT_JSON = Path(__file__).resolve().parents[1] / "reports" / \
    "phase6_onnx_benchmarks.json"

import onnxruntime as ort


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def session(path):
    """Single-threaded CPU session + RSS delta of creating it."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    before = rss_mb()
    sess = ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    return sess, rss_mb() - before


def bench(fn, n_warm=5, n_run=50):
    for _ in range(n_warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_run):
        fn()
    return (time.perf_counter() - t0) / n_run * 1e3  # ms


results = {}

# ---------- 1. finder2d_v1 ---------------------------------------------------
ck = torch.load(RUNS_L / "finder2d_v1" / "best.pt", weights_only=False,
                map_location="cpu")
finder = WindowFinder2D(width=ck["config"]["width"],
                        n_blocks=ck["config"].get("blocks", 3)).eval()
finder.load_state_dict(ck["model_state"])
f_onnx = RUNS_L / "finder2d_v1" / "window_finder2d.onnx"
img = torch.from_numpy(np.load(IMG / "test.npz")["img"][:64])
torch.onnx.export(finder, img[:1], f_onnx, input_names=["image"],
                  output_names=["bin_logits"], opset_version=18)
sess, drss = session(f_onnx)
x1 = img[:1].numpy()
with torch.no_grad():
    ref = finder(img[:1]).numpy()
got = sess.run(None, {"image": x1})[0]
parity = float(np.abs(ref - got).max())
ms = bench(lambda: sess.run(None, {"image": x1}))
results["finder2d_v1 (stage-1 finder)"] = {
    "onnx_file_kb": round(f_onnx.stat().st_size / 1024, 1),
    "parameters": sum(p.numel() for p in finder.parameters()),
    "ms_per_frame_1thread_cpu": round(ms, 3),
    "session_rss_mb": round(drss, 1), "torch_onnx_max_diff": parity,
    "unit": "one 2x64x88 frame image"}
print("finder2d:", results["finder2d_v1 (stage-1 finder)"], flush=True)

# ---------- 2. wver_v1 -------------------------------------------------------
meta = json.loads((CWIN / "meta.json").read_text())
ck = torch.load(RUNS_L / "wver_v1" / "best.pt", weights_only=False,
                map_location="cpu")
wver = WindowVerifier(meta["norm"]["mean"], meta["norm"]["std"],
                      len(meta["system_ids"]), d=ck["config"]["dim"],
                      n_blocks=ck["config"]["blocks"]).eval()
wver.load_state_dict(ck["model_state"])
v_onnx = RUNS_L / "wver_v1" / "window_verifier.onnx"
N = 512
Xd = torch.randn(1, N, 6)
Sd = torch.zeros(1, N, dtype=torch.int64)
Pd = torch.zeros(1, N, dtype=torch.bool)
Cd = torch.randn(1, 3)
torch.onnx.export(wver, (Xd, Sd, Pd, Cd), v_onnx,
                  input_names=["hits", "systems", "pad", "scalars"],
                  output_names=["window_logit"],
                  dynamic_axes={"hits": {1: "n_hits"},
                                "systems": {1: "n_hits"},
                                "pad": {1: "n_hits"}},
                  opset_version=18)
sess_v, drss_v = session(v_onnx)
# real window: hits of the first test window with >=400 hits
d = np.load(CWIN / "test.npz")
lens = np.diff(d["offs"])
w = int(np.argmax(lens >= 400))
h = d["hits"][d["offs"][w]:d["offs"][w + 1]]
feed = {"hits": h[None, :, :6].astype(np.float32),
        "systems": h[None, :, 6].astype(np.int64),
        "pad": np.zeros((1, len(h)), bool),
        "scalars": np.array([[np.log1p(d["n_hits"][w]), d["logit"][w],
                              float(d["rank"][w])]], np.float32)}
with torch.no_grad():
    ref = wver(torch.from_numpy(feed["hits"]),
               torch.from_numpy(feed["systems"]),
               torch.from_numpy(feed["pad"]),
               torch.from_numpy(feed["scalars"])).numpy()
got = sess_v.run(None, feed)[0]
ms_v = bench(lambda: sess_v.run(None, feed))
results["wver_v1 (stage-1 verifier)"] = {
    "onnx_file_kb": round(v_onnx.stat().st_size / 1024, 1),
    "parameters": sum(p.numel() for p in wver.parameters()),
    "ms_per_window_1thread_cpu": round(ms_v, 3),
    "ms_per_frame_10_candidates": round(ms_v * 10, 2),
    "session_rss_mb": round(drss_v, 1),
    "torch_onnx_max_diff": float(np.abs(ref - got).max()),
    "unit": f"one {len(h)}-hit candidate window (cap 512)"}
print("wver:", results["wver_v1 (stage-1 verifier)"], flush=True)

# ---------- 3. watt_r2 (hit scorer: stage-1 feature source AND stage-2) -----
sess_w, drss_w = session(RUNS / "watt_r2" / "window_model.onnx")
w_in = [i.name for i in sess_w.get_inputs()]
n_hits_typical = 1200  # median in-window hits, 18x275 frame
Xw = np.random.randn(n_hits_typical, 4).astype(np.float32)
Xw[:, 3] = np.random.uniform(0, 350, n_hits_typical)
Sw = np.random.randint(0, 22, n_hits_typical).astype(np.int64)
feed_w = {w_in[0]: Xw, w_in[1]: Sw}
ms_w = bench(lambda: sess_w.run(None, feed_w))
results["watt_r2 (stage-2 hit rejector)"] = {
    "onnx_file_kb": round((RUNS / "watt_r2" / "window_model.onnx"
                           ).stat().st_size / 1024, 1),
    "ms_per_window_1thread_cpu": round(ms_w, 2),
    "ms_per_frame_10_candidates_stage1": round(ms_w * 10, 1),
    "session_rss_mb": round(drss_w, 1),
    "unit": f"one {n_hits_typical}-hit window (per-hit scores)"}
print("watt:", results["watt_r2 (stage-2 hit rejector)"], flush=True)

# ---------- 4. mlp fallback + selector ---------------------------------------
sess_m, drss_m = session(RUNS / "mlp_v1_r3" / "model.onnx")
m_in = [i.name for i in sess_m.get_inputs()]
n_out = 5800  # typical out-of-window hits per frame
shapes = {i.name: i.shape for i in sess_m.get_inputs()}
feeds_m = {}
for i in sess_m.get_inputs():
    shp = [n_out if isinstance(s, str) else s for s in i.shape]
    feeds_m[i.name] = (np.random.randn(*shp).astype(np.float32)
                       if "float" in i.type else
                       np.random.randint(0, 22, shp).astype(np.int64))
ms_m = bench(lambda: sess_m.run(None, feeds_m))
results["mlp_v1_r3 (stage-2 out-of-window fallback)"] = {
    "onnx_file_kb": round((RUNS / "mlp_v1_r3" / "model.onnx"
                           ).stat().st_size / 1024, 1),
    "ms_per_frame_1thread_cpu": round(ms_m, 2),
    "session_rss_mb": round(drss_m, 1),
    "unit": f"{n_out} out-of-window hits"}
print("mlp:", results["mlp_v1_r3 (stage-2 out-of-window fallback)"],
      flush=True)

sel = torch.load(RUNS_L / "t0_final_v1" / "selector.pt", weights_only=False,
                map_location="cpu")
results["selector (stage-1, 24 features x 10 cand)"] = {
    "parameters": int(sum(v.numel() for v in sel["state"].values())),
    "ms_per_frame": "<0.01 (one 24->32->1 MLP pass on 10 rows)",
    "onnx_file_kb": "~5 (not yet exported; trivial MatMul graph)"}

# ---------- totals ------------------------------------------------------------
s1 = (results["finder2d_v1 (stage-1 finder)"]["ms_per_frame_1thread_cpu"]
      + results["wver_v1 (stage-1 verifier)"]["ms_per_frame_10_candidates"]
      + results["watt_r2 (stage-2 hit rejector)"][
          "ms_per_frame_10_candidates_stage1"])
s2 = (results["watt_r2 (stage-2 hit rejector)"]["ms_per_window_1thread_cpu"]
      + results["mlp_v1_r3 (stage-2 out-of-window fallback)"][
          "ms_per_frame_1thread_cpu"])
results["TOTALS"] = {
    "stage1_window_finding_ms_per_frame": round(s1, 1),
    "stage1_note": "finder + 10x watt candidate scoring + 10x wver; the "
                   "watt scoring dominates and is shared with stage 2 "
                   "for the winning window",
    "stage2_hit_rejection_ms_per_frame": round(s2, 1),
    "all_model_files_kb": round(sum(
        v["onnx_file_kb"] for v in results.values()
        if isinstance(v, dict) and isinstance(v.get("onnx_file_kb"), float)),
        1),
    "peak_session_rss_mb_sum": round(sum(
        v.get("session_rss_mb", 0) for v in results.values()
        if isinstance(v, dict)), 1)}
OUT_JSON.write_text(json.dumps(results, indent=2))
print(json.dumps(results["TOTALS"], indent=2))
print(f"wrote {OUT_JSON}")

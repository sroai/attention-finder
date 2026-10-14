"""ONNX export + CPU parity / resource check.

The exported graph takes RAW detector values (normalization is baked inside
the model) and outputs a background score in [0,1] per hit:

    inputs : hits    float32 [N, 4]   (x [mm], y [mm], z [mm], t [ns])
             system  int64   [N]      dense system index (see meta.json map)
    output : score   float32 [N]      sigmoid(logit); >= threshold -> filter

N is dynamic, so C++ can score a whole event in one call. The operating
threshold from evaluation is written next to the model in deploy_info.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class _WithSigmoid(nn.Module):
    """Wraps a logit model so the ONNX output is a ready-to-use probability."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hits, system):
        return torch.sigmoid(self.model(hits, system))


def export_onnx(model: nn.Module, onnx_path: str | Path) -> Path:
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = _WithSigmoid(model.cpu().eval())
    dummy = (torch.zeros(3, 4), torch.zeros(3, dtype=torch.int64))
    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        input_names=["hits", "system"], output_names=["score"],
        dynamic_axes={"hits": {0: "n_hits"}, "system": {0: "n_hits"},
                      "score": {0: "n_hits"}},
        opset_version=17, dynamo=False,
    )
    return onnx_path


def parity_and_benchmark(model: nn.Module, onnx_path, X: np.ndarray,
                         sys_idx: np.ndarray, event_size: int = 8000) -> dict:
    """Compare ONNX-CPU scores against pytorch and time a realistic event.

    event_size ~ hits per frame in real running (~7000 from data analysis).
    """
    import onnxruntime as ort

    model = model.cpu().eval()
    with torch.no_grad():
        ref = torch.sigmoid(model(torch.from_numpy(X),
                                  torch.from_numpy(sys_idx))).numpy()

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    got = sess.run(["score"], {"hits": X, "system": sys_idx})[0]

    # latency for one event-sized call on CPU
    Xe, se = X[:event_size], sys_idx[:event_size]
    for _ in range(3):  # warmup
        sess.run(["score"], {"hits": Xe, "system": se})
    t0 = time.perf_counter()
    n_rep = 20
    for _ in range(n_rep):
        sess.run(["score"], {"hits": Xe, "system": se})
    ms_per_event = (time.perf_counter() - t0) / n_rep * 1e3

    return {
        "max_abs_diff": float(np.abs(ref - got).max()),
        "onnx_file_bytes": Path(onnx_path).stat().st_size,
        "cpu_ms_per_event": round(ms_per_event, 3),
        "event_size_hits": event_size,
        "parity_hits_checked": len(X),
    }


def write_deploy_info(onnx_path, threshold: float, meta: dict, extra: dict):
    """Everything the C++ side needs, next to the .onnx file."""
    info = {
        "onnx_file": Path(onnx_path).name,
        "score_threshold": threshold,
        "decision": "score >= threshold means background -> filter out",
        "inputs": {
            "hits": "float32 [N,4] raw (x mm, y mm, z mm, t ns)",
            "system": ("int64 [N], dense index; map raw system_id via "
                       "system_ids list below (position = index)"),
        },
        "system_ids": meta["system_ids"],
        **extra,
    }
    p = Path(onnx_path).with_name("deploy_info.json")
    p.write_text(json.dumps(info, indent=2))
    return p

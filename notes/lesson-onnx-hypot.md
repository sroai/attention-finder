torch.hypot has no ONNX mapping (opset 17, torchscript exporter) — use sqrt(x*x+y*y) in any model meant for export.

Hit this exporting HitMLP: training worked, export failed with
`UnsupportedOperatorError: aten::hypot`. The two forms are numerically
equivalent at detector scales (mm, values < 1e4), so swapping in
sqrt(x^2+y^2) needs no retraining. Rule of thumb adopted: keep model
forward() to plain elementwise ops + Linear/Embedding, and smoke-test
`06_export_onnx.py` right after the first smoke train, not at the end.

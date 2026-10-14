#!/usr/bin/env python
"""Compare all evaluated runs side by side.

Usage (from the repo root):
    uv run python scripts/411_compare_hit_runs.py

Reads every runs/<name>/metrics.json (created by 410_evaluate_hit_classifier.py) and prints a
ranking table on the headline metric: test background rejection at the global
threshold calibrated for the target signal efficiency. Also writes the table
to reports/run_comparison.md so reports can link a stable file.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

RUNS_DIR = CFG.runs_dir
OUT = Path(__file__).resolve().parents[1] / "reports" / "run_comparison.md"

rows = []
for mfile in sorted(RUNS_DIR.glob("*/metrics.json")):
    m = json.loads(mfile.read_text())
    cfg = json.loads((mfile.parent / "config.json").read_text())
    if "test" not in m or "model_kwargs" not in cfg:
        continue  # window-finder/window-model runs use different schemas
    rows.append({
        "run": m["run"],
        "model": f'{cfg["model"]} {cfg["model_kwargs"] or ""}'.strip(),
        "epochs": cfg["epochs"],
        "test_auc": m["test"]["auc"],
        "test_eff": m["test"]["signal_efficiency"],
        "test_rej": m["test"]["background_rejection"],
        "persys_eff": m.get("test_per_system_thresholds_mode", {}).get("signal_efficiency"),
        "persys_rej": m.get("test_per_system_thresholds_mode", {}).get("background_rejection"),
    })

if not rows:
    sys.exit("no evaluated runs found (run 410_evaluate_hit_classifier.py first)")

rows.sort(key=lambda r: r["test_rej"], reverse=True)

header = ("| run | model | epochs | test AUC | eff@global | rej@global "
          "| eff@per-sys | rej@per-sys |")
sep = "|" + "---|" * 8
lines = [header, sep]
for r in rows:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    lines.append("| " + " | ".join(fmt(r[k]) for k in
                 ("run", "model", "epochs", "test_auc", "test_eff",
                  "test_rej", "persys_eff", "persys_rej")) + " |")

table = "\n".join(lines)
OUT.write_text("# Run comparison (test fold)\n\n"
               "Sorted by background rejection at the global threshold "
               "(val-calibrated for target signal efficiency).\n\n"
               + table + "\n")
print(table)
print(f"\nwrote {OUT}")

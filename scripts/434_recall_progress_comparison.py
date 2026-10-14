#!/usr/bin/env python
"""Phase 5 final comparison: end-to-end curves, argmax finder vs verifier.

Usage (from the repo root):
    python scripts/434_recall_progress_comparison.py <win_run>

Reads runs/<win_run>/metrics_end_to_end_r3base.json (argmax finder) and
_r3refined.json (candidate verifier t0) and overlays their test curves with
the perfect-finder bound. Output: plots/phase5/e2e_comparison.png.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

RUNS = CFG.runs_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase5"

win_run = sys.argv[1]
base = json.loads((RUNS / win_run / "metrics_end_to_end_r3base.json").read_text())
ref = json.loads((RUNS / win_run / "metrics_end_to_end_r3refined.json").read_text())

fig, ax = plt.subplots(figsize=(8, 5.2))
for m, color, lab in [
        (base, "#999999", "phase 3/4: argmax finder"),
        (ref, "#1a7f37", "phase 5: top-5 + verifier t0")]:
    curve = m["test_curve"]
    ax.plot([c["signal_efficiency"] for c in curve],
            [c["background_rejection"] for c in curve], "o-",
            color=color, label=lab)
    at99 = m["test"]
    ax.annotate(f"{at99['background_rejection']:.1%}",
                (at99["signal_efficiency"], at99["background_rejection"]),
                textcoords="offset points", xytext=(6, 4), fontsize=9,
                color=color)
# perfect-finder bound if the decomposition was run on either file
for m in (ref, base):
    d = m.get("decomposition", {}).get("finder_true")
    if d:
        ax.plot(d["signal_efficiency"], d["background_rejection"], "*",
                ms=16, color="#c0392b", label="perfect-finder bound")
        break
ax.axvline(0.99, color="k", ls="--", lw=1, label="99% eff target")
ax.set_xlabel("END-TO-END signal efficiency (reconstructable signal)")
ax.set_ylabel("background rejection")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
ax.set_title(f"phase 5 vs phase 3/4 ({win_run} scoring, v3 data, test fold)")
fig.tight_layout()
fig.savefig(PLOTS / "e2e_comparison.png", dpi=130)
print(f"wrote {PLOTS / 'e2e_comparison.png'}")
print("base  test:", json.dumps(base["test"]))
print("refin test:", json.dumps(ref["test"]))

#!/usr/bin/env python
"""Phase 5 step 1: is the true burst VISIBLE to the finder but out-ranked?

Usage (from the repo root):
    python scripts/424_topk_diagnostic_1d.py [wf_run]

For every test frame, rank the finder's 88 bin logits and ask at which rank
the true-t0 bin (±1 bin tolerance = ±25 ns) appears. If missed frames mostly
have the true bin in the top-3/5, a candidate+verifier design (script 23)
will recover them; if the true bin ranks deep, the per-bin features are
blind and feature work comes first.

Outputs: plots/phase5/topk_coverage.png, reports/phase5_stats.json (merged).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.windows import BIN_NS, T_MIN, WindowFinderCNN

WIN_DATA = (CFG.dataset_dir / "windows_v4")
RUNS = CFG.runs_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "phase5"
PLOTS.mkdir(parents=True, exist_ok=True)

wf_run = sys.argv[1] if len(sys.argv) > 1 else "wf_v2_r3"
ckpt = torch.load(RUNS / wf_run / "best.pt", weights_only=False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
wf = WindowFinderCNN(width=ckpt["config"]["width"]).to(dev).eval()
wf.load_state_dict(ckpt["model_state"])

te = np.load(WIN_DATA / "test.npz")
with torch.no_grad():
    logits = wf(torch.from_numpy(te["frames"]).to(dev)).cpu().numpy()

order = np.argsort(-logits, axis=1)          # bins by descending logit
true_bin = te["target"]
# rank of the best bin within ±1 of the true bin (25 ns tolerance)
hit = np.abs(order - true_bin[:, None]) <= 1  # [frames, 88] rank positions
rank = np.argmax(hit, axis=1)                 # first rank where within ±1

t0_est = order[:, 0] * BIN_NS + T_MIN
missed = np.abs(t0_est - te["t0"]) > 50

ks = np.arange(1, 11)
cov_all = [(rank < k).mean() for k in ks]
cov_missed = [(rank[missed] < k).mean() for k in ks]

res = {
    "wf_run": wf_run, "frames": int(len(rank)),
    "missed_frames": int(missed.sum()),
    "coverage_all": {int(k): float(c) for k, c in zip(ks, cov_all)},
    "coverage_missed_frames": {int(k): float(c) for k, c in
                               zip(ks, cov_missed)},
    "median_rank_missed": float(np.median(rank[missed])),
}

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(ks, cov_all, "o-", color="#2c5aa0", label="all test frames")
ax.plot(ks, cov_missed, "s-", color="#c0392b",
        label="finder-MISSED frames only")
ax.axhline(1.0, color="k", lw=0.5)
ax.set_xlabel("k (top-k finder bins considered)")
ax.set_ylabel("frames with true bin (±25 ns) in top-k")
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
ax.legend()
ax.set_title(f"{wf_run}: is the burst visible but out-ranked? "
             f"(missed frames: top-3 = {cov_missed[2]:.0%})")
fig.tight_layout()
fig.savefig(PLOTS / "topk_coverage.png", dpi=130)

stats_file = Path(__file__).resolve().parents[1] / "reports" / "phase5_stats.json"
merged = json.loads(stats_file.read_text()) if stats_file.exists() else {}
merged["topk_diagnostic"] = res
stats_file.write_text(json.dumps(merged, indent=2))
print(json.dumps(res, indent=2))

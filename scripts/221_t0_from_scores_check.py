#!/usr/bin/env python
"""Feasibility check for a two-stage phase-3 model: recover t0 from MODEL
scores instead of raw occupancy.

Raw occupancy peaks fail to find the signal burst (median error ~470 ns,
see 08) because background ALSO arrives in bursts (beam-gas interactions).
Here every hit is weighted by its signal-likeness w = 1 - score(mlp_v1)
before the time histogram is built. If the weighted peak finds t0, then
"stage 1: locate burst with per-hit scores -> stage 2: classify with
dt = t - t0_est" is a viable architecture.

Usage (from the repo root):
    uv run python scripts/221_t0_from_scores_check.py

Appends results to reports/temporal_stats.json and writes
plots/temporal/t0_recovery_scored.png.
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
from trkbkg import temporal as T
from trkbkg.dataset import load_meta
from trkbkg.models import build_model

ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = (CFG.parquet_dir / "18x275_v3")
DATASET_DIR = (CFG.dataset_dir / "18x275_v3")
RUN_DIR = (CFG.runs_dir / "mlp_v1")
PLOTS = ROOT / "plots" / "temporal"

meta = load_meta(DATASET_DIR)
ckpt = torch.load(RUN_DIR / "best.pt", weights_only=False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(ckpt["config"]["model"], meta,
                    **ckpt["config"]["model_kwargs"]).to(dev).eval()
sys_ids = np.array(meta["system_ids"])

BIN = 25.0
edges = np.arange(-100, 2100 + BIN, BIN)
errs_scored, errs_raw = [], []

for tag in ("0022", "0023"):
    d = T.load_events(PARQUET_DIR / f"{tag}.trk_hits.parquet")
    stats = T.per_event_signal_stats(d)

    # model wants raw (x, y, z, t); load_events only kept r, so re-read x,y
    import polars as pl
    df = pl.scan_parquet(PARQUET_DIR / f"{tag}.trk_hits.parquet").select(
        ["trk_hit_pos_x", "trk_hit_pos_y"]).collect()
    X = np.stack([df["trk_hit_pos_x"].to_numpy(),
                  df["trk_hit_pos_y"].to_numpy(), d["z"], d["t"]],
                 axis=1).astype(np.float32)
    sysi = np.searchsorted(sys_ids, d["sys"]).astype(np.int64)

    scores = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 2_000_000):
            scores[i:i + 2_000_000] = torch.sigmoid(model(
                torch.from_numpy(X[i:i + 2_000_000]).to(dev),
                torch.from_numpy(sysi[i:i + 2_000_000]).to(dev))).cpu().numpy()
    w = 1.0 - scores  # signal-likeness

    for i, (s, e) in enumerate(zip(d["starts"], d["ends"])):
        if np.isnan(stats["t0"][i]):
            continue
        t = d["t"][s:e]
        h_w, _ = np.histogram(t, bins=edges, weights=w[s:e])
        h_r, _ = np.histogram(t, bins=edges)
        errs_scored.append(edges[np.argmax(h_w)] - stats["t0"][i])
        errs_raw.append(edges[np.argmax(h_r)] - stats["t0"][i])

errs_scored = np.array(errs_scored)
errs_raw = np.array(errs_raw)

res = {}
for name, errs in [("raw_occupancy", errs_raw), ("score_weighted", errs_scored)]:
    res[name] = {
        "median_abs_err_ns": float(np.median(np.abs(errs))),
        "frac_within_50ns": float((np.abs(errs) <= 50).mean()),
        "frac_within_100ns": float((np.abs(errs) <= 100).mean()),
    }

stats_file = ROOT / "reports" / "temporal_stats.json"
summary = json.loads(stats_file.read_text())
summary["t0_recovery_score_weighted"] = res
stats_file.write_text(json.dumps(summary, indent=2))
print(json.dumps(res, indent=2))

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for errs, label, color in [(errs_raw, "raw occupancy peak", "#777777"),
                           (errs_scored, "score-weighted peak", "#1a7f37")]:
    ax.hist(np.clip(errs, -500, 500), bins=100, histtype="step", lw=1.6,
            label=label, color=color)
ax.set_xlabel("t0 estimate - true t0 [ns] (25 ns bins, clipped)")
ax.set_ylabel("events")
ax.set_yscale("log")
ax.legend()
ax.set_title("t0 recovery: weighting hits by mlp_v1 signal-likeness")
fig.tight_layout()
fig.savefig(PLOTS / "t0_recovery_scored.png", dpi=130)
print(f"wrote {PLOTS / 't0_recovery_scored.png'}")

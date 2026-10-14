#!/usr/bin/env python
"""Build the event-split train/val/test dataset from parquet.

Usage (from the repo root):
    uv run python scripts/120_build_hit_dataset.py

Writes <dataset_dir>/18x275_v3/{train,val,test}.npz + meta.json.
Skips if meta.json already exists (delete the dir to rebuild).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.dataset import build_dataset

PARQUET_DIR = (CFG.parquet_dir / "18x275_v3")
OUT_DIR = (CFG.dataset_dir / "18x275_v3")

if (OUT_DIR / "meta.json").exists():
    print(f"[skip] {OUT_DIR} already built")
else:
    meta = build_dataset(
        {"0022": PARQUET_DIR / "0022.trk_hits.parquet",
         "0023": PARQUET_DIR / "0023.trk_hits.parquet"},
        OUT_DIR,
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "norm"}, indent=2))

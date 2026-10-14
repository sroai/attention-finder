#!/usr/bin/env python
"""Data-understanding stats + plots for each converted parquet file.

Usage (from the repo root):
    uv run python scripts/210_explore_data.py

Outputs:
    plots/data/<tag>_*.png     comparison plots (signal vs background)
    reports/data_stats_<tag>.json   headline numbers
and prints a short summary to stdout.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.analyze import run_analysis

ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = (CFG.parquet_dir / "18x275_v3")
FILES = ["0022.trk_hits.parquet", "0023.trk_hits.parquet"]


def main() -> None:
    for fname in FILES:
        tag = fname.split(".")[0]  # "0022"
        stats = run_analysis(
            PARQUET_DIR / fname,
            plots_dir=ROOT / "plots" / "data",
            stats_path=ROOT / "reports" / f"data_stats_{tag}.json",
            tag=tag,
        )
        print(f"=== {tag} ===")
        print(json.dumps({k: v for k, v in stats.items() if k != "per_system"},
                         indent=2))
        print("per_system:")
        for sid, s in stats["per_system"].items():
            print(f"  system {sid:>3}: {s['hits']:>10,} hits, "
                  f"signal {100*s['signal_fraction']:6.2f}%")


if __name__ == "__main__":
    main()

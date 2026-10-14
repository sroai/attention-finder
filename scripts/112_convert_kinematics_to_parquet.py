#!/usr/bin/env python
"""STEP 112 — convert reco_dis CSVs to a slim per-event kinematics parquet.

Kinematics companion of steps 110/111: <csv_root>/<sample>/*.reco_dis.csv[.zip]
becomes <parquet_dir>/<sample>_kin_v1/<stem>.reco_dis.parquet with one row per
event. The truth_* block (EICrecon InclusiveKinematicsTruth) is the only true
Q²/x source in background-merged files — the generator's dis_* frame params
are dropped by the merge, so mc_* columns are empty there. electron_* keeps
the reconstructed-electron method for truth-vs-reco comparisons.

Usage (see docs/running-on-a-farm.md for the farm profile):
    python scripts/112_convert_kinematics_to_parquet.py 10x100 \
        [--csv-dir DIR] [--files stem1 ...]
"""

import argparse
import sys
import zipfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

COLUMNS = {
    "evt": pl.Int64,
    "truth_x": pl.Float32,
    "truth_q2": pl.Float32,     # [GeV^2], from MC via InclusiveKinematicsTruth
    "truth_y": pl.Float32,
    "truth_nu": pl.Float32,
    "truth_w": pl.Float32,
    "electron_x": pl.Float32,
    "electron_q2": pl.Float32,  # [GeV^2], reconstructed electron method
}

parser = argparse.ArgumentParser()
parser.add_argument("sample", help="csv dir name, e.g. 10x100 or 10x100_minQ2_1")
parser.add_argument("--csv-dir", type=Path, default=None,
                    help="override <csv_root>/<sample>")
parser.add_argument("--files", nargs="*", default=None)
args = parser.parse_args()

csv_dir = args.csv_dir or (CFG.csv_root / args.sample)
out_dir = CFG.parquet_dir / f"{args.sample}_kin_v1"
out_dir.mkdir(parents=True, exist_ok=True)

paths = sorted(csv_dir.glob("*.reco_dis.csv*"))
if args.files:
    paths = [p for p in paths if any(f in p.name for f in args.files)]
if not paths:
    raise SystemExit(f"no *.reco_dis.csv[.zip] in {csv_dir}")

for path in paths:
    stem = path.name.split(".reco_dis")[0]
    out = out_dir / f"{stem}.reco_dis.parquet"
    if out.exists():
        print(f"[skip] {out.name}")
        continue
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            raw = zf.read(zf.namelist()[0])
    else:
        raw = path.read_bytes()
    # polars 1.44 ignores schema_overrides on in-memory bytes: read all-string,
    # cast in the select (empty strings -> null via strict=False).
    df = pl.read_csv(raw, infer_schema=False)
    df = df.select([pl.col(c).cast(t, strict=False).alias(c)
                    for c, t in COLUMNS.items()])
    df.write_parquet(out)
    n_truth = df["truth_q2"].is_not_null().sum()
    print(f"{out.name}: {len(df)} events, truth_q2 filled {n_truth}")

print(f"done -> {out_dir}")

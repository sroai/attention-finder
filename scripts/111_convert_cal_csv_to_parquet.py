#!/usr/bin/env python
"""STEP 111 — convert cal_hits CSVs (digitized calorimeter rec hits) to parquet.

Calorimeter companion of step 110: <csv_root>/<sample>/*.cal_hits.csv[.zip]
becomes <parquet_dir>/<sample>_cal_v1/<stem>.cal_hits.parquet with the slim
column set the window-finder image builder (132) and analysis need. No
relabeling happens here: prt_origin is stored as the converter wrote it
(calorimeter albedo rules are an open phase-10+ study; trk_hit_bkg.md).

CSVs made before 2026-09-30 carry two macro-era names (LEGACY_COLUMNS); the
eic2ai files use the unified vocabulary directly.

Usage (see docs/running-on-a-farm.md for the farm profile):
    python scripts/111_convert_cal_csv_to_parquet.py 10x130_p10 \
        [--csv-dir DIR] [--files stem1 ...]

Without --csv-dir the sample's CSVs are expected in <csv_root>/<sample>/.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

# Slim schema: image channels need pos/time/energy; origin + purity make the
# per-hit labels usable for analysis without rereading the CSV.
COLUMNS = {
    "evt": pl.Int64,
    "cal_hit_system_id": pl.Int32,
    "cal_hit_pos_x": pl.Float32,
    "cal_hit_pos_y": pl.Float32,
    "cal_hit_pos_z": pl.Float32,
    "cal_hit_energy": pl.Float32,   # [GeV], calibrated rec-hit energy
    "cal_hit_time": pl.Float32,     # [ns], digitized (earliest deposit in the cell)
    "sim_time": pl.Float32,         # [ns], earliest truth contribution of the cell
    "prt_origin": pl.Int32,         # energy-majority origin of the cell
    "prt_gen_status": pl.Int32,
    "label_purity": pl.Float32,     # energy fraction of the majority origin class
}

# CSVs converted before 2026-09-30 carry the macro-era names
# (vocabulary unified in simulation-pipeline/eic2ai/column_renames.py).
LEGACY_COLUMNS = {"prt_status": "prt_gen_status", "rec_collection": "cal_hit_collection"}


def convert_one(csv_path: Path, parquet_path: Path) -> None:
    with open(csv_path, "rb") as f:
        magic = f.read(2)
    # Everything reads as strings; the select below casts. Inference
    # mis-types columns (label_purity i64 when early rows are all 1) and
    # schema_overrides did not stick on in-memory sources (polars 1.44).
    if magic == b"PK":  # one-entry zip, the pipeline's container contract
        with zipfile.ZipFile(csv_path) as z:
            with z.open(z.namelist()[0]) as member:
                df = pl.read_csv(member.read(), infer_schema=False)
    else:
        df = pl.read_csv(csv_path, infer_schema=False)
    df = df.rename({old: new for old, new in LEGACY_COLUMNS.items() if old in df.columns})
    df = df.select([pl.col(name).cast(dtype) for name, dtype in COLUMNS.items()])
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path, compression="zstd")
    print(f"  {csv_path.name}: {len(df)} hits, {df['evt'].n_unique()} events "
          f"-> {parquet_path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", help="sample name, e.g. 10x130_p10; parquet goes to "
                                       "<parquet_dir>/<sample>_cal_v1/")
    parser.add_argument("--csv-dir", type=Path, default=None,
                        help="directory holding *.cal_hits.csv[.zip] "
                             "(default: <csv_root>/<sample>/)")
    parser.add_argument("--files", nargs="+", default=None,
                        help="specific stems, e.g. k_lambda_10x130_5000evt_0001 (default: all)")
    args = parser.parse_args()

    csv_dir = args.csv_dir or (CFG.csv_root / args.sample)
    out_dir = CFG.parquet_dir / f"{args.sample}_cal_v1"
    stems = sorted({p.name.split(".cal_hits")[0] for p in csv_dir.glob("*.cal_hits.csv*")})
    if args.files:
        stems = [s for s in stems if s in set(args.files)]
    if not stems:
        raise SystemExit(f"no *.cal_hits.csv[.zip] in {csv_dir}")
    print(f"{len(stems)} file(s) from {csv_dir}\n -> {out_dir}")
    for stem in stems:
        out = out_dir / f"{stem}.cal_hits.parquet"
        if out.exists():
            print(f"[skip] {out.name} exists")
            continue
        src = next(p for suffix in (".cal_hits.csv.zip", ".cal_hits.csv")
                   for p in [csv_dir / f"{stem}{suffix}"] if p.exists())
        convert_one(src, out)


if __name__ == "__main__":
    main()

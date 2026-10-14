#!/usr/bin/env python
"""STEP 110 — convert raw trk_hits CSVs into slim, fast parquet files.

WHAT GOES IN  : <csv_root>/<sample>/xxxx.trk_hits.csv[.zip]     (~11 GB each)
WHAT COMES OUT: <parquet_dir>/<sample>_v3/xxxx.trk_hits.parquet (~0.8 GB)
(both roots come from your config profile — see configs/local.yaml)

Why: the raw CSVs have 32 columns and take minutes to scan; training and
analysis need 14 of them. Parquet is compressed and column-oriented, so a
slim file is ~30x smaller and a single column loads in seconds.

This is also where the ALBEDO relabeling happens (that is why the output
dir is versioned "_v3"): origin-1/2 hits that arrive far too late to be
reconstructable are flipped to prt_origin=5 ("signal albedo" — treated as
don't-care downstream). The two physics thresholds live in trkbkg/labels.py;
the full story is reports/06_event27_anatomy.md.

Usage (from the repo root):
    python scripts/110_convert_csv_to_parquet.py 18x275
    ... 110_convert_csv_to_parquet.py 10x130 --files 0001.trk_hits
    ... 110_convert_csv_to_parquet.py /data/background_dvmp-10x275-25-10

The sample argument is a folder name under csv_root, or an absolute path.
Without --files, every *.trk_hits.csv[.zip] in the folder is converted.
Already-existing outputs are skipped, so re-running is always safe.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.labels import ALBEDO_EVT_TOL_NS, ALBEDO_TOL_NS

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("sample", nargs="?", default="18x275",
                    help="folder name under csv_root, or an absolute dir")
parser.add_argument("--files", nargs="*", default=None,
                    help="specific stems, e.g. 0022.trk_hits (default: all)")
parser.add_argument("--out-version", default="v3",
                    help="suffix of the output dir; bump when the label "
                         "rules change so old datasets stay comparable")
args = parser.parse_args()

data_dir = (Path(args.sample) if args.sample.startswith("/")
            else CFG.csv_root / args.sample)
out_dir = CFG.parquet_dir / f"{data_dir.name}_{args.out_version}"
stems = (args.files if args.files else
         sorted({p.name.split(".csv")[0]
                 for p in data_dir.glob("*.trk_hits.csv*")}))
print(f"{len(stems)} file(s) from {data_dir}\n -> {out_dir}")

# ---- what we keep, and as which dtype --------------------------------------
# f32 is plenty at mm/ns scale; ids and labels are small ints. The first
# 8 columns feed models; the prt_* extras exist for analysis only.
SCHEMA = {
    "evt": pl.UInt32,               # event (= frame) number within the file
    "prt_origin": pl.Int8,          # 1,2 signal | 3,4 background | 0 unknown
    "trk_hit_cell_id": pl.UInt64,   # packed detector-element id
    "trk_hit_system_id": pl.Int16,  # major detector system
    "trk_hit_pos_x": pl.Float32,    # [mm]
    "trk_hit_pos_y": pl.Float32,
    "trk_hit_pos_z": pl.Float32,
    "trk_hit_time": pl.Float32,     # [ns], TOF-corrected toward the origin
    "prt_pdg": pl.Int32,
    "prt_gen_status": pl.Int32,
    "prt_energy": pl.Float64,
    "prt_charge": pl.Float32,
    "prt_vtx_time": pl.Float32,
    "trk_hit_edep": pl.Float32,
}


# CSVs converted before 2026-09-30 carry the macro-era name of the generator status
# (the vocabulary was unified in simulation-pipeline/eic2ai/column_renames.py).
LEGACY_COLUMNS = {"prt_status": "prt_gen_status"}


def convert_one(csv_path: Path, parquet_path: Path) -> None:
    """The whole conversion for one file, readable top to bottom.

    Albedo rules (the physics is in reports/06):
      (a) per-particle: hit made > ALBEDO_TOL_NS after its own particle's
          creation ("lateness");
      (b) per-event: hit > ALBEDO_EVT_TOL_NS after the event's earliest
          origin-1 hit ("dt_event") — catches albedo whose persisted parent
          was itself born late, which rule (a) alone misses.
    trk_hit_time needs NO path/c correction (verified: it is TOF-corrected
    toward the origin). Both helper columns are kept for later analysis.
    """
    lateness = pl.col("trk_hit_time") - pl.col("prt_vtx_time")
    t = pl.col("trk_hit_time")
    t0_evt = pl.coalesce(  # earliest origin-1 hit; fallback: origin-1/2
        t.filter(pl.col("prt_origin") == 1).min().over("evt"),
        t.filter(pl.col("prt_origin").is_in([1, 2])).min().over("evt"))
    is_albedo = (pl.col("prt_origin").is_in([1, 2])
                 & ((pl.col("lateness") > ALBEDO_TOL_NS)
                    | (pl.col("dt_event") > ALBEDO_EVT_TOL_NS)))
    with open(csv_path, encoding="utf-8", errors="ignore") as csv_file:
        header = csv_file.readline().rstrip("\n").split(",")   # the first line alone; the file is up to 11 GB
    legacy = {old: new for old, new in LEGACY_COLUMNS.items() if old in header}
    overrides = {next((old for old, new in legacy.items() if new == name), name): dtype
                 for name, dtype in SCHEMA.items()}
    (
        pl.scan_csv(csv_path, schema_overrides=overrides,
                    infer_schema_length=100_000)
        .rename(legacy)
        .select(list(SCHEMA))
        .with_columns(lateness.cast(pl.Float32).alias("lateness"),
                      (t - t0_evt).cast(pl.Float32).alias("dt_event"))
        .with_columns(pl.when(is_albedo)
                      .then(pl.lit(5, dtype=pl.Int8))
                      .otherwise(pl.col("prt_origin"))
                      .alias("prt_origin"))
        .collect()   # the per-event window needs the whole file (~4 GB RAM)
        .write_parquet(parquet_path, compression="zstd",
                       row_group_size=1_000_000)
    )


out_dir.mkdir(parents=True, exist_ok=True)
for stem in stems:
    out = out_dir / f"{stem}.parquet"
    if out.exists():
        print(f"[skip] {out.name} exists")
        continue

    csv = data_dir / f"{stem}.csv"
    tmp_csv = None
    # some samples ship an EMPTY plain csv next to a valid zip (dvmp 0054);
    # fall back to the zip, extracted to tmp (polars streaming wants a
    # plain file) and removed right after.
    if not csv.exists() or csv.stat().st_size == 0:
        zip_path = data_dir / f"{stem}.csv.zip"
        if not zip_path.exists():
            print(f"[skip] {stem}: no usable csv or zip", flush=True)
            continue
        CFG.tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[unzip] {zip_path.name}", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(f"{stem}.csv", CFG.tmp_dir)
        tmp_csv = csv = CFG.tmp_dir / f"{stem}.csv"

    t_start = time.time()
    print(f"[convert] {csv.name} -> {out}", flush=True)
    try:
        convert_one(csv, out)
        print(f"[done] {out.name}: {out.stat().st_size / 1e9:.2f} GB "
              f"in {time.time() - t_start:.0f}s", flush=True)
    except Exception as exc:      # keep converting the rest of the list
        print(f"[FAIL] {stem}: {exc}", flush=True)
        out.unlink(missing_ok=True)
    finally:
        if tmp_csv is not None:
            tmp_csv.unlink(missing_ok=True)

"""Build train/val/test arrays from slim parquet files.

Split rule (leakage-safe): the unit of splitting is the EVENT, identified by
(file_tag, evt). All hits of one event land in exactly one fold, because hits
from the same event are physically correlated — splitting by hit would leak
information from train into test and inflate metrics.

Output layout (one directory per dataset version):
    <out_dir>/meta.json     split definition, normalization stats, id mapping
    <out_dir>/train.npz     X float32 [N,4] (x,y,z,t), sys uint8, y uint8, evt uint32
    <out_dir>/val.npz       (same keys)
    <out_dir>/test.npz      (same keys)

sys is the system_id remapped to a dense 0..K-1 index (embedding input).
evt is a globally unique event index (for per-event metrics later).
Normalization stats (mean/std of x,y,z,t and of r=hypot(x,y)) are computed on
the TRAIN fold only and stored in meta.json; the model bakes them in as
constant buffers so raw detector units go straight into ONNX at runtime.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from .labels import origin_to_label3

MODEL_COLUMNS = ["evt", "prt_origin", "trk_hit_system_id",
                 "trk_hit_pos_x", "trk_hit_pos_y", "trk_hit_pos_z",
                 "trk_hit_time"]


def build_dataset(parquet_files: dict[str, str | Path], out_dir: str | Path,
                  fractions=(0.70, 0.15, 0.15), seed: int = 42) -> dict:
    """parquet_files: {file_tag: path}. Returns the meta dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    frames, event_keys = [], []
    n_unknown = 0
    for file_idx, (tag, path) in enumerate(sorted(parquet_files.items())):
        df = pl.scan_parquet(path).select(MODEL_COLUMNS).collect()
        n_unknown += df.filter(pl.col("prt_origin") == 0).height
        df = df.filter(pl.col("prt_origin") != 0)
        # globally unique event id: file_idx * 1e6 + evt
        df = df.with_columns((pl.lit(file_idx * 1_000_000, pl.UInt32)
                              + pl.col("evt")).alias("gevt"))
        frames.append(df)
        event_keys.extend(file_idx * 1_000_000 + e for e in df["evt"].unique())

    df = pl.concat(frames)
    del frames

    # --- event-level split -------------------------------------------------
    event_keys = np.array(sorted(event_keys), dtype=np.uint32)
    rng.shuffle(event_keys)
    n = len(event_keys)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    folds = {
        "train": event_keys[:n_train],
        "val": event_keys[n_train:n_train + n_val],
        "test": event_keys[n_train + n_val:],
    }

    # --- dense system-id mapping (fit on ALL data: the id set is fixed by
    #     detector geometry, not by statistics, so this is not leakage) ------
    sys_ids = sorted(df["trk_hit_system_id"].unique().to_list())

    meta = {
        "seed": seed,
        "fractions": list(fractions),
        "files": {t: str(p) for t, p in sorted(parquet_files.items())},
        "unknown_origin_dropped": int(n_unknown),
        "system_ids": [int(s) for s in sys_ids],
        "fold_events": {k: len(v) for k, v in folds.items()},
    }

    gevt_all = df["gevt"].to_numpy()
    X_all = np.stack([df["trk_hit_pos_x"].to_numpy(),
                      df["trk_hit_pos_y"].to_numpy(),
                      df["trk_hit_pos_z"].to_numpy(),
                      df["trk_hit_time"].to_numpy()], axis=1).astype(np.float32)
    # dense index = position of the raw id in the sorted system_ids list
    sys_all = np.searchsorted(np.array(sys_ids),
                              df["trk_hit_system_id"].to_numpy()).astype(np.uint8)
    # y: 0 = signal, 1 = background, 2 = albedo (don't-care; excluded from
    # loss and efficiency denominators, reported separately)
    y_all = origin_to_label3(df["prt_origin"].to_numpy()).astype(np.uint8)
    del df

    for fold, keys in folds.items():
        mask = np.isin(gevt_all, keys)
        np.savez(out_dir / f"{fold}.npz", X=X_all[mask], sys=sys_all[mask],
                 y=y_all[mask], evt=gevt_all[mask])
        meta.setdefault("fold_hits", {})[fold] = int(mask.sum())
        meta.setdefault("fold_signal_hits", {})[fold] = int((y_all[mask] == 0).sum())
        meta.setdefault("fold_albedo_hits", {})[fold] = int((y_all[mask] == 2).sum())

    # --- normalization stats from the train fold only ----------------------
    tr = np.load(out_dir / "train.npz")
    Xt = tr["X"]
    r = np.hypot(Xt[:, 0], Xt[:, 1])
    meta["norm"] = {
        "mean": [float(m) for m in Xt.mean(0)] + [float(r.mean())],
        "std": [float(s) for s in Xt.std(0)] + [float(r.std())],
        "order": ["x", "y", "z", "t", "r"],
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_fold(dataset_dir: str | Path, fold: str):
    """Returns (X [N,4] f32, sys [N] int64, y [N] f32, evt [N] u32)."""
    d = np.load(Path(dataset_dir) / f"{fold}.npz")
    return (d["X"], d["sys"].astype(np.int64), d["y"].astype(np.float32),
            d["evt"])


def load_meta(dataset_dir: str | Path) -> dict:
    return json.loads((Path(dataset_dir) / "meta.json").read_text())

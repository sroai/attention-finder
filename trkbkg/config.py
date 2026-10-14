"""Machine profile for all scripts: where data lives on THIS machine.

Usage in a script:

    from trkbkg.config import CFG
    files = sorted((CFG.parquet_dir / "18x275_v3").glob("*.parquet"))

CFG is loaded once, from the first of:
  1. $TRKBKG_CONFIG            — point this at your own profile (farm etc.)
  2. <repo>/configs/local.yaml — the default workstation profile

All *_dir / *_root keys are pathlib.Path; a relative value is taken
relative to the checkout, so the shipped default profile works from any
working directory. Everything else (device, ...) comes through as-is. The YAML is deliberately ONE self-contained file —
if you need different paths, copy configs/farm-example.yaml and edit it;
do not build include chains.

Design note: this module is for STATIC machine facts only. Experiment
knobs (run names, epochs, batch sizes) are argparse flags in the scripts,
so `--help` always shows everything you can vary.
"""

import os
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_DIR / "configs" / "local.yaml"


def _as_path(value: str) -> Path:
    """A profile path: absolute as given, relative to the checkout otherwise."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_DIR / path


def _load() -> SimpleNamespace:
    cfg_path = Path(os.environ.get("TRKBKG_CONFIG", DEFAULT_CONFIG))
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"config profile not found: {cfg_path}\n"
            f"Set TRKBKG_CONFIG to a valid profile, or restore "
            f"{DEFAULT_CONFIG}")
    raw = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    ns = SimpleNamespace(**{
        k: _as_path(v) if isinstance(v, str) and (
            k.endswith("_dir") or k.endswith("_root")
            or k in ("eic_workbench",)) else v
        for k, v in raw.items()})
    ns.config_path = cfg_path          # so scripts can log which profile ran
    ns.repo_dir = REPO_DIR
    ns.plots_dir = REPO_DIR / "plots"      # small artifacts stay in the repo
    ns.reports_dir = REPO_DIR / "reports"
    return ns


CFG = _load()

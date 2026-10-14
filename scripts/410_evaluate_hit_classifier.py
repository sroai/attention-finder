#!/usr/bin/env python
"""Evaluate a trained run on the test fold and produce result plots.

Usage (from the repo root):
    uv run python scripts/410_evaluate_hit_classifier.py <run_name>

Protocol: threshold is picked on VAL at the target signal efficiency from the
run config, then applied to TEST. Outputs:
    runs/<run_name>/metrics.json            (also used by 411_compare_hit_runs.py)
    plots/results/<run_name>_*.png
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg import evaluate as ev
from trkbkg.dataset import load_fold, load_meta
from trkbkg.models import build_model
from trkbkg.scoring import score_fold

DATASET_DIR = (CFG.dataset_dir / "18x275_v3")
RUNS_DIR = CFG.runs_dir
PLOTS = Path(__file__).resolve().parents[1] / "plots" / "results"

run_name = sys.argv[1]
run_dir = RUNS_DIR / run_name
PLOTS.mkdir(parents=True, exist_ok=True)

ckpt = torch.load(run_dir / "best.pt", weights_only=False)
config = ckpt["config"]
meta = load_meta(DATASET_DIR)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(config["model"], meta, **config["model_kwargs"]).to(dev)
model.load_state_dict(ckpt["model_state"])

folds = {}
for fold in ("val", "test"):
    X, sysi, y, evt = load_fold(DATASET_DIR, fold)
    scores = score_fold(model, torch.from_numpy(X).to(dev),
                        torch.from_numpy(sysi).to(dev), evt)
    folds[fold] = (X, sysi, y, scores)

target = config["target_eff"]
y_va, s_va = folds["val"][2], folds["val"][3]
rej_va, thr = ev.rejection_at_efficiency(y_va, s_va, target)

_, sys_te, y_te, s_te = folds["test"]

# alternative operating mode: one threshold per detector system (see
# evaluate.per_system_thresholds for the rationale)
sys_va = folds["val"][1]
thr_sys = ev.per_system_thresholds(y_va, s_va, sys_va,
                                   len(meta["system_ids"]), target)

metrics = {
    "run": run_name,
    "epoch": ckpt["epoch"],
    "target_signal_efficiency": target,
    "threshold_from_val": thr,
    "val": ev.evaluate_fold(y_va, s_va, thr),
    "test": ev.evaluate_fold(y_te, s_te, thr),
    "test_per_system": ev.per_system_table(y_te, s_te, sys_te,
                                           meta["system_ids"], thr),
    "per_system_thresholds": {str(s): float(t) for s, t in
                              zip(meta["system_ids"], thr_sys)},
    "test_per_system_thresholds_mode":
        ev.evaluate_fold_per_system(y_te, s_te, sys_te, thr_sys),
}
ev.save_metrics(metrics, run_dir / "metrics.json")

history = [json.loads(l) for l in (run_dir / "history.jsonl").read_text().splitlines()]
ev.plot_training_history(history, PLOTS / f"{run_name}_training.png")
ev.plot_score_distribution(y_te, s_te, thr, PLOTS / f"{run_name}_scores.png",
                           title=f"{run_name}: test-fold score distribution")
ev.plot_eff_rejection_curve(y_te, s_te, target,
                            PLOTS / f"{run_name}_eff_rejection.png",
                            title=f"{run_name}: test fold")
ev.plot_per_system(metrics["test_per_system"],
                   PLOTS / f"{run_name}_per_system.png",
                   title=f"{run_name}: test fold at threshold {thr:.3f}")

print(json.dumps({k: metrics[k] for k in
                  ("threshold_from_val", "val", "test",
                   "test_per_system_thresholds_mode")}, indent=2))

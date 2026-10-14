# Running on a farm: other data dirs, other GPU, slurm, results in 10 hours

The scenario this page answers: *"I need to train on the farm, my data is
in a different place, the GPU is different, I want to submit tonight and
look at results tomorrow."*

## 1. One file describes the machine

Copy the example profile and edit the paths (it is fully commented):

```bash
cp configs/farm-example.yaml ~/trkbkg-farm.yaml
$EDITOR ~/trkbkg-farm.yaml        # csv_root, results_root, runs_local_dir
```

Every script honors `TRKBKG_CONFIG`, so the SAME commands from the README
run unchanged on the farm — only the profile differs. Nothing else in the
repo needs editing; if you find yourself editing a script to change a
path, that path belongs in the profile instead.

GPU differences are per-script flags, not config: every trainer documents
its memory levers in `--help` (`--batch-size`, `--cpu-data`,
`--windows-per-batch`, `--batch`). Defaults fit a 12 GB card; halve the
batch flags for smaller cards.

## 2. One job script, end to end

`sbatch` example — convert, build, train, evaluate, all in one job
(adjust partition/account to your site; conversion and dataset building
are CPU-bound, training wants the GPU):

```bash
#!/bin/bash
#SBATCH --job-name=trkbkg-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G          # check your site's provisioning ratio
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out        # NOT on a shared /work-style FS if your
                                  # site forbids streaming logs there

set -euo pipefail
cd /path/to/checkout              # the repo, with its .venv set up via uv
export TRKBKG_CONFIG=$HOME/trkbkg-farm.yaml
PY=.venv/bin/python

# data preparation (idempotent — skips whatever already exists)
$PY scripts/110_convert_csv_to_parquet.py 18x275
$PY scripts/120_build_hit_dataset.py

# smoke first: 1 epoch, fails fast if anything is misconfigured
$PY scripts/310_train_hit_classifier.py farm_v1 --smoke

# the real training + evaluation + deployable model
$PY scripts/310_train_hit_classifier.py farm_v1 --epochs 12
$PY scripts/410_evaluate_hit_classifier.py farm_v1
$PY scripts/510_export_hit_model_onnx.py farm_v1

# live checkpoints were on node-local scratch (runs_local_dir);
# the finished run + metrics + onnx are already under runs_dir.
```

```bash
sbatch train_trkbkg.sbatch
```

## 3. What to look at in the morning

* `runs_dir/farm_v1/history.jsonl` — one line per epoch; `val_rej@0.99`
  should climb and plateau.
* `runs_dir/farm_v1/metrics.json` — the test numbers.
* `plots/results/` in the checkout — evaluation plots (they are small and
  live in the repo, so `rsync` or open them directly).

## 4. Practical notes

* **Idempotency is the recovery story**: every builder skips existing
  outputs, trainers write `last.pt` each epoch. If the job dies at hour 7,
  resubmit — it fast-forwards through the done parts.
* **Array jobs for bulk conversion**: `110_convert…` takes `--files`, so
  `#SBATCH --array=0-9` + `--files $(printf "%04d.trk_hits" $SLURM_ARRAY_TASK_ID)`
  spreads a big sample over nodes.
* **Window-finding trainings** (321/330/340) are the GPU-hungry ones;
  their `--help` documents the exact memory-vs-speed flags. Budget:
  finder2d ≈ 10 min, watt ≈ 30 min, verifier ≈ 2–3 h on a 12 GB card.
* The shared-GPU OOM-retry warnings (`CUDACachingAllocator`) are benign;
  a hard `torch.OutOfMemoryError` means halve the batch flag.

#!/bin/bash
# Phase-6b augmentation-diversity experiments on the winner arch (w32 b3).
# Evidence base: sweep showed capacity saturated (v1 87.9 best) and v3_wide
# overfit (train loss 0.03) -> more augmentation diversity is the open lever.
# Expectations (recorded per protocol):
#   v5_roll       per-sample cyclic shifts, 90 ep: +0.5-1.5 pts — batch-shared
#                 shifts leave samples of a batch correlated; per-sample rolls
#                 give the full ~90x augmentation factor.
#   v6_rollsmooth v5 + label smoothing 0.1 on adjacent bins, 90 ep: +0-1 pt —
#                 near-bin-edge t0 makes a hard one-hot target noisy.
set -e
set -o pipefail
cd /work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python
S=scripts

for spec in "finder2d_v5_roll --per-sample-roll" \
            "finder2d_v6_rollsmooth --per-sample-roll --smooth 0.1"; do
  name=${spec%% *}
  echo "=== train $name ==="
  $PY $S/321_train_window_finder_2d.py $spec --epochs 90 --local-run-dir 2>&1 | tail -1
  echo "=== eval $name ==="
  $PY $S/421_evaluate_window_finder_2d.py $name 2>&1 | grep -A7 '"overall"' | head -8
done
echo AUG_SWEEP_DONE

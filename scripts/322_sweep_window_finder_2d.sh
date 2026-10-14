#!/bin/bash
# Phase-6 CNN tuning sweep. Expectations (recorded per protocol):
#   v2_deep  (w32 b4, 60ep): +1-2 pts — larger spatial receptive field
#   v3_wide  (w64 b4, 60ep): capacity check — small gain expected, data is
#            plentiful now so underfitting is plausible for v1's 350k params
#   v4_bn    (w64 b4 + batchnorm, 60ep): faster convergence, mild extra gain
# Baseline finder2d_v1 (w32 b3, 40ep): val 88.2 / test-18x275 88.3.
set -e
set -o pipefail
cd /work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python
S=scripts

for spec in "finder2d_v2_deep --width 32 --blocks 4" \
            "finder2d_v3_wide --width 64 --blocks 4" \
            "finder2d_v4_bn --width 64 --blocks 4 --batchnorm"; do
  name=${spec%% *}
  echo "=== train $name ==="
  $PY $S/321_train_window_finder_2d.py $spec --epochs 60 --local-run-dir 2>&1 | tail -1
  echo "=== eval $name ==="
  $PY $S/421_evaluate_window_finder_2d.py $name 2>&1 | grep -A7 '"overall"' | head -8
done
$PY $S/421_evaluate_window_finder_2d.py finder2d_v1 > /dev/null 2>&1 || true
$PY $S/422_finder2d_tuning_report.py
echo SWEEP_DONE

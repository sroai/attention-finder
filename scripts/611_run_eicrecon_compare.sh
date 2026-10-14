#!/bin/bash
# Phase 7 statistics runs: modified eicrecon on the 18x275 background files,
# three variants per file:
#   finder = AI filter with ONNX window finder (deployment mode)
#   ideal  = AI filter with MC-truth window (upper bound)
#   unfilt = tracking rewired back to unfiltered CentralTrackingRecHits (baseline)
# Output: /data/msf-2026-06/ai-results/eicrecon/{variant}_{file}.root + .log
# Analysis: scripts/610_eicrecon_tracking_compare.py
set -uo pipefail
OUT=/data/msf-2026-06/ai-results/eicrecon
IN_DIR=/data/msf-2026-06/dd4hep-background/18x275
FILES="0022 0023"
COLS="EventHeader,CentralTrackSeeds,CentralCKFTracks,CentralCKFTrackParameters,CentralCKFTrajectories,CentralCKFTrackAssociations,CentralTrackVertices,CentralTrackingRecHits,AiSignalTrackingRecHits,AiNoiseTrackingRecHits,SiBarrelTrackerRecHits,SiBarrelVertexRecHits,SiEndcapTrackerRecHits,MPGDBarrelRecHits,OuterMPGDBarrelRecHits,BackwardMPGDEndcapRecHits,ForwardMPGDEndcapRecHits,B0TrackerRecHits,ForwardRomanPotRecHits,ForwardOffMTrackerRecHits,TOFBarrelSharedRecHits,TOFEndcapSharedRecHits"

run () {
  local tag="$1" fno="$2"; shift 2
  echo "=== ${tag}_${fno} start $(date +%H:%M:%S) ==="
  /usr/bin/time -v /eic-workbench/run_ai_eicrecon.sh \
    "$IN_DIR/k_lambda_18x275_5000evt_${fno}.edm4hep.root" \
    "$OUT/${tag}_${fno}.root" 18x275 \
    -Pnthreads=12 -Ppodio:output_collections="$COLS" "$@" \
    > "$OUT/${tag}_${fno}.log" 2>&1
  echo "=== ${tag}_${fno} exit=$? end $(date +%H:%M:%S) ==="
}

for fno in $FILES; do
  run finder "$fno"
  run ideal  "$fno" -Ptracking:AiCentralTrackingHitFilter:windowMode=ideal
  run unfilt "$fno" -Ptracking:CentralWithoutTOFTrackerMeasurements:InputTags=CentralTrackingRecHits \
                    -Ptracking:CentralTrackSeeds:InputTags=CentralTrackingRecHits
done
echo ALL_DONE

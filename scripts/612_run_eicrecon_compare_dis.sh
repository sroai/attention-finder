#!/bin/bash
# Phase 7b: 3-variant eicrecon runs over the 26.07.0 DIS+background files in
# /data/background-2026-07 (pythia8 NC DIS 10x275 minQ2=1000, 99 frames/file).
# Skips files modified less than 2 minutes ago (downloads in progress).
# Output: /data/msf-2026-06/ai-results/eicrecon-dis/{variant}_{fileno}.root
set -uo pipefail
OUT=/data/msf-2026-06/ai-results/eicrecon-dis
IN_DIR=/data/background-2026-07
mkdir -p "$OUT"
COLS="EventHeader,CentralTrackSeeds,CentralCKFTracks,CentralCKFTrackParameters,CentralCKFTrajectories,CentralCKFTrackAssociations,CentralTrackVertices,CentralTrackingRecHits,AiSignalTrackingRecHits,AiNoiseTrackingRecHits,SiBarrelTrackerRecHits,SiBarrelVertexRecHits,SiEndcapTrackerRecHits,MPGDBarrelRecHits,OuterMPGDBarrelRecHits,BackwardMPGDEndcapRecHits,ForwardMPGDEndcapRecHits,B0TrackerRecHits,ForwardRomanPotRecHits,ForwardOffMTrackerRecHits,TOFBarrelSharedRecHits,TOFEndcapSharedRecHits"

run () {
  local tag="$1" fno="$2" input="$3"; shift 3
  [ -s "$OUT/${tag}_${fno}.root" ] && { echo "skip ${tag}_${fno} (exists)"; return; }
  echo "=== ${tag}_${fno} start $(date +%H:%M:%S) ==="
  /usr/bin/time -v /eic-workbench/run_ai_eicrecon.sh "$input" "$OUT/${tag}_${fno}.root" 10x275 \
    -Pnthreads=12 -Ppodio:output_collections="$COLS" "$@" \
    > "$OUT/${tag}_${fno}.log" 2>&1
  echo "=== ${tag}_${fno} exit=$? end $(date +%H:%M:%S) ==="
}

now=$(date +%s)
for input in "$IN_DIR"/pythia8NCDIS_*.edm4hep.root; do
  mt=$(stat -c %Y "$input")
  if [ $((now - mt)) -lt 120 ]; then echo "skip $input (recent, may be downloading)"; continue; fi
  fno=$(echo "$input" | grep -oE '1\.[0-9]+' | tr -d '.')   # hiDiv_1.0531 -> 10531
  run finder "$fno" "$input"
  run ideal  "$fno" "$input" -Ptracking:AiCentralTrackingHitFilter:windowMode=ideal
  run unfilt "$fno" "$input" -Ptracking:CentralWithoutTOFTrackerMeasurements:InputTags=CentralTrackingRecHits \
                             -Ptracking:CentralTrackSeeds:InputTags=CentralTrackingRecHits
done
echo ALL_DONE

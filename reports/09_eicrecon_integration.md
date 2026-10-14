# Phase 7 — EICrecon integration of the AI tracker-hit background filter

Date: 2026-07-14. Proof-of-work integration: the phase-4/5 two-stage ML
pipeline (2D-CNN window finder + attention hit classifier + MLP fallback) now
runs INSIDE EICrecon via ONNX Runtime, splits the central tracking hits into
signal/noise collections, and central tracking consumes only the signal hits.

## What was built (all in /eic-workbench/eicrecon, install /eic-workbench/eicrecon-install)

- **Upstream layout (restructured for the PR, 2026-07-14)** — no private
  plugin any more; everything sits where EICrecon expects it:
  `src/algorithms/tracking/AiTrackerHitFilter.{h,cc}` + `...Config.h`,
  `src/factories/tracking/AiTrackerHitFilter_factory.h`, wired in
  `src/global/tracking/tracking.cc`. Verified byte-identical outputs before
  and after the move (30 events, every collection event-by-event).
  - `AiTrackerHitFilter` (algorithm): per frame — gather all tracker hits
    (12 collections incl. far-forward, matching the training converter), find
    t0 (three modes), score every hit, split `CentralTrackingRecHits` into
    `AiSignalTrackingRecHits` + `AiNoiseTrackingRecHits` (podio subset
    collections, matched by ObjectID).
    Modes (`-Ptracking:AiCentralTrackingHitFilter:windowMode=...`):
    `finder` (default) = finder2d_v1 ONNX argmax; `ideal` = MC-truth t0 from
    raw-hit associations (origin walk identical to the training converter,
    two-rule albedo exclusion); `off` = passthrough.
  - In-window hits ([t0-50, t0+300] ns) -> watt_r2 attention model, one call
    per frame; out-of-window -> mlp_v1_r3; keep iff score < threshold
    (r3base calibration 0.99198 / 0.81686). Full contract in
    notes/cxx_porting_spec.md.
  - Collectors `AiAllTrackerRecHits` / `AiAllTrackerRawHitAssociations`
    (12 collections; TOF names are `TOF*Shared*` in this EICrecon,
    RICHEndcapN rec hits no longer exist).
- **Rewiring** (`src/global/tracking/tracking.cc`): `CentralTrackSeeds` and
  `CentralWithoutTOFTrackerMeasurements` read `AiSignalTrackingRecHits`.
  Revert to stock at runtime:
  `-Ptracking:CentralWithoutTOFTrackerMeasurements:InputTags=CentralTrackingRecHits
   -Ptracking:CentralTrackSeeds:InputTags=CentralTrackingRecHits`.
- The two Ai collections are added to the default podio output
  (JEventProcessorPODIO.cc). Nothing else outside the tracking plugin is
  touched: eicrecon.cc and src/global/CMakeLists.txt are back to stock.
  ONNX Runtime is linked by `algorithms/tracking` and `global/tracking`
  (`plugin_add_onnxruntime`).
- **Weights** local in `/eic-workbench/ai-trkfilter-models/` (finder2d_v1
  re-saved as IR-9 single file for the container's onnxruntime 1.17 — parity
  diff 0.0; watt_r2 + mlp_v1_r3 as exported). Total 1.1 MB.
- Wrapper: `/eic-workbench/run_ai_eicrecon.sh` (sets LD_LIBRARY_PATH order,
  ignores broken particle_flow plugin — see notes/lesson-eicrecon-build-env.md).

## Validation of the mechanics (5-event batches, 18x275 file 0022)

- signal+noise == input central hits exactly, no unmatched hits.
- C++ finder t0 vs C++ ideal t0 on the same events: -4.4, -25.1, -7.4,
  -16.5, -14.6 ns — 5/5 within 50 ns (matches python finder quality).
- Runtime cost of the filter is negligible: filtered runs are not slower
  than unfiltered (seeding gets 4x cheaper, which pays for the ONNX).

## Full-statistics comparison (10 000 events, files 0022+0023, 12 threads)

Files: /data/msf-2026-06/ai-results/eicrecon/{unfilt,finder,ideal}_{0022,0023}.{root,log}
(runner scripts/41_run_eicrecon_compare.sh); analysis
scripts/40_eicrecon_tracking_compare.py; plots plots/phase7/; numbers
reports/phase7_tracking_compare.json.

The plot set was rebuilt after studying the tracking-WG presentations
(/work/BkgImpact_PV.pdf, ePIC_Tracking_ShyamKumar_26Mar2026.pdf,
beampipe_gold_coating_2026-2.pdf) to cover all three directions the WG uses
to prove algorithms under background — seeding, track fitting, vertexing —
with their standard observables: efficiency AND purity vs eta in momentum
bins, nMeasurements per track, dp/p and DCA_r resolutions vs p/pT in eta
regions, N vertices, tracks-in-vertex-fit, vertex chi2/ndf, vertex residuals
and core resolutions with the smallest-chi2/ndf vertex selection. (Their
vertex-perf-vs-N_MC axis is degenerate for exclusive k_lambda — N_MC ~ 2
charged primaries per event — so vertex metrics are single numbers here.)

| metric (central tracking + vertexing) | unfiltered | AI finder | AI ideal-t0 |
|----------------------------------------|-----------:|----------:|------------:|
| track seeds / event                    |      577.9 |     147.7 |       148.4 |
| CKF tracks / event                     |       2.60 |      2.11 |        2.14 |
| signal-track efficiency (19958 charged primaries, w>0.5) | 96.8% | 95.6% | **96.8%** |
| track purity (signal fraction)         |      74.3% |     90.2% |       90.3% |
| measurements / track (mean)            |       6.66 |      7.23 |        7.23 |
| sigma(dp/p) signal tracks              |      0.94% |     0.94% |       0.94% |
| sigma(DCA_r) signal tracks             |     8.5 um |    8.4 um |      8.5 um |
| **vertex found within 1 mm of MC PV**  |  **82.8%** | **87.0%** |   **88.9%** |
| sigma(vertex x / y / z), core          | 12.9/14.3/10.0 um | 8.8/8.7/6.5 um | 8.8/8.7/6.6 um |

Headlines:
1. **Vertexing improves outright** (the metric the WG decks care most
   about): the probability that the selected (smallest-chi2/ndf) vertex lies
   within 1 mm of the MC primary vertex rises 82.8% -> 87.0% (finder) /
   88.9% (ideal), and the core vertex resolution improves ~35% in every
   coordinate (z: 10.0 -> 6.5 um). Unfiltered has MORE events with a vertex
   (95.7% vs 93.3%) but background tracks pull the fit: its residual tails
   are visibly fatter (plots/phase7/vertex_residuals.png), more vertices are
   fit from 3-4 tracks where only 2 signal tracks exist, and 2.4x more
   vertices land in the chi2/ndf >= 50 overflow.
2. **CKF tracks get cleaner internally**: the 4-measurement spike of the
   unfiltered sample (background hits attached / short fake tracks)
   collapses by 3x and tracks migrate to 7-9 measurements (mean 6.66 ->
   7.23) — with background hits absent, CKF picks up the RIGHT hits.
3. **Fit-quality resolutions are untouched**: dp/p and DCA_r vs p/pT overlap
   within errors in all eta regions — matching the WG conclusion pattern
   ("similar resolution, cleaner sample"). The filter removes hits, it does
   not distort surviving tracks.
4. **Purity vs eta localizes the gain**: at 1<p<5 GeV the signal fraction
   rises from ~20% (unfiltered) to ~60-90% (filtered) across eta; almost
   all remaining fakes live at p < 1 GeV.
5. Prior headlines confirmed at 2x statistics: seeds /3.9, track purity
   74 -> 90%, ideal window costs zero efficiency (96.83% vs 96.83%), finder
   argmax costs 1.3 pp (95.6%).

### Earlier 5000-event (0022-only) headlines, superseded by the table above:
1. **Seeding load /3.9** (577 -> 148 seeds/event) — the filter removes the
   background hits before they ever reach seeding.
2. **Fake tracks /3.4** (25% -> 9% of the track sample) at track level.
3. **Ideal window loses nothing**: 97.0% vs 97.0% signal-track efficiency
   (-0.02 pp, 2 tracks in 9684). The per-hit classifier at the deployed
   operating point is track-level lossless when t0 is right.
4. **The finder mode costs 1.5 pp** signal-track efficiency (95.5%) — this is
   the ~8% of frames where the window finder picks the wrong bin (phase-6
   finder recall 92% on 18x275). Efficiency-vs-eta shows the loss is flat-ish
   with a backward bias (eta < -1.5 dips to ~89%). The phase-5/6 reranker
   stack (top-10 + verifier, recall 93.4%) is NOT ported yet — it would
   recover roughly half the gap; that plus threshold recalibration against
   finder2d (thresholds were calibrated for wf_v2_r3) are the obvious next
   steps.

## Generalization: 26.07.0 DIS campaign (2026-07-14, run 8c)

New campaign-style files appeared in /data/background-2026-07 (pythia8 NC DIS
10x275 minQ2=1000 with machine background, geometry release 26.07.0, 99
frames/file). Compatibility check: the 26.07.0 cellID system ids match our
22-id training vocabulary exactly, and the SAME eicrecon binary + ONNX
weights ran the files unmodified (no new epic build needed); C++ finder t0
agreed with MC truth within one 25 ns bin on 5/5 spot-checked frames.
6 complete files (594 frames) x 3 variants (runner
scripts/42_run_eicrecon_compare_dis.sh; file 1.0544 arrived truncated and is
excluded; rerun the runner to sweep up files that finish downloading later).
Analysis: `40_... --dataset dis26`; plots plots/phase7_dis/; numbers
reports/phase7_tracking_compare_dis.json.

| metric (central tracking + vertexing) | unfiltered | AI finder | AI ideal-t0 |
|----------------------------------------|-----------:|----------:|------------:|
| track seeds / event                    |      507.8 |     104.9 |       104.9 |
| CKF tracks / event                     |       9.36 |      8.81 |        8.94 |
| signal-track efficiency (6022 charged primaries) | 80.7% | 79.5% | **80.8%** |
| track purity (signal fraction)         |      92.1% |     96.4% |       96.5% |
| measurements / track (mean)            |       6.42 |      6.51 |        6.52 |
| sigma(dp/p) signal tracks              |      1.45% |     1.45% |       1.45% |
| sigma(DCA_r) signal tracks             |      76 um |     79 um |       78 um |
| vertex found within 1 mm of MC PV      |      96.8% |     97.8% |       98.0% |
| sigma(vertex z), core                  |    17.9 um |   17.2 um |     17.0 um |

Reading: a different physics process, different beam energy, and a NEWER
geometry release than anything in training — and the pattern holds: seeding
/4.8, purity +4.4 pp, vertexing slightly better, ideal window again at
efficiency parity (+0.1 pp), finder argmax costs 1.1 pp. DIS events carry
~10 signal tracks each, so vertexing is already robust unfiltered (96.8%);
the vs-N_MC view (plots/phase7_dis/vertex_perf_vs_nmc.png) is enabled for
this dataset. Slides updated: rewiring diagram + both result tables
(slides 21-23 of /work/slides/trk_hit_bkg_progress.pptx; build_slides.py
reads the JSON reports at build time).

## Caveats / proof-of-work shortcuts

- Only ARGMAX of finder2d is deployed, not the candidates+verifier reranker
  (t0_final_v1 pipeline) — C++ port of the 11-feature selector is bigger work.
- Thresholds are the r3base calibration (wf_v2_r3-based); a recalibration on
  finder2d t0 estimates would shift the eff/rejection trade slightly.
- TOF cluster hits enter CentralTrackerMeasurements directly (bypass the
  filter); B0/far-forward tracking left stock.
- Hit-level eff/rejection was NOT re-measured in C++ (python numbers:
  99.0%/39.7%); the track-level numbers above are the deployment-relevant ones.
- edm4hep input is digitized fresh by this EICrecon version — hits are
  statistically, not bitwise, identical to the training CSVs.

## Reproduce

```bash
# build (already done): cmake -S /eic-workbench/eicrecon -B build && cmake --build build && cmake --install build
# one event batch, finder mode, debug prints:
/eic-workbench/run_ai_eicrecon.sh <in.edm4hep.root> out.root 18x275 \
  -Pjana:nevents=5 -Ptracking:AiCentralTrackingHitFilter:LogLevel=debug
# three-variant full comparison + plots:
bash scripts/41_run_eicrecon_compare.sh   # 3 variants x ~5 min

python scripts/40_eicrecon_tracking_compare.py
```

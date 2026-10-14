# Phase 4: attention inside the window — end-to-end results

> **v2 rerun (2026-07-13, albedo-aware data — CURRENT RESULTS).** After the
> albedo relabeling (`prt_origin=5`, reports/06; lateness > 100 ns; ~0.84% of
> former signal), the whole pipeline was rebuilt and retrained on
> `parquet/18x275_v2` (runs `mlp_v1_r2`, `wf_v2_r2`, `watt_r2`, `wmlp_r2`).
> Test, efficiency over *reconstructable* signal only:
>
> | system | signal eff | bkg rejection | albedo filtered |
> |---|---|---|---|
> | mlp_v1_r2 alone | 98.8% | 14.2% | 0.2% |
> | window + dt (wmlp_r2) | 98.9% | 34.1% | 0.9% |
> | **window + dt + attention (watt_r2)** | **98.9%** | **39.9%** | 8.9% |
> | perfect-finder bound (same models) | 99.0% | **97.5%** | — |
>
> Same conclusions as v1, slightly better numbers: the cleaner in-window
> efficiency target let the inside threshold tighten (in-window background
> rejected: 49% → **67.5%**). Albedo hits are don't-care: excluded from loss
> and efficiency, their filtered fraction reported for information. Window
> finder: 77.9% of frames within 50 ns; containment 88.2%; back-edge loss in
> found frames dropped 0.69% → **0.48%** (albedo *was* a third of it). The
> finder remains the bottleneck. Historical v1 numbers below for the record.

*2026-07-13 — runs `watt_v1` (2 attention blocks) and `wmlp_v1` (identical
but 0 blocks); window-hit dataset `window_hits_v1` (front 50 / back 300 ns
around wf_v2's t0_est, train windows jittered ±25 ns). Plots in
`plots/window/`; spec in `/work/trk_hit_bkg.md` Phase 4.*

## Pipeline being evaluated (per frame)

1. `wf_v2` window finder → t0_est (phase 3).
2. Hits inside [t0_est−50, t0_est+300] ns → `watt_v1` scores. Input per hit:
   (x, y, z, **dt = t − t0_est**) + system embedding.
3. Hits outside → phase-1 `mlp_v1` scores (the window is a feature, not a
   hard cut — the finder misses ~20% of frames, and a hard cut would cap
   efficiency at 88%).
4. Two thresholds calibrated on val (grid search): maximize total rejection
   subject to ≥99% TOTAL signal efficiency, window losses charged.

## Headline numbers (test fold, efficiency vs ALL signal hits)

| system | signal eff | bkg rejection |
|---|---|---|
| phase 1: mlp_v1 alone | 98.8% | 14.1% |
| window + dt, NO attention (wmlp_v1) | 99.0% | 34.2% |
| **window + dt + attention (watt_v1)** | **99.0%** | **39.3%** |

End-to-end curve (`watt_v1_e2e_curve.png`): 27% rejection @ 99.5% eff,
48% @ 98.4%, 81% @ 95%. In-window score separation:
`watt_v1_e2e_scores.png`.

Attribution of the gain (in-window hit classification, val):

| model | in-window AUC | in-window rej @ 99.9% in-window eff |
|---|---|---|
| phase-1 features (no dt) | 0.865 | — |
| + dt (wmlp_v1) | 0.963 | 20% |
| + attention (watt_v1) | **0.991** | **51%** |

So the user's hypothesis is confirmed twice over: (a) the temporal window is
the single most valuable piece of information in the problem, and (b)
hit-to-hit attention inside the window sees the burst coherence that per-hit
models cannot — it alone multiplies in-window rejection by 2.5×.

## Deployment (all ONNX, CPU, parity-checked)

| component | file | size | CPU latency |
|---|---|---|---|
| window finder | runs/wf_v2/window_finder.onnx | 42 KB | 0.025 ms/frame |
| in-window model | runs/watt_v1/window_model.onnx | 316 KB | 2.2 ms/window (~1200 hits) |
| outside model | runs/mlp_v1/model.onnx | 33 KB | 1.35 ms/event |

Whole pipeline ≲ 4 ms per frame single-threaded, < 1 MB of weights —
compatible with the ≤2 GB / prefer-≪1 GB constraint by orders of magnitude.
Glue logic for C++: per-bin counting for the finder (documented in
`window_deploy_info.json`), one subtraction for dt, threshold table in
`metrics_end_to_end.json`.

## Decomposition — why 39% and not more (script 18, `metrics_end_to_end.json`)

Sanity challenge: the window ALONE rejects 83.5% of background — why does the
full pipeline only reach 39.3%? Because the window alone keeps only 88.4% of
signal (hard cut = illegal at the 99% requirement). The stranded 11.6% of
signal — **94% of it from the frames where the finder missed the burst** —
must be recovered through the outside path, which forces a loose mlp_v1
threshold there (only 37.4% rejection). Exact budget (test):

| region | share of bkg | rejected there | contribution to total |
|---|---|---|---|
| inside window | 16.5% | 49.1% (watt_v1) | 8.1 pts |
| outside window | 83.5% | 37.4% (mlp_v1, loose) | 31.2 pts |
| **total** | | | **39.3%** |

So the in-window 51% number never could scale to the total: in-window hits
are only 16.5% of all background. The lever is the 83.5% outside, and it is
held hostage by finder misses.

**Perfect-finder bound (same models, true t0, same calibration protocol):
99.1% efficiency at 97.3% rejection.** With true t0 only 0.74% of signal is
outside, so the outside path can be cut at 99.9% rejection, and the
in-window model runs at a comfortable threshold (84% in-window rejection).
The finder gap (78.5% of frames found vs 100%) therefore costs ~58 points of
rejection (39% → 97%) — every point of finder recall is worth roughly 2.5
points of end-to-end rejection. This is the single number that sets the
phase-5 agenda.

## Caveats & recommended next steps

1. **The window finder is now the bottleneck.** 20% of frames fall back to
   the phase-1 path (only 14% rejection there). Finder improvements (top-k
   candidate windows + verifier; sub-bin regression) convert directly into
   end-to-end rejection: perfect finding would push toward the ~60–70%
   range implied by in-window performance.
2. Threshold transfer val→test was clean (99.0% → 98.96%); add margin if
   the 99% must hold strictly.
3. The per-system efficiency caveat from phase 1 (endcap disks eaten by a
   global threshold) should be rechecked for this pipeline — per-system
   thresholds inside the window are the same cheap fix if needed.
4. Attention capacity was not tuned (2 blocks, d=64, single head, 15
   epochs); the val curve was still improving. A modest sweep is likely
   worth a few points, but finder work should come first.

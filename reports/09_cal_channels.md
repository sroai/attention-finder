# Phase 10 — calorimeter channels for the 2D window finder

**Result: adding two digitized-calorimeter image channels raises test
recall@50 ns from 92.0 % to 99.97 % on 10x130 background — the miss rate
drops 320-fold (8.0 % → 0.025 %, one missed frame in 3,988). The finder's
top-1 prediction now matches the phase-5..8 candidates+verifier ceiling
without any of that machinery.**

## Data

8 full 10x130 background files (k_lambda_10x130_5000evt_{0001-0003,
0005-0009}), converted 2026-09-30 with eic2ai (unified vocabulary, roles
trk_hits + cal_hits) into
`meson-structure-2026-06/csv-2026-09-bkg-phase10/10x130/` — 8 slurm jobs,
node-local staging via `--gres=disk:100G`, 38-41 min/job. 40k frames:
train 32,018 / val 3,986 / test 3,988 (80/10/10 by frame, seed 42; frames
without signal tracker hits dropped, as in phase 6).

## Images

`[4, 64, 88]` per frame (trkbkg/windows2d.py, scripts/132):

| ch | content | ranges |
|---|---|---|
| 0 | (z,t) tracker hit counts, log1p | phase-6 grid |
| 1 | (r,t) tracker hit counts, log1p | phase-6 grid |
| 2 | (z,t) calorimeter energy, 10 MeV units, log1p | z −4.5..8 m (ZDC clips to edge row), r 0..2.8 m |
| 3 | (r,t) calorimeter energy, 10 MeV units, log1p | same |

Energy weighting is the design choice from the frame displays
(cal_hit_bkg_plan.md §6): a p-gas burst deposits GeV where albedo deposits
MeV — counts would hide exactly the separation that matters.

## Protocol

A/B on the SAME dataset, splits and hyperparameters; the only difference
is the channel count (`321 --channels 2` slices the tracker-only view):

    321_train_window_finder_2d.py p10_trk --dataset images2d_p10 --channels 2 \
        --per-sample-roll --smooth 0.05
    321_train_window_finder_2d.py p10_cal --dataset images2d_p10 \
        --per-sample-roll --smooth 0.05

WindowFinder2D width 32, 3 blocks, 40 epochs, cyclic per-sample roll,
label smoothing 0.05. Tesla T4 (user's optim-gpu allocation), 16.8 min per
training, ~5 GB RSS. Runs + metrics:
`optim-out/phase10/runs/{p10_trk,p10_cal}` on the farm.

## Results (test, 3,988 frames)

Two metric families. SINGLE-SHOT grades the argmax prediction alone —
|predicted − t0| within a tolerance; the three tolerances are nested, so
the rows can only grow downward. SHORTLIST CEILING asks whether the true
column is among the k best-scored columns — the best any top-k +
verifier second stage could reach.

Single-shot (argmax within tolerance):

| tolerance | tracker-only | + calorimeter |
|---|---|---|
| within 25 ns | 90.5 % | **99.90 %** |
| within 50 ns | 92.0 % | **99.97 %** |
| within 100 ns | 92.5 % | 99.97 % |

Shortlist ceiling (true column in top-k):

| shortlist | tracker-only | + calorimeter |
|---|---|---|
| top-5 | 97.8 % | **100 %** |
| top-10 | 99.0 % | 100 % |

(val best@50: 91.5 % vs 99.97 %.) With calorimeter channels the argmax
row reaches the top-5 ceiling — a shortlist + verifier stage has nothing
left to recover; tracker-only had an 8-point gap between those worlds,
which is what justified the verifier.

Reading: the tracker-only misses (~8 %) are the known occupancy-trap
frames where the burst outshines a dim signal column; the calorimeter
energy map separates them almost perfectly — an ep interaction deposits a
prompt, radially-symmetric GeV-scale energy comb that beam-gas afterglow
does not reproduce, and the network finds it. Since top-1 already reaches
the old top-k ceiling, the candidates+verifier stage (phases 5, 8) is
unnecessary at this energy when calorimeter channels are available.

Caveats: one energy preset (10x130), single seed, 40k frames; the
baseline here (92.0 %) is itself above the published multi-preset 10x130
number (97.5 % recall@50 in phase 8 was with verifier; the phase-3-style
argmax comparison is what the A/B controls). Albedo relabeling for
calorimeter hits is not applied (open study); cal channels use raw
digitized hits as converted.

## Next (recommended)

1. Repeat at 18x275 (and eventually DVMP/DIS) to confirm across presets.
2. ONNX export of the 4-channel finder + EICrecon wiring (phase 7 path).
3. If the 0.03 % residual matters: inspect the single missed frame.

## Inference latency (script 522, 500 test frames, batch = 1)

| model | 1 CPU thread median / p99 | Tesla T4 median / p99 |
|---|---|---|
| tracker-only (2 ch) | 12.0 / 12.5 ms | 0.72 / 0.74 ms |
| + calorimeter (4 ch) | 10.1 / 10.2 ms | 0.66 / 0.73 ms |

The extra channels cost nothing measurable (only the first conv sees
them; differences are kernel-selection noise). Distributions are tight —
no tail. Torch-CPU numbers; the production ONNX path is expected at or
below these. Histograms: plots/phase10/finder2d_latency.png.

## Autopsy of the single missed frame (script 428)

`k_lambda_10x130_5000evt_0003` event 634: t0 = 167 ns, predicted 1412 ns.
The frame contains a proton-beam-gas hadronic burst at ~1400-2000 ns whose
calorimeter energy column (~27 GeV in-column) EXCEEDS the signal's
(~17 GeV in-column; the signal is modest — 36 tracker hits, 208 cal hits,
31 GeV total). The model picked the larger energy comb.

Verdict: the known occupancy-trap failure pushed to its energy extreme —
a burst that out-deposits the signal. Crucially the true bin ranks #2
with a logit gap of only 0.40: the finder SEES the interaction and loses
a photo-finish. Any top-2 disambiguation (the existing verifier, TTA, or
a shape-aware feature — the burst lacks the signal's radial symmetry and
backward-Ecal balance) recovers this frame. At one frame in 3,988 it is
not worth machinery unless the pattern repeats at other presets.
Figure: plots/phase10/p10_cal_miss_2000634.png.

## Super-frame scaling (script 523; k stacked 2 us frames, same weights)

The finder is fully convolutional along time, so a [4, 64, 88k] image runs
unchanged and yields 88k column logits. Measured (p10_cal, batch 1):

| k | CPU 1-thread total | CPU per frame | GPU total | GPU per frame |
|---|---|---|---|---|
| 1 | 10.0 ms | 10.0 | 0.71 ms | 0.71 |
| 2 | 19.0 ms | 9.5 | 1.04 ms | 0.52 |
| 3 | 27.8 ms | 9.3 | 1.14 ms | 0.38 |
| 5 | 47.0 ms | 9.4 | 1.79 ms | 0.36 |
| 10 | 95.4 ms | 9.5 | 1.97 ms | 0.20 |

CPU scales linearly (conv work ~ time width; no amortization to gain at
one thread). GPU is strongly sub-linear — the T4 is under-utilized at
88 columns, so stacking is nearly free: 3.5x the per-frame throughput at
k=10. Super-frames favor GPU deployment.

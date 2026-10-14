# Phase 2: temporal structure of frames — analysis & phase-3 recommendations

*2026-07-12 — files 0022+0023 (10,000 frames). Plots in `plots/temporal/`
(events in `plots/temporal/events/`, first 50 frames of 0022); numbers in
`reports/temporal_stats.json`. Phase-1 plots untouched.*

## Setup

One "event" = one 2000 ns streaming-readout frame containing exactly one
physics interaction at a random time t0, on top of continuous background.
Here t0 := earliest hit time of `prt_origin` 1/2 in the frame, and
dt := hit_time − t0.

## Findings

**F1. The signal burst is real, tight, and was completely washed out in
phase 1 by the random t0.** Aligned to t0 (`aligned_dt.png`), signal hits
are a sharp spike: median full span 76 ns, median 95%-span 44 ns, and on
average **99.3% of signal hits lie within t0+200 ns** (`signal_timing.png`).
The phase-1 "time is uniform for both classes" was exactly the un-aligned
projection of this. t0 itself is uniform in [0, 2000] ns as expected.

**F2. Purity vs dt is dramatic.** Signal fraction by dt window
(baseline 0.5%): [0,25) ns → **18.3%**, [25,50) → 8.7%, [50,100) → 0.7%,
[100,150) → 3.6% (a secondary bump — far-forward/secondary propagation,
visible as the second peak in `aligned_dt.png` zoom), beyond 200 ns → ~0.01%.
A model that knows dt gets a feature worth more than everything phase 1 had.

**F3. One interaction per frame — confirmed.** No second signal cluster:
hits after t0+200 are sparse stragglers (mean 0.7% of a frame's signal,
purity < 0.02%), not a cluster. 5.1% of frames have an intra-signal time gap
> 500 ns caused by these stragglers (`signal_max_gap.png`). Two frames in
10,000 have no signal tracker hits at all. Rare extreme tails exist: a few
frames have signal-secondary hits out to 42–71 µs (loopers / neutron
capture), far beyond the frame window.

**F4. Background ALSO arrives in bursts — the central complication.**
The event displays (`events/0022_evt0003.png` is a textbook example) show
origin-4 background forming vertical time-stripes with the same burst
morphology as the signal (beam-gas interactions are themselves point events
in time). The generator-status view (`events_status/*.png`, second display
set colored by status band: synrad=2000, ebrems=3000, …, and G4 secondaries
by ancestry) shows the composition: the flat gray carpet is synrad —
97% of all hits (34.3M of 35.3M in 0022) — while the time-bursts are G4
secondary showers of the other background sources (bands 3000–6000 are only
~14k hits total). So the frame is "synrad haze + a few shower bursts + one
signal burst". Consequences, measured:
  * Raw occupancy peak finds t0 in only **21%** of frames within 100 ns
    (median error 473 ns) — it usually locks onto a background burst
    (`t0_recovery.png`).
  * Weighting hits by mlp_v1 signal-likeness does NOT help (21.3%;
    `t0_recovery_scored.png`, script 09) — consistent with the phase-1
    feature ceiling: per-hit properties don't distinguish burst types.
  * What DOES differ is **collective geometry**: the signal burst spans all
    barrel layers out to r≈750 mm AND far-forward stations (z≈6 m, arriving
    ~20–150 ns later), radiating from (0,0,0); background bursts concentrate
    at low r / specific z with no such coherent radial pattern.

## Recommendations for phase 3 (temporal models)

Ranked; 1 is the recommended primary path.

1. **Two-stage: burst finder → per-hit classifier with dt.**
   Stage 1 scans the frame in time windows (e.g. 25 ns steps) and scores
   each window with a tiny NN on *aggregate* window features (hit count,
   max r, r-span, fraction of hits at r>600, far-forward count, z-spread,
   mean |z|) — exactly the geometry that separates signal bursts from
   beam-gas bursts (F4). Stage 2 = per-hit model (mlp_v1-like) plus
   dt = t − t0_est. Both stages are trivially ONNX/CPU compatible.
   Upper bound if stage 1 were perfect, from F2: rejecting everything
   outside [t0−25, t0+200] alone keeps ~99% of signal while rejecting
   ~89% of background — a 6× improvement over phase 1's 14%.
   First measurable milestone: stage-1 accuracy (frac of frames with
   |t0_est − t0| ≤ 50 ns); phase 2's 16% baseline must jump to >90%.

2. **End-to-end event model with time-aware attention/pooling** (extension
   of ec_v1): per-hit encoder + attention over hits with relative-time
   encodings, letting the network find the burst implicitly. More elegant,
   ONNX-exportable in principle, but harder to train/debug and CPU cost
   grows with N² unless windowed. Try only if the two-stage model
   under-delivers.

3. **Keep the per-hit stage-2 features minimal**: dt, plus the existing
   (x, y, z, system). Phase 1 showed capacity is not the issue; the win is
   the new information, not architecture size.

4. **Labeling detail for stage 1 training**: define the target window from
   t0 of origin-1/2 hits; treat the 2 signal-less frames and >2 µs
   straggler hits as background by construction.

5. **Evaluation additions**: report stage-1 t0 accuracy separately from
   end-to-end eff/rejection so failures are attributable; keep the
   per-system threshold analysis from phase 1 (the endcap-disk caveat
   still applies).

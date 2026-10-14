# Phase 3: window finder — locating the interaction in the frame

> **v2 rerun (albedo-aware data): run `wf_v2_r2`** — test 77.9% within 50 ns,
> median error 10.3 ns, containment (f50/b300) 88.2% — statistically
> unchanged vs v1 (78.5%): the finder never depended on albedo hits, and t0
> targets were already effectively albedo-free. Edge analysis on v2: front
> loss in found frames still 0.00%, back-edge loss 0.69% → **0.48%**
> (albedo removal explains the drop). Plots regenerated as
> `plots/window/wf_v2_r2_*` and `plots/window/events/wf_v2_r2_gevt*.png`.

*2026-07-13 — runs `wf_v1`/`wf_v2`; dataset `windows_v2` (per-frame [11, 88]
feature images from the phase-1 event-split folds). Plots in `plots/window/`;
spec in `/work/trk_hit_bkg.md` Phase 3.*

## Model

Scan-and-classify: 25 ns bins over the frame (88 bins), per-bin aggregate
geometry features, small dilated 1D CNN (~20k params) → per-bin logits,
softmax cross-entropy against the bin containing true t0. Inference =
argmax bin. ONNX: fixed [1, 11, 88] input, exported by script 13.

Features per bin (trkbkg/windows.py): total/outer-barrel/far-forward/
backward/vertex counts (log), r/z spread, occupancy contrast, and — added in
wf_v2 — burst-coherence counts: distinct systems firing, distinct
barrel-stack layers firing, BarrelTOF count. The coherence features encode
what separates the ep burst from beam-gas shower bursts (phase-2 F4).

## Results (test fold, 1499 frames)

| metric | value |
|---|---|
| frames with \|t0_est − t0\| ≤ 25 ns | 75.8% |
| ≤ 50 ns | **78.5%** (phase-2 occupancy-peak baseline: 16%) |
| ≤ 100 ns | 80.0% |
| median \|error\| | 10.2 ns |
| late by >25 ns (cuts burst front) | 10.3% |

wf_v1 (8 geometry features): 70% @50 ns → coherence features gave +8 points.
Training curves in `runs/wf_v2/history.jsonl`; error histogram
`plots/window/wf_v2_error_hist.png`; per-frame validation displays with the
estimated window drawn as a band vs true start/end lines:
`plots/window/events/wf_v2_gevt*.png`.

## Containment — the number that drives the phase-4 design

Signal-hit containment of the window [t0_est − front, t0_est + back]
(`wf_v2_containment.png`, full grid in `runs/wf_v2/metrics.json`):

| window | signal inside | background inside |
|---|---|---|
| front 50 / back 300 ns | **88.4%** | 16.5% |
| front 100 / back 1000 ns | 92.1% | 42.7% |

Containment saturates at ~88–92% because the ~20% of frames where the finder
locks onto a background burst lose their *entire* signal — wider margins
cannot buy it back. **Therefore the window must be used as a FEATURE, not a
hard cut**: a hard cut caps end-to-end signal efficiency at ~88%, far below
the 99% requirement. The phase-4 pipeline scores in-window hits with the
dt-aware model and lets out-of-window hits fall back to the phase-1 per-hit
model, with two thresholds calibrated jointly for ≥99% total efficiency
(script 16).

## Edge analysis: START vs END effectiveness (script 19)

Question: of the 11.6% signal lost by the window, how much is the front
edge's fault, how much the back edge's? Answer (test,
`wf_v2_edge_signal_dt.png`, `wf_v2_edge_margins.png`; numbers in
`runs/wf_v2/metrics.json → edge_analysis`):

| loss channel | % of ALL signal | interpretation |
|---|---|---|
| front (dt < −50), finder-found frames | **0.00%** | start edge is perfect |
| back (dt > +300), finder-found frames | **0.69%** | late stragglers only |
| any edge, finder-MISSED frames | 10.9% | not an edge effect at all |

* **Start edge:** in frames where the finder is right (±50 ns), *zero*
  signal hits fall before the window opening. The margin scan shows the
  knee at ≈35 ns; the chosen front=50 sits safely past it, and widening it
  further only admits background (red dashed curve).
* **End edge:** 0.69% of signal leaks past +300 ns in found frames. The scan
  knee is at ≈200 ns (the second burst peak at dt≈130 ns must stay inside);
  stretching back to 1000 ns recovers only ~0.2 more points of signal while
  more than doubling in-window background — the physics-important late
  component is better recovered by the outside-model path than by a wider
  window.
* **The dominant "edge" loss is fictitious:** in missed frames the burst
  sits at a random offset from the wrong window (flat red histogram over
  ±2000 ns), so its split into "front/back" is meaningless — it is pure
  finder failure (10.9% of all signal).

Bottom line: the window edges are ~99.3% effective on their own terms
(0.69% combined loss when the finder is right). Window *finding*, not
window *sizing*, is where all remaining signal loss lives.

## Failure modes & improvement candidates (not yet done)

* The finder is a single argmax — no second chance. Top-k candidate windows
  (k=3) + a burst *verifier* on each candidate would recover frames where
  the true burst ranks 2nd/3rd.
* Sub-bin regression head (predict offset inside the 25 ns bin) would tighten
  the 10 ns median error further; only worth it after top-k.
* Failed frames correlate with weak/low-multiplicity bursts (visible in the
  validation displays); some may be irreducible.

## Operational notes

Two infrastructure incidents hit this phase (both diagnosed and documented):
a killed writer left an un-unlinkable file on /data that hung every rebuild
attempt in uninterruptible D-state (`notes/lesson-data-mount-poisoned-file.md`
— dataset moved to `windows_v2`), and wedged `uv run` locks serialized later
commands (bypass: call `python` directly).

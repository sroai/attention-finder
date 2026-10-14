# C++ / ONNX Runtime porting spec — tracker-hit background-filter pipeline

Self-contained specification for re-implementing the phase-4/5 inference
pipeline (window finder + two hit scorers + thresholds) in C++ with ONNX
Runtime. All facts below were extracted from the python source and verified
by loading and running every ONNX file with onnxruntime 1.27.0
(CPUExecutionProvider). No python knowledge is required to implement this.

Verification date: 2026-07-14. Exporter: PyTorch 2.13.0.

---

## 0. Pipeline overview (per 2000 ns frame)

Input per frame: N tracker hits, each with
`x, y, z [mm]`, `t [ns]` (TOF-corrected hit time), and the raw ePIC
`system_id` (detector system integer from definitions.xml).

1. **Window finder** (`finder2d_v1`): build a 2-channel 64x88 image from all
   hits of the frame, run the CNN, argmax over the 88 output logits gives the
   25 ns time bin of the physics interaction.
   `t0_est_ns = -100.0 + 25.0 * argmax_bin`  (bin LEFT edge, not center).
2. **Window** = `[t0_est - FRONT, t0_est + BACK]` with
   **FRONT = 50.0 ns, BACK = 300.0 ns** (fixed constants).
   Hit is *inside* iff `dt >= -50.0 && dt <= 300.0` where `dt = t - t0_est`
   (both ends inclusive).
3. **In-window hits** -> attention model `watt_r2` (one call per frame with
   all in-window hits of that frame together — the model exchanges
   information between the hits, so do NOT mix hits from different frames
   and do NOT split a window across calls).
   **Out-of-window hits** -> per-hit MLP `mlp_v1_r3` (any batching is fine,
   hits are scored independently).
4. **Decision** (per hit): both models output a background probability in
   [0,1] (sigmoid already applied inside the ONNX graph).
   `keep = score < threshold` (strict `<`); `score >= threshold` => hit is
   background => filter out.
   Two thresholds: `thr_in` for inside hits, `thr_out` for outside hits
   (numbers in section 5).

The window is a *feature*, not a hard cut: outside hits are not dropped,
they just fall back to the per-hit model.

---

## 1. Model: window finder `finder2d_v1`

* ONNX file: `./runs_local/finder2d_v1/window_finder2d.onnx`
* **External data file REQUIRED**: `window_finder2d.onnx.data` (719104 bytes)
  must ship in the same directory as the .onnx (8 weight initializers point
  at it with relative location `window_finder2d.onnx.data`). The other two
  models are fully self-contained.
* opset 18, IR version 10 (onnxruntime >= ~1.15; verified on 1.27.0).
* File sizes: onnx 33132 B + data 719104 B; ~178k parameters.

### I/O (verified by running the session)

| tensor | name | shape | dtype |
|---|---|---|---|
| input | `image` | `[1, 2, 64, 88]` **fixed, batch=1 only** | float32 |
| output | `bin_logits` | `[1, 88]` | float32 |

The batch dimension is NOT dynamic (exported without dynamic axes) — one
frame per `Run()` call.

Output semantics: raw logits (no softmax) over the 88 time bins; the model
was trained with softmax + cross-entropy on "which bin contains true t0".
Deployment uses only `argmax`. If a confidence is wanted, apply softmax over
the 88 values.

### Image construction (preprocessing done in C++)

Time grid (shared with everything else in the pipeline):

```
BIN_NS  = 25.0 ns
T_MIN   = -100.0 ns,  T_MAX = 2100.0 ns
N_BINS  = 88
t_bin(t) = clamp( (int64)((t - T_MIN) / BIN_NS), 0, 87 )
```

(Truncation vs floor is irrelevant after the clamp: only t < T_MIN would
differ and it clamps to 0 either way.)

Spatial grids (64 bins each, out-of-range values CLIP into the edge bins —
this is intentional: e.g. RomanPots at z = 32 m land in the top z row):

```
N_SPACE = 64
Z_RANGE = [-2500.0, 8000.0] mm          (bin width 164.0625 mm)
R_RANGE = [0.0, 800.0] mm               (bin width 12.5 mm)
z_bin = clamp( (int64)((z - (-2500.0)) / 10500.0 * 64), 0, 63 )
r     = sqrt(x*x + y*y)
r_bin = clamp( (int64)((r - 0.0) / 800.0 * 64), 0, 63 )
```

Fill (all hits of the frame, no origin/system selection):

```
image[0][z_bin][t_bin] += 1     // channel 0 = (z, t) hit-count map
image[1][r_bin][t_bin] += 1     // channel 1 = (r, t) hit-count map
then elementwise: image = log(1 + image)      // natural log, applied AFTER counting
```

Tensor memory layout is standard NCHW contiguous:
`image[batch=1][channel=2][space=64][time=88]`, i.e. flat index
`((c * 64) + s) * 88 + tbin`. No further normalization — log1p counts go
straight into the graph.

### t0 from the output

```
bin      = argmax over the 88 logits
t0_est   = -100.0 + 25.0 * bin        // ns, LEFT edge of the winning bin
```

(Exactly what the python evaluation does: `logits.argmax(1) * BIN_NS + T_MIN`.)

Reference accuracy (finder2d_v1 metrics.json, multi-preset test):
within-25ns 86.0%, top-10 coverage 98.9%.

---

## 2. Model: in-window hit scorer `watt_r2` (WindowNet, 2 attention blocks, d=64)

* ONNX file: `/data/msf-2026-06/ai-results/runs/watt_r2/window_model.onnx`
  (316503 bytes, self-contained, opset 17).

### I/O (verified)

| tensor | name | shape | dtype |
|---|---|---|---|
| input | `hits` | `[n_hits, 4]` (n_hits dynamic) | float32 |
| input | `system` | `[n_hits]` | int64 |
| output | `score` | `[n_hits]` | float32 |

* `hits[i] = { x [mm], y [mm], z [mm], dt [ns] }` — **column 3 is
  `dt = t - t0_est`, NOT the raw time**. That subtraction is the only
  preprocessing; everything else (r = sqrt(x^2+y^2) as 5th feature,
  standardization, system embedding) is baked inside the graph.
* `system[i]` = **dense system index** (0..21), see section 4.
* `score[i]` = background probability, **sigmoid already applied** in-graph
  (higher = more background-like). No mask/padding input exists: the graph
  is the single-window path, full n_hits x n_hits self-attention over
  whatever you pass.

Call contract: one `Run()` per frame with exactly the in-window hits of that
frame (typical n_hits ~ 1200 for 18x275; dt in [-50, 300]). Cost is
O(n_hits^2); ~3 ms/1200-hit window single-thread CPU.

Normalization constants baked into the graph (for parity testing only — do
NOT re-apply). Feature order `(x, y, z, dt, r)`; from watt_r2's training
build (`wf_v2_r2` window_hits, recorded in its config.json):

```
mean = [-10.943440437316895, 0.14427055418491364, 76.47489166259766,
        118.74766540527344, 249.7185516357422]
std  = [232.4148406982422, 235.02938842773438, 2527.39404296875,
        99.47467803955078, 220.91590881347656]
```

Torch-vs-ONNX parity at export: max_abs_diff 4.59e-06.

---

## 3. Model: out-of-window hit scorer `mlp_v1_r3` (HitMLP 64-64-32, emb_dim 8)

* ONNX file: `/data/msf-2026-06/ai-results/runs/mlp_v1_r3/model.onnx`
  (32764 bytes, self-contained, opset 17).

### I/O (verified; identical contract for mlp_v1_r2)

| tensor | name | shape | dtype |
|---|---|---|---|
| input | `hits` | `[n_hits, 4]` (n_hits dynamic) | float32 |
| input | `system` | `[n_hits]` | int64 |
| output | `score` | `[n_hits]` | float32 |

* `hits[i] = { x [mm], y [mm], z [mm], t [ns] }` — **RAW frame time here**
  (no dt subtraction; the frame time distribution was part of training).
* `system[i]` = same dense index as watt_r2 (section 4).
* `score[i]` = background probability, sigmoid applied in-graph.
* Per-hit independent — batch size free, order irrelevant.

Normalization baked in (feature order `(x, y, z, t, r)`; from
`/data/msf-2026-06/ai-results/dataset/18x275_v3/meta.json`, train fold):

```
mean = [-4.744136810302734, 0.1873074322938919, -106.2083969116211,
        695.6661376953125, 242.22637939453125]
std  = [207.5590057373047, 223.3238067626953, 1087.27197265625,
        604.5661010742188, 209.90855407714844]
```

Torch-vs-ONNX parity at export: max_abs_diff 2.98e-07.

---

## 4. Dense system-index mapping (both hit scorers)

`system` input = position of the raw ePIC `system_id` in this exact sorted
22-element list (embedding vocabulary size = 22, values 0..21):

```
index:      0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21
system_id: 31  59  60  61  64  68  69  70  72  77  78  79  82  92 122 150 155 156 159 160 161 162
```

Names (for reference): 31 VertexBarrel_0; 59/60/61/64 TrackerBarrel_0/1/2/5;
68/69/70/72 TrackerEndcapN_0/1/2/4; 77/78/79/82 TrackerEndcapP_0/1/2/5;
92 BarrelTOF; 122 ForwardTOF; 150 B0Tracker_Station_1;
155/156 ForwardRomanPot_Station_1/2; 159-162 ForwardOffMTracker_station_1-4.

**Unknown-id handling: there is none in the model.** An index outside 0..21
is undefined behavior in the ONNX Gather (embedding lookup). The C++ side
MUST guard: a hit whose raw system_id is not in the list should be passed
through unscored (keep it) or mapped to a documented fallback — it was never
seen in training either way. Do not clamp silently without logging.

This list is also machine-readable in
`/data/msf-2026-06/ai-results/runs/mlp_v1_r3/deploy_info.json`
(`system_ids`, position = dense index).

---

## 5. Thresholds and decision rule

Rule (identical for both scorers):

```
keep_hit = (score < threshold)        // strict less-than
// score >= threshold  =>  background => filter out
```

`threshold = thr_in` if the hit is inside the window, `thr_out` otherwise.
Thresholds were calibrated on the validation fold by grid search: among
(eff_in, eff_out) splits reaching >= 99% TOTAL signal efficiency (window
losses included), pick maximum total background rejection.

### Operating points (most recent, r3-data calibration, 2026-07-13)

**Primary — argmax-t0 mode** (finder argmax -> t0_est; matches the pipeline
in section 0). Source:
`/data/msf-2026-06/ai-results/runs/watt_r2/metrics_end_to_end_r3base.json`
(win_run watt_r2, wf_run wf_v2_r3, outside_run **mlp_v1_r3**):

```
thr_in  = 0.9919785857200623
thr_out = 0.8168560862541199
test: signal_efficiency 0.98969, background_rejection 0.39706,
      kept_fraction_of_all_hits 0.60483
```

**Alternative — refined-t0 mode** (phase-5 refined t0 instead of raw argmax;
use ONLY if the C++ side implements t0 refinement). Source:
`.../watt_r2/metrics_end_to_end_r3refined.json` (same model trio):

```
thr_in  = 0.9958364367485046
thr_out = 0.7269938588142395
test: signal_efficiency 0.98961, background_rejection 0.52865
```

Older r2 calibration (watt_r2 + wf_v2_r2 + **mlp_v1_r2**, superseded; this
pair is what `window_model_deploy_info.json` still records):
`thr_in = 0.9846063256263733, thr_out = 0.8204870223999023`.
=> pairing answer: **mlp_v1_r3 pairs with the r3 thresholds above;
mlp_v1_r2 pairs only with the old 0.98461/0.82049 pair.** Do not mix.

Standalone mlp_v1_r3 threshold (no window pipeline, all hits through the
MLP): `0.900826632976532` (from its metrics.json / deploy_info.json).

**Calibration caveat**: the r3base/r3refined thresholds were calibrated with
the phase-3 finder `wf_v2_r3` producing t0_est, not with `finder2d_v1`
(no end-to-end calibration against finder2d exists yet as of this writing).
finder2d_v1 has comparable-or-better t0 accuracy, so r3base is the best
available operating point for the C++ port — but plan a recalibration pass
once the C++ pipeline runs.

---

## 6. Ideal-window mode (MC-truth t0) — for validation in C++

Training/evaluation truth definition:

```
t0_true = min over hits with prt_origin in {1, 2} of trk_hit_time
```

where `prt_origin`: 1 = signal particle, 2 = Geant4 secondary of signal
(3/4 = background, 0 = unknown/dropped). **Albedo hits are excluded** from
the min: an origin-1/2 hit is re-flagged as albedo (origin 5) when

```
(t_hit - prt_vtx_time > 100.0 ns)              // per-particle lateness
OR (t_hit - t0_event  > 300.0 ns)              // per-event lateness
```

with `t0_event` = earliest origin-1 hit time of the event (fallback:
earliest origin-1/2). `trk_hit_time` is already TOF-corrected toward the
origin, so no path/c term is subtracted. (In practice the albedo exclusion
rarely moves t0_true — albedo is by definition late — but implement it for
exactness.)

Ideal-window evaluation ("finder_true" mode in the python) uses the RAW
`t0_true` (NOT quantized to a 25 ns bin):
`inside = (t - t0_true >= -50.0) && (t - t0_true <= 300.0)`.
The bin quantization (`targets_from_t0` = `t_bin(t0_true)`) is only the
training target of the finder; when the finder runs, its argmax bin's left
edge is the estimate (section 1).

Frames with no signal hit have no t0 (dropped in training); in deployment
the finder always outputs some bin.

---

## 7. Gotchas checklist

1. `window_finder2d.onnx` needs `window_finder2d.onnx.data` next to it.
2. Finder input batch dim is fixed at 1; hit-scorer n_hits is dynamic.
3. `t0_est` = bin LEFT edge: `-100 + 25*argmax` (never add half a bin).
4. Window test is inclusive on both ends: `dt >= -50 && dt <= 300`.
5. watt_r2 wants `dt` in column 3; mlp_v1_r3 wants raw `t`. Same 4-column
   layout otherwise; r is computed inside both graphs — never pass 5 columns.
6. All normalization is baked into the graphs — feed raw mm/ns values.
7. Both hit-scorer outputs are probabilities (sigmoid applied); keep rule is
   strict `score < thr`.
8. One watt_r2 call per frame window (attention couples the hits);
   mlp calls batch freely.
9. Dense system index 0..21 via the sorted id list; out-of-vocabulary raw
   ids must be guarded in C++ (undefined behavior in the graph).
10. Thresholds are calibrated per (window-model, finder, outside-model,
    dataset) combination — mlp_v1_r3 => 0.99198/0.81686 (argmax mode);
    never combine mlp_v1_r2 with r3 thresholds or vice versa.
11. log1p (natural log) is applied AFTER count accumulation, on every image
    cell; out-of-range z/r/t hits clip INTO edge bins (they are not
    discarded) — dropping them changes the finder input.
12. Measured single-threaded CPU timings
    (`./reports/phase6_onnx_benchmarks.json`):
    finder2d 4.5 ms/frame, watt_r2 9.2 ms/1200-hit window,
    mlp_v1_r3 1.0 ms/5800 hits; summed session RSS ~10 MB — far under the
    2 GB budget.

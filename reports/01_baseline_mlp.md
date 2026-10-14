# Baseline report: per-hit MLP for tracker background filtering

> **v2 rerun:** with albedo relabeling (origin 5, don't-care; see reports/06)
> the retrained baseline `mlp_v1_r2` gives test 98.8% eff / 14.2% rejection —
> conclusions unchanged. Numbers below are the original v1-data run.

*Run: `mlp_v1` — 2026-07-12 — dataset `18x275_v1` (files 0022+0023, 10,000 events)*

## Task

Classify every tracker hit as **keep** (signal, `prt_origin` 1/2) or **filter**
(background, `prt_origin` 3/4) using only `x, y, z, time, system_id`
(`cell_id` reserved for later). Requirement: ≥99% signal-hit efficiency;
maximize background rejection at that point. Deployment target: ONNX on CPU,
well under 1 GB.

## Data (see plots/data/)

| | 0022 | 0023 |
|---|---|---|
| hits | 35,282,728 | 35,305,419 |
| events | 5000 | 5000 |
| signal fraction | 0.49% | 0.51% |
| hits/event (mean) | 7057 | 7061 |
| signal hits/event (mean) | 34.9 | 36.0 |
| `prt_origin = 0` | 0 | 0 |

Key observations (`0022_*.png`):

* **Severe imbalance**: ~1 signal hit per 200 background hits.
* **Detector system is highly informative**: central trackers
  (VertexBarrel, TrackerBarrel_*, TrackerEndcap*, BarrelTOF; ids 31, 59–92)
  are 0.07–2.6% signal; far-forward systems (B0Tracker, RomanPots,
  OffMomentumTrackers; ids 150–162) are 69–96% signal.
* **Hit time does NOT separate classes globally**: both classes are ~uniform
  over the 0–2000 ns integration window (`0022_features.png`). Only the
  leading edge differs slightly (`0022_time_zoom.png`).
* Position (r, z) shows moderate separation; `edep` would separate well but
  is not an allowed input.

## Setup

* **Split by event** (leakage-safe): 7000/1500/1500 events →
  49.4M / 10.6M / 10.6M hits (train/val/test), seed 42.
* **Model** `HitMLP`: system-id embedding (22×8) ⊕ standardized
  (x, y, z, t, r) → MLP 64-64-32-1 → background score. Normalization is baked
  into the graph, so ONNX consumes raw detector units. ~13k parameters.
* **Training**: weighted BCE (signal hits weighted n_bkg/n_sig ≈ 198),
  Adam 1e-3, batch 65k, 25 epochs ≈ 4 s/epoch (whole fold GPU-resident).
* **Protocol**: threshold fixed on *val* at 99% signal efficiency, applied
  unchanged to *test*.

## Results (test fold; plots/results/mlp_v1_*.png)

| operating mode | signal eff | bkg rejection | hits kept |
|---|---|---|---|
| global threshold 0.907 | **98.8%** | **14.1%** | 86.0% |
| per-system thresholds | **98.5%** | 6.2% | 93.8% |

AUC = 0.864. Training curves: `mlp_v1_training.png`; full efficiency/rejection
curve: `mlp_v1_eff_rejection.png` (rejection falls steeply above 95% eff —
the last percent of signal is expensive).

**Caveat 1 — threshold generalization**: 99.0% val efficiency became 98.8% on
test. If the 99% must hold strictly in production, calibrate with margin
(e.g. target 99.3% on val).

**Caveat 2 — global threshold hides local damage**: at the global operating
point the background-dominated inner tracker-endcap disks lose far more than
1% of their signal (`mlp_v1_per_system.png`):

| system | local signal eff | local rejection |
|---|---|---|
| 69 TrackerEndcapN_1 | 88.5% | 29.6% |
| 70 TrackerEndcapN_2 | 89.6% | 30.9% |
| 77 TrackerEndcapP_0 | 82.1% | 26.4% |
| 78 TrackerEndcapP_1 | 63.4% | 54.2% |

The per-system threshold mode (one threshold per detector, calibrated on val)
repairs this — every system ≥~99% locally — but overall rejection drops to
6.2%, showing that most of the global mode's rejection was bought by
sacrificing signal exactly in those systems. **Which trade-off is right is a
physics decision** (does downstream tracking rely on those endcap disks?); both
modes ship with the model (`deploy_info.json` + `per_system_thresholds` in
`metrics.json`).

## Deployment (runs/mlp_v1/model.onnx)

* ONNX file: **33 KB**; opset 17; inputs `hits float32 [N,4]` (raw mm/ns),
  `system int64 [N]` (dense index, map in `deploy_info.json`).
* torch↔onnxruntime-CPU parity: max |Δscore| = 3.6e-7 over 200k hits.
* CPU latency: **1.35 ms per 8000-hit event** (single session, default
  threads). Memory footprint is dominated by onnxruntime itself (tens of MB)
  — orders of magnitude under the 2 GB cap.

## Honest assessment & next steps

14% rejection at 99% efficiency is modest. The per-hit feature set is the
limiting factor, not model capacity or training length (loss and AUC plateau
by epoch ~15). Ranked next steps:

1. **Wider MLP + longer schedule** (`mlp_v2`) — cheap sanity check on
   capacity; expect at most a few points of rejection.
2. **`cell_id` granularity** — decode layer/module bits (needs the DD4hep
   readout spec per system) or hash-embed; gives sub-system geometry the
   MLP can't see. Moderate effort, plausible gain.
3. **Event-context model** — background hits are correlated within an event
   (beam-gas showers, synchrotron fans). A point-cloud/attention model over
   all ~7k hits of a frame is the physically motivated upgrade and the most
   likely route to a step change. Must be checked against the CPU budget.
4. **Second energy preset** — retrain/evaluate on the other beam config for
   robustness.

# attention-finder — finding the ep interaction in streaming ePIC frames, and filtering background hits

In streaming readout there are no triggers: the detector delivers a
continuous 2000 ns **frame** of ~7,000 tracker hits, of which ~34 belong to
the ep collision (0.5 %; the rest is machine background — synchrotron
radiation, beam-gas, their Geant4 secondaries). Two networks here solve the
two halves of that problem, using only quantities a reconstruction
framework has at inference time (`x, y, z, time, system_id`, plus
calorimeter energy for the newer models):

1. **the finder** — *when* did the interaction happen? A small 2D CNN reads
   the frame as a (z,t) + (r,t) occupancy image and picks the 25 ns column
   containing t0.
2. **the filter** — is this hit signal or background? A self-attention
   model scores the hits inside the found window, an MLP scores the rest;
   a score above threshold means "drop".

A third head, trained jointly with the finder, answers **what kind of
event** it was (k-Λ vs DIS/NC).

Everything is deployable as ONNX on CPU inside EICrecon (~1 MB weights,
~12 ms/frame), and the integration has been measured there.

## Results in one table

| question | model | measured on test | where |
|---|---|---|---|
| t0 within 50 ns, tracker only, 4 energy presets | finder + candidates + verifier (`t0_final_v2`) | **93.6 %** (18x275 95.1 / 10x130 97.5 / DVMP 87.0 / DIS-NC 84.8), top-20 ceiling 99.6 % | `reports/08_finder2d.md` |
| t0 within 50 ns, + calorimeter channels, 10x130 | 4-channel finder argmax (`p10_cal`) | **99.97 %** vs 92.0 % tracker-only — no verifier stage needed | `reports/09_cal_channels.md` |
| event type k-Λ vs DIS/NC | two-head `WindowTypeNet` (`p12bgx_cal`) | **99.40 %** accuracy, AUC 0.9996 (98.38 % tracker-only); window head 99.93 % recall@50 ns | `reports/10_event_type.md` |
| per-hit signal/background, frame-wide | `HitMLP` (`mlp_v1_r3`) | AUC 0.864, ~14 % background rejection at 99 % signal efficiency — the hard ceiling of per-hit features | `reports/01_baseline_mlp.md`, `notes/lesson-per-hit-feature-ceiling.md` |
| per-hit signal/background, inside the window | `watt` attention (`watt_all_v1`) | in-window AUC 0.993–0.996 across all four presets | `reports/05_window_attention.md` |
| cost and benefit inside EICrecon | ONNX filter in the tracking plugin | full reconstruction 607 → 278 ms/event, pure tracking 377 → 58 ms, filter itself ~12 ms; seeds /3.9, track purity 74 → 90 %, signal-track efficiency −1.3 pp | `reports/09_eicrecon_integration.md` |

Every number above is on a held-out test fold, split **by event**, with
thresholds calibrated on validation.

## Install

```bash
git clone https://github.com/sroai/attention-finder
cd attention-finder
uv sync                      # or: python -m venv .venv && .venv/bin/pip install -e .
```

`uv sync` installs PyTorch, polars, matplotlib, omegaconf, onnxruntime and
uproot into `.venv/`. A GPU is not required to evaluate the shipped models
(CPU inference is ~12 ms/frame); it is required in practice for training —
see "Hardware and time budgets".

## Where your data goes

Input is CSV produced by [**eic2ai**](https://github.com/sroai/eic2ai) from
ePIC simulation + reconstruction output, one folder per sample, e.g.

```
data/csv/18x275/0022.trk_hits.csv.zip      role trk_hits  — always needed
data/csv/18x275/0022.cal_hits.csv.zip      role cal_hits  — calorimeter models
data/csv/10x100/0001.reco_dis.csv.zip      role reco_dis  — event typing (truth Q²)
data/csv/10x100/0001.event_index.csv.zip   role event_index
```

`data/` and `results/` are gitignored; a symlink to a big disk is the usual
choice:

```bash
mkdir -p data && ln -s /my/big/disk/csv data/csv
```

Machine-specific paths live in **one** file, `configs/local.yaml`
(fully commented). Its defaults are repo-relative, so a fresh clone works
as-is. To use other locations, copy a profile and point at it:

```bash
cp configs/workstation-example.yaml ~/my-machine.yaml   # or farm-example.yaml
export TRKBKG_CONFIG=~/my-machine.yaml
```

The rule: **the profile holds machine facts, flags hold experiment knobs.**

One raw file is ~11 GB of CSV (~5000 frames) and converts to ~0.8 GB of
parquet. The full window-finding chain needs ~30 GB of datasets; the
per-hit path alone needs ~2 GB.

## The three paths

Each script's docstring says what goes in, what comes out, and why it
exists; the trainers and builders also list every knob under `--help`.
(Some analysis scripts load their dataset at import time, so their `--help`
only works once the data they read exists.) Run them from the repo root.

### A. per-hit filter (the simplest end-to-end loop)

```bash
PY=.venv/bin/python
$PY scripts/110_convert_csv_to_parquet.py 18x275     # CSV -> parquet (+ albedo labels)
$PY scripts/120_build_hit_dataset.py                 # event-split train/val/test
$PY scripts/210_explore_data.py                      # first plots and stats
$PY scripts/310_train_hit_classifier.py my_run --smoke   # 1 epoch: fails fast
$PY scripts/310_train_hit_classifier.py my_run --epochs 25   # as mlp_v1_r3 was trained
$PY scripts/410_evaluate_hit_classifier.py my_run    # test metrics + plots
$PY scripts/510_export_hit_model_onnx.py my_run      # deployable model
```

`120_build_hit_dataset.py` is hardwired to the two 18x275 files the
published dataset was built from (`0022`, `0023`) — edit the dict at the
top of it for your own files.

### B. the finder (the window-finding chain)

The per-hit classifier plateaus at ~14 % rejection; rejection lives in
knowing *when* the interaction happened. Tracker-only, this takes a
shortlist and a verifier:

```bash
# the finder itself (131 reuses the frozen 18x275 fold assignment that 130
# writes, so run path A's 110 + 120 and then 130 first)
$PY scripts/130_build_window_features_1d.py                 # 1D features + the frozen folds
$PY scripts/131_build_window_images_2d.py                   # frames -> 2-channel images
$PY scripts/321_train_window_finder_2d.py finder2d_v1       # all defaults: 40 epochs, width 32, 3 blocks
$PY scripts/421_evaluate_window_finder_2d.py finder2d_v1    # recall@25/50/100 ns, top-k ceiling

# per-hit attention inside the window (the candidate scorer)
$PY scripts/141_build_window_hits_all_presets.py            # true-t0 windows + hard negatives
$PY scripts/330_train_inwindow_attention.py watt_all_v1 --data window_hits_all_v1 \
    --epochs 12 --windows-per-batch 8
$PY scripts/426_watt_preset_audit.py watt_all_v1            # per-preset in-window AUC

# shortlist of 20 candidate windows, hand-feature reranking, learned verifier
$PY scripts/150_build_candidate_windows.py --k 20 --finder finder2d_v1 \
    --win-run watt_all_v1 --out-name cand_windows_v2
$PY scripts/350_rerank_candidates_hand_features.py --k 20 --finder finder2d_v1 \
    --win-run watt_all_v1 --out-name t0_rerank_all_v2
$PY scripts/340_train_window_verifier.py wver_v3 --data cand_windows_v2 \
    --blocks 3 --dim 96 --epochs 14 --batch 32
$PY scripts/351_select_windows_combined.py --wver wver_v3 \
    --rerank-name t0_rerank_all_v2 --cwin-name cand_windows_v2 --out-name t0_final_v2

```

Those are the run names and flags of the shipped checkpoints, read back
from their `config.json` (`reports/08_finder2d.md` §9 tells the story) —
pick your own names if you do not want to overwrite them.

To score hits end-to-end with an estimated t0 rather than the truth, use
`430_evaluate_end_to_end.py`. Mind its wiring: it builds `--wf-run` as the
**1D** `WindowFinderCNN` (so pass `wf_v2_r3`, not a 2D finder), reads the
window geometry and normalization from the `window_hits_v3` dataset
(step 140), and takes `--t0-dir` as a full path to a directory of
`{fold}.npz` t0 estimates:

```bash
$PY scripts/140_build_window_hits.py wf_v2_r3               # window_hits_v3
$PY scripts/430_evaluate_end_to_end.py watt_r2 --wf-run wf_v2_r3 \
    --outside-run mlp_v1_r3 --t0-dir <results>/dataset/t0_final_v2
```

That combination has not been re-measured with the newer `t0_final_v2`
selector (`reports/08_finder2d.md` §9 lists it as not done); the published
end-to-end numbers in `reports/07_phase5_verifier.md` are for the earlier
selector.

With **calorimeter channels** the shortlist and verifier become
unnecessary — the finder's top-1 already reaches the old top-5 ceiling:

```bash
$PY scripts/111_convert_cal_csv_to_parquet.py 10x130            # cal_hits -> parquet
$PY scripts/132_build_window_images_2d_cal.py 10x130 --out-name images2d_p10
$PY scripts/321_train_window_finder_2d.py p10_cal --dataset images2d_p10 \
    --per-sample-roll --smooth 0.05
$PY scripts/321_train_window_finder_2d.py p10_trk --dataset images2d_p10 \
    --channels 2 --per-sample-roll --smooth 0.05      # the tracker-only A/B arm
$PY scripts/421_evaluate_window_finder_2d.py p10_cal --dataset images2d_p10
$PY scripts/522_finder2d_latency.py --runs p10_trk p10_cal \
    --dataset images2d_p10                                      # ms/frame, CPU and GPU
```

### C. event typing (finder + class head)

```bash
# kinematics (truth Q², x) for both classes, then the two-class image sets
$PY scripts/112_convert_kinematics_to_parquet.py 10x100
$PY scripts/134_build_event_type_dataset_bgx.py --klambda 10x100 \
    --dis 10x100_minQ2_1 10x100_minQ2_10 10x100_minQ2_100

$PY scripts/322_train_event_type_2d.py p12_cal --dataset images2d_p12bgx \
    --per-sample-roll --smooth 0.05
$PY scripts/322_train_event_type_2d.py p12_trk --dataset images2d_p12bgx \
    --channels 2 --per-sample-roll --smooth 0.05
$PY scripts/422_evaluate_event_type.py p12_cal --dataset images2d_p12bgx
```

(`133_build_event_type_dataset.py` builds the plain, non-exchanged version
of the same frames — keep it only as the control that shows the trap.)

**Use the background-exchanged dataset.** Trained on the plain one, the
class head scores ~100 % by recognizing the *production campaign* rather
than the physics (a pure-background control still scored 99.8 %). Step 134
rebuilds every frame as "signal image + background image of a random frame
of either class", which makes background class-neutral by construction.
The story is in `reports/10_event_type.md` — it is the single most
important methodological warning in this repo.

## Shipped models

`models/` holds the checkpoints the published numbers come from. Your own
runs land next to them (`models/<run_name>/`), and git ignores them.

| run | what it is | files | produced by |
|---|---|---|---|
| `finder2d_v1` | 2-channel window finder, 4 presets | `best.pt`, `window_finder2d.onnx(.data)` | 321 |
| `watt_all_v1` | in-window attention, all presets + hard negatives | `best.pt` | 330 |
| `wver_v3` | candidate verifier (3 blocks, dim 96) | `best.pt` | 340 |
| `t0_final_v2` | combined selector — the shipped t0 estimator | `selector.pt`, `metrics.json` | 351 |
| `mlp_v1_r3` | per-hit MLP, the EICrecon out-of-window filter | `best.pt`, `model.onnx`, `deploy_info.json` | 310/510 |
| `wf_v2_r3` | 1D window finder, the first EICrecon finder | `best.pt`, `window_finder.onnx` | 320/511 |
| `watt_r2` | in-window attention as deployed in EICrecon | `best.pt`, `window_model.onnx` | 330/512 |
| `p10_cal` / `p10_trk` | 4-channel (calorimeter) and 2-channel finder, 10x130 A/B | `best.pt` | 321 |
| `p12bgx_cal` / `p12bgx_trk` | two-head finder + event type, background-exchanged data — the published event-type numbers | `best.pt`, `eval_frames.npz` | 322 |
| `p12_cal` / `p12_trk` | the same nets trained on the *plain* dataset — kept as the control that shows the campaign-fingerprint trap (99.98 % accuracy, physically meaningless) | `best.pt` | 322 |

A run directory is self-contained: `config.json` (including which profile
ran), `history.jsonl` (one line per epoch), `best.pt`, `metrics.json`, and
the ONNX export where one exists. `last.pt` is stripped from the shipped
copies.

Every checkpoint is stored as CPU tensors, so it loads on a machine
without a GPU. `reports/09_cal_channels.md` and `reports/10_event_type.md`
give the exact commands and hyperparameters behind the `p10*`/`p12*` runs
(~17 min per training on a T4; 40 epochs, 40–80k frames).

To evaluate a shipped model you need the dataset it was trained on — the
builders (1xx) rebuild those from CSV; the datasets themselves are too
large to ship.

## Checking your setup against the published numbers

The shipped checkpoints are stored as CPU tensors, so they load with or
without a GPU. Rebuild the dataset a model was trained on, re-run its
evaluation script, and compare against the `reports/*.json` file of the
same name — that is the cheapest proof that your data and environment
match ours. For example, re-running
`421_evaluate_window_finder_2d.py finder2d_v1` on a CPU-only machine
reproduced `reports/phase6_finder2d_v1.json` with 25 of 31 metrics
bit-identical and the rest within 0.15 pp (one or two frames, from
argmax ties resolved differently on CPU and GPU).

Evaluation scripts overwrite the `reports/*.json` and `plots/` files they
own, so `git diff` after a re-run is exactly the comparison you want.

## What the analysis results are

* `reports/*.md` — the research write-ups, in order: baseline MLP,
  architecture experiments, temporal analysis, window finder, window
  attention, albedo anatomy, verifier, the 2D finder, calorimeter
  channels, EICrecon integration, event typing. Each states its protocol,
  its numbers, and what was rejected.
* `reports/*.json` — the machine-readable metrics behind those numbers
  (several figure scripts read them directly).
* `reports/PLOTS_INDEX.md` — every figure, what it shows, which script
  makes it.
* `plots/` — the figures themselves. Per-event galleries (thousands of
  frames) are not shipped; the 2xx and 7xx scripts regenerate them.
* `notes/lesson-*.md` — things that were measured and did **not** work, or
  worked for a surprising reason. Read these before re-trying an "obvious"
  improvement; several of them cost days.
* `notes/cxx_porting_spec.md` — the full inference contract for the C++ /
  ONNX Runtime port (feature order, normalization, thresholds).

## Conventions that matter

* **Labels are 3-state**: y=0 signal (keep), y=1 background (filter),
  y=2 **albedo** — physically unreconstructable late signal debris
  (`prt_origin=5`, assigned at conversion by two timing rules). Albedo has
  weight 0 in training and is excluded from the efficiency denominator.
  Physics: `reports/06_event27_anatomy.md`.
* **The model score is a background probability**: ≥ threshold ⇒ filter out.
* **Thresholds are calibrated on validation, reported on test** — never
  tuned on test.
* **Split by event, never by hit.** Hits inside an event are correlated;
  splitting by hit leaks the answer.
* **Version suffixes are contracts.** `_v3` parquet = two-rule albedo
  labels; `images2d_v1` fixes the frozen 18x275 test fold every finder
  since the 2D work is compared on. Never overwrite a versioned directory
  in place — bump the version so results stay comparable.
* `prt_origin` itself comes from eic2ai (`src/origin.hpp`): 1/2 signal,
  3/4 background, 0 unknown; value 5 (albedo) is added by step 110.

## How the scripts are organized

The number says what KIND of step it is, the name says what it does:

| range | meaning | examples |
|---|---|---|
| 1xx | data preparation (builders) | `110_convert_csv_to_parquet`, `131_build_window_images_2d` |
| 2xx | understanding the data (no training) | `220_temporal_analysis`, `240_event_displays_colored` |
| 3xx | training | `310_train_hit_classifier`, `321_train_window_finder_2d` |
| 4xx | evaluation | `421_evaluate_window_finder_2d`, `425_coverage_at_k` |
| 5xx | export / deployment (ONNX, latency) | `510_export_hit_model_onnx`, `522_finder2d_latency` |
| 6xx | EICrecon integration studies | `610_eicrecon_tracking_compare` |
| 7xx | figures | `712_finder2d_illustration`, `721_selection_anatomy` |

Two rules keep this readable:

* **A script's logic lives in the script.** You can read
  `110_convert_csv_to_parquet.py` or `310_train_hit_classifier.py` top to
  bottom and see everything that happens. `trkbkg/` holds only genuinely
  shared code: label rules (`labels.py`), architectures (`models.py`,
  `attention.py`, `windows2d.py`, `verifier.py`), metrics (`evaluate.py`),
  dataset loading (`dataset.py`), the config loader (`config.py`).
* **Flags are for experiments, the profile is for machine facts.** No
  config include-chains, ever.

Reports written before 2026-07-15 cite the older script numbers (01…57);
`scripts/LEGACY_NUMBERING.md` decodes them. `trkbkg` is the historical
package name (the project was `ai-trk-bkg`) and is kept so the reports and
checkpoints stay valid.

### Scripts that need more than CSV

The CSV → training → analysis path above needs nothing else. These do:

* `250`, `252`, `253` (calorimeter event displays) read ePIC simulation /
  reconstruction **ROOT** files — pass `--input`; their defaults point at
  the original workstation.
* `610`, `611`, `612`, `620` run or analyse **EICrecon** with the filter
  compiled in; they need an EICrecon build with the ONNX models wired up
  (`reports/09_eicrecon_integration.md`,
  `notes/lesson-eicrecon-build-env.md`).

## Hardware and time budgets

| step | time | measured |
|---|---|---|
| 321 finder training, 40 epochs, 32k frames | ~17 min | Tesla T4, `reports/09_cal_channels.md` |
| 330 in-window attention | ~30 min | 12 GB card, `docs/running-on-a-farm.md` |
| 340 verifier | 2–3 h | same |
| 310 hit classifier, 49 M hits | ~1.4 min/epoch | CPU-only check of this repo |
| inference, one frame, 1 CPU thread | 10–12 ms | `reports/09_cal_channels.md`, script 522 |

The data-preparation steps (110/111/112, 13x) are CPU- and I/O-bound and
dominate the wall clock on a full sample — budget hours, not minutes, and
run them as array jobs.

Every builder skips outputs that already exist, and trainers write
`last.pt` every epoch: if a job dies at hour 7, resubmit and it
fast-forwards. `docs/running-on-a-farm.md` has a complete slurm example
("submit tonight, read results in the morning"); `docs/data-flow.md` maps
every dataset directory to what builds it and what consumes it.

GPU memory levers are per-trainer flags (`--batch`, `--batch-size`,
`--windows-per-batch`, `--cpu-data`); defaults fit 12 GB. All-preset
windows reach 6700 hits, so `330` batches by attention budget (B·L² ≤ 14M)
instead of a fixed batch size.

## Status

The finder, the filter and the event-type head are research code with
measured results, not a product. Known gaps, all documented in the
reports: the calorimeter finder is validated at one energy preset and one
seed; the two-head network has no ONNX export yet; EICrecon thresholds
were calibrated against the older 1D finder and need recalibration against
`finder2d`; the candidates+verifier stack is not ported to C++.

## License

LGPL-3.0-or-later, following the EIC software convention
(`eic/epic`, `eic/EICrecon`, `eic/EDM4eic`): see `LICENSE` and
`LICENSE.spdx`.

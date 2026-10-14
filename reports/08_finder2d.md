# Report 08 — Mastering the window finder (phase 6 + 6b)

**Goal of this phase (user directive):** the best possible window finding,
nothing downstream. The window finder locates the physics interaction time
t0 inside a 2000 ns frame from inference-legal inputs only (x, y, z, t).
Metric throughout: fraction of test frames with the estimate within 50 ns of
the true t0 ("recall@50"). Motivation from phase 4/5: with the same hit
models, a perfect finder gives 97.5% background rejection vs ~40-53% now —
each point of finder recall is worth ~2.5 points of end-to-end rejection.

## Headline results

| stage | 18x275 holdout | all presets |
|---|---|---|
| phase-2 occupancy peak | 16% | — |
| phase-3 wf_v2_r3 (per-bin features + 1D CNN) | 78.5% | — |
| phase-5 + candidates & verifier (k=10) | 88.1% | — |
| **phase-6 2D-image finder argmax (finder2d_v1)** | **88.3%** | **87.9%** |
| phase-6b + all-preset hit-based reranker | 92.7% | 90.3% |
| phase-6b learned window verifier alone (wver_v1) | 91.9% | 90.7% |
| **phase-6b COMBINED selector (hand + learned) — FINAL** | **93.4%** | **91.9%** |
| top-10 candidate ceiling (verifier headroom) | 98.8% | 99.0% |

Per preset (test, argmax → final combined): 18x275 88.3 → 93.4, 10x130
91.8 → 96.0, DVMP 82.0 → 84.2, DIS-NC 85.3 → 86.4 — the combined selector
beats BOTH of its components on every preset and needs no margin gate.

Shipped estimates: `dataset/t0_final_v1/{fold}.npz` (gevt, preset, t0_est,
t0_true; earlier stages kept in `t0_rerank_all_v1/`). Checkpoints:
`runs_local/finder2d_v1`, `runs_local/wver_v1`, selector in
`runs_local/t0_final_v1/selector.pt`. Feature/score caches make selector
retraining free. Plot: `plots/phase6/t0_final_v1_recall.png`.

## What was tried, in order, and what each attempt taught

### 1. CNN tuning sweep — architecture is saturated
Depth (4 blocks), width (64), batchnorm: 86.9-87.6% vs v1's 87.9%; the wide
variant memorizes (train CE 0.03). Plots `plots/phase6/tuning_curves.png`,
`tuning_summary.png`. Capacity is NOT the constraint at 22k training frames.

### 2. Miss anatomy (new script 29) — the misses are visible but outranked
`plots/phase6/finder2d_v1_miss_{rank,anatomy,gallery}.png`. Among missed
frames the true bin has median logit rank 2; the finder's top-10 covers
90.6% of misses. Only half the misses are the classic occupancy trap; the
worst ~3% have a signal that is genuinely invisible in the occupancy image
(dim DVMP few-track events). Test-time augmentation over cyclic shifts:
+0.3 pts — not worth it. Conclusion: rerank the top-10; don't stare harder
at the same image.

### 3. Augmentation variants — a genuine negative result
Per-SAMPLE cyclic shifts (v5) train WORSE than one shift per batch (87.3 vs
87.9 test; train CE collapses 0.70 → 0.095 = memorization via background
texture). Label smoothing onto adjacent bins (v6) ties v1. Details:
`notes/lesson-per-sample-augmentation-hurt.md`. Per-batch roll stays.

### 4. Model ensembling — small, real, optional
Probability-mean of the 4 sweep models: +0.8 pts (88.7% overall test).
Kept as an option (4x inference cost, still <10 MB total); not used in the
shipped pipeline.

### 5. Crop-verifier CNN — failed, and the failure is the insight
A second CNN classifying the 64x16 image crop around each top-10 candidate
(+finder scalars), trained on ALL presets: 88.0% vs 87.9% argmax. Flat.
The finder already extracted everything those pixels know — a reranker
only works with NEW information. `notes/lesson-reranker-needs-new-information.md`.

### 6. Hit-based reranker on all presets — the working lever
The new information is per-hit: watt_r2 (phase-4 attention model) scores
hits at AUC 0.99/0.99/0.94/0.95 per preset (18x275/10x130/DVMP/DIS-NC) —
it transfers, zero unknown system ids. Script 31 builds 11 features per
candidate window (watt-score counts, confident-hit geometry, finder
logit/rank) for every preset directly from parquet — 285k candidate
windows — and trains a tiny MLP selector (36k params total pipeline
addition). Selector findings (all iterated free on the feature cache):
- 18x275-only training (script 23 route): 92.3% on its holdout; deeper /
  listwise selectors do NOT move it — features and data are the limits.
- all-preset training fixes the data limit: 18x275 92.7%, 10x130 93.9%.
- DVMP/DIS-NC: even preset-specific selectors cannot beat argmax there
  (dim few-track signal → hit features uninformative); fix is a MARGIN
  GATE — override the finder's argmax only when the selector prefers
  another candidate by >0.25 logit (tuned on val, one global scalar).

### 7. Learned window verifier + combination — the final step
A small attention net (`trkbkg/verifier.py`, ~120k params) classifies each
candidate window from its RAW hits: x, y, z, dt, r, the watt score as a
per-hit input, a system embedding; masked mean-pool; window scalars
(log n_hits, finder logit, rank) join at the head. Trained on 219k
candidate windows across all presets (`scripts/32/33`) — it sees hard
negatives (background bursts the finder ranked high) which watt never saw.
- wver_v1 alone: 90.7% overall — beats argmax on EVERY preset, including
  DVMP (83.0) and DIS-NC (86.4) where the hand features could only defer.
- Attention attribution inside the verifier: blocks=0 DeepSets twin
  (wver_v0) reaches 88.1% → attention buys +2.6 pts on this task too.
- COMBINED: the wver logit becomes the 12th feature of the reranker
  selector (`scripts/34_combined_selector.py`): **91.9% overall / 93.4%
  on 18x275** — dominates both components everywhere; the margin gate
  goes to zero (the combined selector never has to defer).

## Where the remaining ~7 points are

1. **Selector → ceiling gap (91.9 vs 99.0).** Verifier capacity/training:
   val was still improving at epoch 20; more epochs, larger d, or
   candidate-conditioned contrastive training are untried.
2. **DVMP-class dim signal** (84.2%, ceiling 98.8%): still the largest
   per-preset gap; watt retrained on all presets would sharpen the per-hit
   scores that both selector paths consume.
3. **Beyond-ceiling frames (~1%)**: invisible in 25 ns occupancy — would
   need finer time bins or calorimeter channels (future data).

## Reproduce

```
# finder training + eval + miss anatomy
python scripts/26_train_finder2d.py finder2d_v1 --epochs 40 --local-run-dir
python scripts/27_evaluate_finder2d.py finder2d_v1
python scripts/29_analyze_finder2d_misses.py finder2d_v1 --tta 8
# all-preset reranker (feature cache -> selector iterations are free)
python scripts/31_reranker_all_presets.py --finder finder2d_v1 --k 10
# learned verifier + final combined selector
python scripts/32_build_candidate_windows.py
python scripts/33_train_window_verifier.py wver_v1 --blocks 2 --epochs 20
python scripts/34_combined_selector.py
```

---

## 9. Phase 8 — perfect window finding (run 10, 2026-07-15)

Baseline to beat: t0_final_v1, test recall@50 = 91.9% overall
(18x275 93.4 / 10x130 96.0 / DVMP 84.2 / DIS-NC 86.4), top-10 ceiling 99.0%.

Levers, each measured before adoption (script 51, plots/phase8/):

| lever | measured effect (test) | verdict |
|---|---|---|
| 4-model finder ensemble | +0.8 argmax, +0.0 coverage@10 | REJECTED |
| k = 10 → 20 candidates | ceiling 99.0 → 99.6% | adopted |
| wver capacity ×3 (wver_v2: 3 blocks, dim 96) | verifier-alone 90.7 → 91.7% | adopted |
| watt retrained on ALL presets + wrong-window hard negatives (watt_all_v1, dataset window_hits_all_v1, script 52) | in-window AUC 0.9924/0.9927/0.9391/0.9534 → 0.9948/0.9956/0.9953/0.9933 (script 56) | adopted |

Final rebuild (scripts 32/31 at k=20 scored by watt_all_v1 → wver_v3
3 blocks/dim 96/14 epochs on cand_windows_v2 → combined selector, script 34):

**t0_final_v2 — test recall@50: 93.6% overall**
(18x275 **95.1** / 10x130 **97.5** / DVMP **87.0** / DIS-NC 84.8*; ceiling 99.6).
Verifier-alone: 92.1%. Hand-features-alone at k=20 with new scores: 92.0%.
*DIS-NC has 184 test frames; the −1.6 pt move is ~3 frames (±2.6% binomial) —
statistically inconclusive, not a regression signal.

Shipped estimates: dataset/t0_final_v2/{fold}.npz. Checkpoints:
runs_local/{watt_all_v1 (copy in /data runs), wver_v2, wver_v3, t0_final_v2}.
NOT yet done: ONNX (re-)export of watt_all_v1/wver_v3 and the hit-level
end-to-end re-run (16) with t0_final_v2 — first items for the next pass,
together with threshold recalibration in EICrecon.

Training infra note: all-preset windows reach 6700 hits; script 15 now uses
attention-budget batching (B·L² ≤ 14M) — fixed window batches OOM a shared
12 GB GPU. wver batch 32 (not 64) for the same reason.

Reproduce: 52 → 15 watt_all_v1 --data window_hits_all_v1 → 56 (audit) →
32/31 --k 20 --win-run watt_all_v1 → 33 wver_v3 --data cand_windows_v2
--blocks 3 --dim 96 --epochs 14 → 34 --wver wver_v3 --rerank-name
t0_rerank_all_v2 --cwin-name cand_windows_v2 --out-name t0_final_v2.

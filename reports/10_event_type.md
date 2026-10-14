# Phase 12 — event-type classification: k-Λ vs DIS/NC

**Result: the two-head network (phase-10 backbone + window head + a
band-pooled class head) identifies the signal event type at 99.4 %
accuracy (AUC 0.9996) with calorimeter channels, 98.4 % tracker-only —
measured honestly, after a background-exchange construction removed a
campaign fingerprint that had made the naive result (100 %) meaningless.
Window finding is unchanged: 99.93 % recall@50 ns, replicating the
phase-10 calorimeter gain at 10x100 (96.3 % → 99.94 % on the plain
dataset).**

## Data

Class 0: k-Λ 10x100, 8 full msf files (~40k frames, phase-10 pipeline).
Class 1: ePIC campaign 26.07.0 `Bkg_Exact1S_2us` DIS/NC 10x100 — 405
file pairs (99 events each, ~40k frames) staged from JLab tape (jcache →
/cache; campaign dir `disnc-2026-07-background/` holds symlinks +
re-runnable `stage_and_link.sh`). Q² covered ~log-uniform 1–500 GeV² by
equal counts from minQ2 = 1 / 10 / 100 (minQ2=1000 excluded). Converted
with eic2ai (roles trk_hits, cal_hits, reco_dis, event_index) in the
**26.07 container** — 26.06 podio cannot read schema-81000 collections
(they silently vanish). Per-event truth Q²/x from the new `truth_*`
columns of reco_dis (`InclusiveKinematicsTruth`; the merge drops
generator `dis_*` frame params, so this is the only true-Q² source in
ANY background-merged file, msf included).

## Model and protocol (user decisions)

`WindowTypeNet` = phase-10 backbone + unchanged 88-column window head +
class head pooling backbone features over a −1/+6-column band around the
window (true column in training — teacher forcing — predicted column at
inference). Joint loss: window CE (roll augmentation + neighbour
smoothing 0.05) + 1.0 × class CE. 80k frames, 80/10/10 by frame seed 42,
classes balanced. A/B via `--channels`: 2 (tracker) vs 4 (+calorimeter).
Checkpoint: mean of val recall@50 and class accuracy. Scripts 133/134
(datasets), 322 (train), 422 (eval). Trained on an A800 (4-CPU sleeper
allocation).

## The campaign-fingerprint trap (and the fix)

First training on the plain dataset (133): 100.0 % accuracy tracker-only,
AUC 1.0 — and a control pooling on a PURE-BACKGROUND band 44 columns from
t0 still scored 99.8 %. The classifier identified the production campaign
(26.06 msf vs 26.07 cocktail: DIS frames carry 4.6 % more tracker hits —
7,382 vs 7,058 — and a different texture; a single occupancy cut alone
gives 78 %). Those numbers carry no physics.

Fix (step 134, "background exchange"): per-hit truth splits every frame
into signal (prt_origin ≤ 2) and background (> 2) images; each output
frame = signal image + background image of a random same-fold frame of
EITHER class (donor class fraction 0.493). Background is class-neutral by
construction. Same split, same seed — comparable frame-for-frame.

## Results (test, 8,031 frames, background-exchanged)

| metric | tracker-only | + calorimeter |
|---|---|---|
| class accuracy | 98.4 % | **99.4 %** |
| ROC-AUC | 0.9993 | **0.9996** |
| window recall@50 ns | 95.6 % | 99.93 % |
| pure-background control | **49.5 %** | **49.9 %** |

Eval-device note: the table holds the CPU fp32 evaluation — the
deployment-faithful numbers (production inference is CPU/ONNX). The A800
evaluation differed: cal model within ±0.1 % (99.51 % accuracy), but the
2-channel model moved more (97.8 % vs 98.4 % accuracy, 97.6 % vs 95.6 %
window recall) — the A800 computes matmuls in TF32 by default and the
tracker-only logit landscape is flat enough that reduced precision flips
many near-tie columns. Lesson: evaluate on the deployment device, and
pin torch.backends precision when comparing runs.

The control at coin-flip level is the validity certificate: the
classifier reads the event, not the frame's background. Calorimeter
channels cut the typing error 2.7× (1.6 % → 0.6 %) — energy topology
separates the classes better than track geometry alone.

Accuracy vs truth Q² (4-channel, Wilson 68 % intervals on the plot):
≥ 99 % everywhere except two thin bins (97.6 % at Q² ≈ 75, 95.0 % at
Q² ≈ 750 with 40 frames). Miss autopsy (eval_frames.npz): above Q² = 50
only 3 of 1,504 DIS frames are missed — one borderline (P(DIS) = 0.38)
and two CONFIDENTLY wrong (0.07 and 0.05) at extreme x (0.29 and 0.64,
Q² = 135 and 797): very-high-x DIS puts a leading baryon-like system
forward and genuinely resembles the exclusive k-Λ topology. So the dips
are mostly statistics with a small physics-flavored tail at extreme x.
Misses are asymmetric overall: 39 of 48 are k-Λ frames read as DIS.
Plots: plots/phase12/ (acc_vs_q2 with error bars, score distributions).

## Caveats / open

1. The exchanged background is a construction: production frames carry
   whatever real background exists. The classifier keys on signal shape,
   which should transfer — verifying on same-campaign classes (e.g. k-Λ
   re-produced under 26.07) remains the gold standard.
2. Albedo of the donor frame's event rides along in the exchanged
   background (origin 5 > 2) — mild, arguably realistic.
3. One seed, one preset pair; SIDIS/DVMP/multi-class next.
4. High-Q² bins are thin; a dedicated high-Q² sample would settle the
   dip.

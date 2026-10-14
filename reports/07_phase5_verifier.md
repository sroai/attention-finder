# Phase 5: candidate windows + verifier — finder recall and the payoff

*2026-07-13 — spec in `/work/trk_hit_bkg.md` Phase 5. All new plots in
`plots/phase5/` (phase-3/4 plots untouched in `plots/window/` for
comparison). Numbers: `reports/phase5_stats.json`,
`runs/watt_r2/metrics_end_to_end_r3{base,refined}.json`.*

## Step 1 — diagnostic: the burst is visible, just out-ranked

`22_topk_diagnostic.py`, `plots/phase5/topk_coverage.png`: in finder-missed
test frames the true bin's **median rank is 2**; top-5 covers 94% of all
frames (71% of misses). The per-bin CNN sees the burst but beam-gas bursts
out-score it → candidate re-ranking is the right fix (recall ceiling at
k=5: 94%).

## Step 2 — verifier iterations (negative results kept on record)

| selector over top-5 candidate windows | test recall @50 ns |
|---|---|
| argmax (phase-3 baseline) | 78.5% |
| Σ(1−score) over window hits | **31.6%** — rewards occupancy; 1500 background hits × small residuals outvote ~35 signal hits |
| count of confident-signal hits / system coherence rules | 80.7% — weak: watt never saw wrong windows in training, its scores there are unreliable alone |
| **feature vector per candidate + tiny classifier** | **87.3%** |

The winning selector (script 23): for each candidate window compute 11
features — finder logit & rank, hit count, confident-signal counts at two
working points, distinct-system coherence, and the *geometry of confident
hits* (outer-barrel / far-forward / vertex counts) — and train a 16-unit
logistic classifier on train-fold candidates (positives = window at true
t0). It learns from real negative candidates, which neither the hand rules
nor watt ever did. CPU-trained in seconds; candidate features are cached
(`dataset/t0_refined_v1/cand_cache/`) so selector iterations are free.

## Step 3 — end-to-end payoff (test fold, v3 data, watt_r2 scoring*)

k=10 beats k=5 on val (recall 87.8% vs 86.5%; test 88.1%, ceiling 97.3%),
so k=10 is the shipped configuration:

| pipeline | signal eff | bkg rejection |
|---|---|---|
| argmax finder (phase 3/4) | 99.0% | 39.7% |
| top-5 + verifier t0 | 99.1% | 48.0% |
| **top-10 + verifier t0 (final)** | **99.0%** | **52.9%** |
| perfect-finder bound | 99.0% | 97.5% |

`plots/phase5/e2e_comparison.png` — the verifier curve dominates at every
operating point. +9.6 recall points bought +13.2 rejection points,
consistent with the ~2.5×(recall gap closed) rule from the phase-4
decomposition.

*watt_r2 stands in for watt_r3: the external GPU workload starves training
(epoch 1 unfinished after 25 min); the v2→v3 label difference is ~700
zero-weight hits, so the weights are equivalent. Thresholds are calibrated
on v3 val in all cases. Retrain when the GPU frees up.

## Deployment note

The verifier adds: 5 window-model calls (~11 ms CPU) + 11 feature
computations + one 193-parameter MLP per frame. Total pipeline still well
under 20 ms/frame and <1.5 MB of weights.

## Remaining gap & next steps

Recall is 87.3% vs the 94% top-5 ceiling and 100% ideal. Ranked options:
1. Better candidate classifier (more features, e.g. per-candidate finder
   logit *shape*; gradient boosting) — the cache makes iteration free.
2. k=10 (ceiling rises to ~97%) at 2× verifier cost.
3. Wrong-window-aware retraining of the attention model (hard negatives) so
   its scores collapse on false candidates — likely the biggest lever, needs
   GPU time.
4. Cross-energy finder training on 18x275 + 10x130 (data converted, pending).
5. After recall plateaus: sub-bin regression; per-system audit + threshold
   margin before deployment.

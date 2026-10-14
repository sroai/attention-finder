# Plots index — every step/try and where its plots live

One line per artifact; read top-to-bottom to replay the whole study.
(All paths relative to ``.)

## Phase 1 — data understanding & per-hit baseline (reports/01, 02)

| step | plots |
|---|---|
| raw data composition, features, occupancy (per file) | `plots/data/0022_*.png`, `plots/data/0023_*.png` (origin counts, hits/event, per-system, features, time zoom, r-z occupancy) |
| mlp_v1 baseline (13k MLP) | `plots/results/mlp_v1_training.png`, `_scores.png`, `_eff_rejection.png`, `_per_system.png` |
| mlp_v2 capacity check (20×) | `plots/results/mlp_v2_*.png` (same set) — flat vs v1 ⇒ capacity not the limit |
| ec_v1 global-event-context check | `plots/results/ec_v1_*.png` — no gain ⇒ per-hit feature ceiling |
| run ranking | `reports/run_comparison.md` (table) |

## Phase 2 — temporal structure (reports/03)

| step | plots |
|---|---|
| per-frame displays, origin colors | `plots/temporal/events/0022_evt00NN.png` (50) |
| per-frame displays, generator-status colors | `plots/temporal/events_status/0022_evt00NN_status.png` (50) |
| signal timing (t0, durations, promptness) | `plots/temporal/signal_timing.png` |
| single-interaction check | `plots/temporal/signal_max_gap.png` |
| aligned dt distribution (the missed feature) | `plots/temporal/aligned_dt.png` |
| t0 recovery: occupancy peak fails | `plots/temporal/t0_recovery.png` |
| t0 recovery: score-weighted also fails | `plots/temporal/t0_recovery_scored.png` |

## Phase 3 — window finder (reports/04)

| step | plots |
|---|---|
| wf_v1 vs wf_v2 learning curves (coherence-feature value) | `plots/window/wf_training_comparison.png` |
| wf_v2 t0 error (signed; late = dangerous) | `plots/window/wf_v2_error_hist.png` |
| containment vs margins (phase-4 decision plot) | `plots/window/wf_v2_containment.png` |
| frame feature images + logits examples | `plots/window/wf_v2_examples.png` |
| 50 validation displays (est. window band vs true lines) | `plots/window/events/wf_v2_gevt*.png` |
| START vs END edge effectiveness | `plots/window/wf_v2_edge_signal_dt.png`, `wf_v2_edge_margins.png` |

## Phase 4 — in-window attention & end-to-end (reports/05)

| step | plots |
|---|---|
| attention vs no-attention learning curves | `plots/window/watt_training_comparison.png` |
| end-to-end eff/rejection curve vs phase-1 point | `plots/window/watt_v1_e2e_curve.png` (also `wmlp_v1_e2e_curve.png`) |
| in-window score separation | `plots/window/watt_v1_e2e_scores.png` |
| background budget, real vs PERFECT finder (oracle) | `plots/window/watt_v1_e2e_budget.png` |

## v2 rerun — albedo-aware data (reports/05 header table; runs *_r2)

All phase 1–4 outputs regenerated on `parquet/18x275_v2` (albedo = origin 5,
purple in event displays). Same filenames for data/temporal plots (updated
in place); window/e2e plots carry the r2 run names:

| step | plots |
|---|---|
| data plots incl. albedo bar in origin counts | `plots/data/*` (regenerated) |
| temporal event displays with albedo colored | `plots/temporal/events*/*` (regenerated) |
| mlp_v1_r2 baseline | `plots/results/mlp_v1_r2_*.png` |
| wf_v2_r2 finder eval + edges + window displays | `plots/window/wf_v2_r2_*`, `plots/window/events/wf_v2_r2_gevt*.png` |
| watt_r2 / wmlp_r2 end-to-end | `plots/window/watt_r2_e2e_*.png`, `wmlp_r2_e2e_curve.png` |
| budget bars real vs perfect finder (r2) | `plots/window/watt_r2_e2e_budget.png` |
| learning-curve comparisons (r2) | `plots/window/wf_training_comparison.png`, `watt_training_comparison.png` |

## Phase 5 — finder recall via candidate verifier (reports/07; plots/phase5/)

| step | plots |
|---|---|
| top-k diagnostic (burst visible but out-ranked) | `plots/phase5/topk_coverage.png` |
| verifier recall, argmax vs classifier selection | `plots/phase5/verifier_recall.png` |
| end-to-end curves, phase 3/4 vs phase 5 | `plots/phase5/e2e_comparison.png` |
| e2e detail plots for the two variants | `plots/phase5/watt_r2_e2e_*_r3base.png`, `*_r3refined.png` |

## Phase 6/6b — 2D-image finder + all-preset reranker (reports/08; plots/phase6/)

| step | plots |
|---|---|
| finder2d_v1 vs phases 2/3/5 + per-process bars | `plots/phase6/finder2d_v1_comparison.png` (same pattern per variant) |
| CNN tuning sweep (depth/width/bn saturate) | `plots/phase6/tuning_curves.png`, `tuning_summary.png` |
| miss anatomy: true-bin rank, occupancy trap, brightness | `plots/phase6/finder2d_v1_miss_rank.png`, `_miss_anatomy.png` |
| worst-miss gallery (invisible dim signals) | `plots/phase6/finder2d_v1_miss_gallery.png` |
| crop-verifier NEGATIVE result (no new info = no gain) | `plots/phase6/cropver_v1_selection.png` |
| all-preset hit-based reranker, per preset | `plots/phase6/t0_rerank_all_v1_recall.png` |
| learned window verifier (wver_v1) selection | `plots/phase6/wver_v1_recall.png` (v0 = DeepSets twin) |
| FINAL: combined selector (hand + learned) | `plots/phase6/t0_final_v1_recall.png` |
| 18x275-only reranker on finder2d candidates | `plots/phase5/verifier_recall.png` (overwritten by 23 --finder2d run) |

## Colored event displays (user request, run 7)

| step | plots |
|---|---|
| background by SOURCE (synrad amber, e-gas blue/teal/magenta, p-gas brown) + G4 secondaries by ancestry (green=of signal, red=of background) | `plots/temporal/events_color/0022_evt00NN.png` (50 events, script 35) |
| showcase: events RICHEST in each rare source (rare bands are 0.04% of hits) | `plots/temporal/events_color/showcase/` (14 events, e.g. 2591 = brems + 2 G4 showers) |
| 50 events around 0022/2231 — the multi-source event (brems + touschek + p-gas + 3 G4 showers vs a dim 13-hit signal) | `plots/temporal/events_color/around_evt2231/` (script 35 --around 2231) |

## Investigations (reports/06)

| step | plots |
|---|---|
| event 0022/27 anatomy (neutron albedo) | `plots/window/event27_anatomy.png` |

## Phase 7 — EICrecon integration (reports/09; plots/phase7/)

Six eicrecon runs (10k evts: 18x275 files 0022+0023 x unfiltered / AI-finder /
AI-ideal-window), plot set modeled on the tracking-WG decks (BkgImpact_PV,
ePIC_Tracking_ShyamKumar, beampipe_gold_coating). Scripts 40 (analysis) + 41
(runner); data /data/msf-2026-06/ai-results/eicrecon/.

| step | plots |
|---|---|
| SEEDING: load before/after | `plots/phase7/seeds_per_event.png` (578 -> 148 seeds/evt) |
| track multiplicity | `plots/phase7/tracks_per_event.png` |
| fit quality chi2/ndf | `plots/phase7/chi2_ndf.png` |
| measurements per track | `plots/phase7/nmeasurements.png` (4-hit fake spike /3) |
| track pT/eta | `plots/phase7/track_kinematics.png` |
| signal-track efficiency vs pT/eta | `plots/phase7/signal_efficiency.png` (96.8/95.6/96.8%) |
| efficiency vs eta in p bins (WG std) | `plots/phase7/eff_vs_eta_pbins.png` |
| purity vs eta in p bins (WG std) | `plots/phase7/purity_vs_eta_pbins.png` (1-5 GeV: 20 -> 60-90%) |
| dp/p resolution vs p (eta regions) | `plots/phase7/dpp_resolution.png` (unchanged) |
| DCA_r resolution vs pT (eta regions) | `plots/phase7/dca_resolution.png` (unchanged) |
| what tracks really are | `plots/phase7/track_truth_classes.png` (purity 74 -> 90%) |
| VERTEXING: N vertices/event | `plots/phase7/vertices_per_event.png` |
| tracks in vertex fit + vertex chi2/ndf | `plots/phase7/vertex_ntracks_chi2.png` |
| vertex residuals x/y/z | `plots/phase7/vertex_residuals.png` (tails suppressed) |
| vertex eff + resolutions summary | `plots/phase7/vertex_summary.png` (82.8 -> 87/89%; sigma_z 10 -> 6.5um) |

## Phase 7b — 26.07.0 DIS campaign generalization (reports/09; plots/phase7_dis/)

Same plot set as phase 7 on 6 files (594 frames) of pythia8 NC-DIS 10x275
minQ2=1000 + background, geometry 26.07.0, same binary/weights (scripts 40
`--dataset dis26` + 42). Extra: `plots/phase7_dis/vertex_perf_vs_nmc.png`
(vs-N_MC axis meaningful for DIS multiplicity).

| investigation | plots |
|---|---|
| background funnel: 55k particles -> 9 tracks, stage by stage (script 43; notes/stock-eicrecon-has-no-time-window.md) | `plots/phase7_dis/background_funnel.png` |
| ONNX filter cost vs whole-event reco CPU (script 44; reports/phase7_onnx_timing.json) | `plots/phase7/onnx_timing_context.png` (12/7 ms vs 607->278 ms) |

Numbers behind every plot: `runs/<run>/metrics*.json`, `reports/*_stats.json`,
`reports/phase7_tracking_compare.json`, `reports/phase7_tracking_compare_dis.json`.

## Slide figures (run 9 — built for the deck, but they are results in their own right)

| what it shows | plots |
|---|---|
| what the attention layer actually looks at: two query hits + attention received per hit; the burst is a landmark, not a voting bloc (script 45; reports/attention_maps.json; notes/lesson-attention-is-a-landmark-not-a-vote.md) | `plots/window/attention_maps.png` |
| the same, window by window: 50 pages, each with (a) signal-query links, (b) background-query links, (c) attention received = the network's saliency map, (d) the model's score vs the deployed threshold (script 49; per-window numbers in `attention_events/index.json`) | `plots/window/attention_events/window_*.png` |
| how the 2D finder sees a frame: the two image channels + the per-column softmax; the collision is invisible by eye, the CNN calls it at p=0.85 (script 47) | `plots/phase6/finder2d_illustration.png` |
| timing, one question per figure (script 46; same JSON as script 44, which is kept as-is) | `plots/phase7/time_full_reco.png` (607→278 ms), `plots/phase7/time_tracking.png` (377→58 ms), `plots/phase7/time_onnx_cost.png` (12 ms added) |
| memory footprint: what the models add (34 MB RAM, 1.05 MB weights, vs the 2 GB brief) and what the whole job peaks at (5.96 → 5.55 GB) (script 50; reports/phase7_onnx_memory.json) | `plots/phase7/memory_footprint.png` |
| the background funnel, three readable views — colour-coded particles / hits / tracking objects (script 48; counts cached by script 43 in reports/phase7_funnel.json) | `plots/phase7_dis/background_funnel_log.png` (countable log), `..._linear.png` (linear, 55 366 bar broken), `..._hits.png` (hits only: 8 935 tracker hits → 9 tracks) |

## Phase 8 — perfect window finding (run 10; plots/phase8/)

| investigation | plots |
|---|---|
| candidate ceiling: coverage@k (k=1..20) for finder2d_v1 vs 4-model ensemble on val+test; ensemble adds argmax but NOT ceiling, k=20 lifts ceiling 99.0 -> 99.6% (script 51; reports/phase8_coverage.json) | `plots/phase8/coverage_at_k.png` |
| candidate selection on real frames: hit z vs t + top-10 candidate bands (finder rank numbers), selector pick green, true t0 dashed; categories rescued / argmax-ok / still-missed (script 53; index in `events/index.json`) | `plots/phase8/events/*.png` |
| anatomy of ONE frame's decision, 4 stacked panels: both image channels -> per-column finder logits with top-k -> verifier score per candidate (script 55) | `plots/phase8/selection_anatomy_*.png` |
| per-preset in-window AUC of watt hit scores — the verifier's 6th feature; watt_r2 baseline 0.992/0.993/0.939/0.953 (script 56; reports/phase8_watt_audit.json) | `plots/phase8/watt_preset_auc.png` |
| FINAL phase-8 result: t0_final_v2 combined selector recall per preset (script 34; reports/phase6_t0_final_v2.json) | `plots/phase6/t0_final_v2_recall.png` (also wver_v3_recall.png, t0_rerank_all_v2_recall.png) |
| audience 2-preset variants for the deck (script 57, reads the JSONs) | `plots/phase8/finder_2presets.png`, `final_2presets.png`, `negatives_2presets.png` |

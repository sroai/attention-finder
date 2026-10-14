# Legacy script numbering (before the 2026-07-15 reorganization)

Old reports/notes/STATUS entries cite the old numbers; this table
is the decoder. Groups: 1xx data preparation, 2xx understanding
the data, 3xx training, 4xx evaluation, 5xx export/deploy,
6xx EICrecon studies, 7xx figures & slides.

| old | new |
|---|---|
| `01_convert_csv.py` | `110_convert_csv_to_parquet.py` |
| `03_build_dataset.py` | `120_build_hit_dataset.py` |
| `10_build_window_dataset.py` | `130_build_window_features_1d.py` |
| `25_build_images.py` | `131_build_window_images_2d.py` |
| `14_build_window_hits.py` | `140_build_window_hits.py` |
| `52_build_window_hits_all.py` | `141_build_window_hits_all_presets.py` |
| `32_build_candidate_windows.py` | `150_build_candidate_windows.py` |
| `02_analyze_data.py` | `210_explore_data.py` |
| `08_temporal_analysis.py` | `220_temporal_analysis.py` |
| `09_t0_from_scores.py` | `221_t0_from_scores_check.py` |
| `20_event27_anatomy.py` | `230_event27_albedo_anatomy.py` |
| `35_plot_events_color.py` | `240_event_displays_colored.py` |
| `04_train.py` | `310_train_hit_classifier.py` |
| `11_train_window_finder.py` | `320_train_window_finder_1d.py` |
| `26_train_finder2d.py` | `321_train_window_finder_2d.py` |
| `sweep_finder2d.sh` | `322_sweep_window_finder_2d.sh` |
| `sweep_finder2d_aug.sh` | `323_sweep_window_finder_2d_aug.sh` |
| `15_train_window_model.py` | `330_train_inwindow_attention.py` |
| `33_train_window_verifier.py` | `340_train_window_verifier.py` |
| `30_train_crop_verifier.py` | `341_train_crop_verifier.py` |
| `31_reranker_all_presets.py` | `350_rerank_candidates_hand_features.py` |
| `34_combined_selector.py` | `351_select_windows_combined.py` |
| `23_candidate_verifier.py` | `352_candidate_verifier_1d.py` |
| `05_evaluate.py` | `410_evaluate_hit_classifier.py` |
| `07_compare_runs.py` | `411_compare_hit_runs.py` |
| `12_evaluate_window_finder.py` | `420_evaluate_window_finder_1d.py` |
| `27_evaluate_finder2d.py` | `421_evaluate_window_finder_2d.py` |
| `28_finder_tuning_report.py` | `422_finder2d_tuning_report.py` |
| `29_analyze_finder2d_misses.py` | `423_analyze_finder2d_misses.py` |
| `22_topk_diagnostic.py` | `424_topk_diagnostic_1d.py` |
| `51_coverage_at_k.py` | `425_coverage_at_k.py` |
| `56_watt_preset_audit.py` | `426_watt_preset_audit.py` |
| `16_evaluate_end_to_end.py` | `430_evaluate_end_to_end.py` |
| `18_decompose_e2e.py` | `431_decompose_end_to_end.py` |
| `19_window_edge_analysis.py` | `432_window_edge_analysis.py` |
| `21_plot_step_summaries.py` | `433_plot_step_summaries.py` |
| `24_phase5_comparison.py` | `434_recall_progress_comparison.py` |
| `06_export_onnx.py` | `510_export_hit_model_onnx.py` |
| `13_export_window_onnx.py` | `511_export_window_finder_onnx.py` |
| `17_export_window_model_onnx.py` | `512_export_inwindow_model_onnx.py` |
| `36_onnx_export_benchmark.py` | `520_onnx_export_benchmark.py` |
| `50_memory_footprint.py` | `521_onnx_memory_footprint.py` |
| `40_eicrecon_tracking_compare.py` | `610_eicrecon_tracking_compare.py` |
| `41_run_eicrecon_compare.sh` | `611_run_eicrecon_compare.sh` |
| `42_run_eicrecon_compare_dis.sh` | `612_run_eicrecon_compare_dis.sh` |
| `43_background_funnel.py` | `620_background_funnel.py` |
| `44_onnx_timing_plot.py` | `621_onnx_timing_plot.py` |
| `46_timing_panels.py` | `622_timing_panels.py` |
| `48_funnel_views.py` | `623_funnel_views.py` |
| `45_attention_maps.py` | `710_attention_maps.py` |
| `49_attention_event_gallery.py` | `711_attention_event_gallery.py` |
| `47_finder2d_illustration.py` | `712_finder2d_illustration.py` |
| `53_candidate_event_displays.py` | `720_candidate_event_displays.py` |
| `55_selection_anatomy.py` | `721_selection_anatomy.py` |
| `57_two_preset_slide_plots.py` | `722_two_preset_slide_plots.py` |

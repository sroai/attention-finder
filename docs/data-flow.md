# Data flow: every dataset directory, what builds it, what reads it

All paths below are relative to the roots in your config profile
(`configs/local.yaml` or `$TRKBKG_CONFIG`). "npz per fold" always means
`{train,val,test}.npz` + `meta.json` (normalization, ids, provenance).

```
csv_root/<sample>/*.trk_hits.csv[.zip]
        │  110_convert_csv_to_parquet  (albedo rules applied here — v3)
        ▼
parquet_dir/<sample>_v3/*.parquet          14 columns + lateness, dt_event
        │
        ├─ 120_build_hit_dataset ────────► dataset_dir/18x275_v3/
        │      per-hit tensors, event-split 70/15/15      (npz per fold)
        │      consumed by: 310 train, 410 evaluate, 510 export
        │
        ├─ 131_build_window_images_2d ──► dataset_dir/images2d_v1/
        │      2-ch (z,t)+(r,t) 64x88 images, 1 per frame  (npz per fold)
        │      frame identity = gevt (file*1e6+evt, +preset*1e8)
        │      consumed by: 321 train finder, 421/423/425 eval, 150 builder
        │
        ├─ 140_build_window_hits ───────► dataset_dir/window_hits_v3/
        │      hits inside the ESTIMATED window, dt feature (npz per fold)
        │      consumed by: 330 train in-window attention ("watt")
        │
        ├─ 141_build_window_hits_all_presets ► dataset_dir/window_hits_all_v1/
        │      TRUE-t0 windows for all samples + wrong-window hard
        │      negatives in train — retrains watt across beam settings
        │
        └─ 150_build_candidate_windows ─► dataset_dir/cand_windows_v2/
               top-k finder candidates per frame, raw hits + watt scores
               consumed by: 340 train verifier, 351 combined selector

dataset_dir/t0_final_v2/{fold}.npz    ◄─ 351_select_windows_combined
        the SHIPPED per-frame t0 estimates (t0_est, t0_true, preset, gevt)
        consumed by: 430 end-to-end evaluation, 720/721 display scripts,
        EICrecon C++ port (thresholds + contract: notes/cxx_porting_spec.md)

runs_dir/<run>/          finished trainings: config.json, history.jsonl,
                         best.pt, metrics*.json, model.onnx, deploy_info.json
runs_local_dir/<run>/    live checkpoints during training (fast local disk)
```

Version suffixes are contracts: `_v3` parquet = two-rule albedo labels;
`images2d_v1` fixes the frozen 18x275 test fold that every finder since
phase 6 is compared on. Never overwrite a versioned dir in place — bump
the version (builders take flags / edit the OUT name) so results stay
comparable.

Sizes for planning: one 5000-frame sample file ≈ 11 GB CSV ≈ 0.8 GB
parquet; the 18x275_v3 hit dataset ≈ 1.5 GB; images2d_v1 ≈ 0.9 GB;
cand_windows_v2 ≈ 6 GB; a trained run directory ≈ 1–10 MB.

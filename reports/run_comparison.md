# Run comparison (test fold)

Sorted by background rejection at the global threshold (val-calibrated for target signal efficiency).

| run | model | epochs | test AUC | eff@global | rej@global | eff@per-sys | rej@per-sys |
|---|---|---|---|---|---|---|---|
| mlp_v2 | hit_mlp {'hidden': [256, 256, 128], 'emb_dim': 16} | 40 | 0.8639 | 0.9878 | 0.1439 | 0.9850 | 0.0623 |
| mlp_v1_r2 | hit_mlp {'hidden': [64, 64, 32], 'emb_dim': 8} | 25 | 0.8635 | 0.9876 | 0.1421 | 0.9840 | 0.0604 |
| mlp_v1 | hit_mlp | 25 | 0.8635 | 0.9881 | 0.1406 | 0.9846 | 0.0623 |
| ec_v1 | event_context {'hidden': [64, 64, 32], 'emb_dim': 8} | 40 | 0.8623 | 0.9878 | 0.1357 | 0.9825 | 0.0547 |

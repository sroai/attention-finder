# Architecture experiments: the per-hit feature ceiling

*2026-07-12 — follows `01_baseline_mlp.md`; comparison table in `run_comparison.md`*

## What was tried and why

Stated expectation before running (per protocol): baseline AUC plateaued at
0.865 by epoch ~15, so capacity was suspected NOT to be the bottleneck; the
physically-motivated hope was event context (background is correlated within
an event — showers, occupancy variation 6.5k–11k hits/event).

| run | idea | params | result (test, global thr) |
|---|---|---|---|
| mlp_v1 | baseline MLP 64-64-32, emb 8 | 13k | AUC 0.8635, rej **14.1%** |
| mlp_v2 | 20× capacity: 256-256-128, emb 16 | 273k | AUC 0.8639, rej **14.4%** |
| ec_v1 | DeepSets event context: per-hit encoder + event mean-pool | 40k | AUC 0.8623, rej **13.6%** |

## Conclusion

All three land within noise of each other (ΔAUC < 0.002). The inputs
`x, y, z, t, system_id` **per hit are exhausted at AUC ≈ 0.864 /
~14% rejection @ 99% signal efficiency**. Neither 20× capacity nor a global
per-event context vector adds information. This is consistent with the data
analysis: time is class-uniform, and position mostly encodes *which detector*
— which the system embedding already captures.

**Deployment recommendation: `mlp_v1`** — equal performance, smallest and
simplest (33 KB ONNX, 1.35 ms/event CPU). Already exported with
`deploy_info.json`.

## Where the next real gain must come from

1. **`cell_id` sub-detector bits** (layer/module: `system_id = cell_id & 0xFF`,
   the remaining 56 bits are per-system DD4hep fields). Needs the readout
   spec per system, or a data-driven bit-field probe. Gain uncertain —
   position already proxies layer geometry.
2. **Local (not global) event context** — ec_v1 pooled the *whole* event into
   one mean vector, which mostly measures occupancy. Background showers are
   *locally* clustered: k-nearest-neighbour / cylindrical-cell neighbourhood
   features (hit density around each hit in (z, φ, t)) are the physically
   right context. This is the recommended next experiment. ONNX/CPU cost must
   be watched (neighbour search per event).
3. **Cross-check labels' learnability ceiling**: a gradient-boosted tree on
   the same 5 features would confirm the ceiling cheaply and independently.

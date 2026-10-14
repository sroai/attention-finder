Attention pays off only after the temporal window shrinks the problem: in-window AUC went 0.865 (phase-1 features) -> 0.963 (adding dt) -> 0.991 (adding 2 self-attention blocks), and in-window rejection @99.9% eff went 20% -> 51% — but the same attention idea over the full 2000ns frame would have been 50x the compute aimed mostly at background-background pairs.

Measured on watt_v1 vs wmlp_v1 (identical except n_blocks) — always train the
0-block twin as the control, it isolates the architecture's contribution from
the feature's. Also: never hard-cut on a fallible upstream stage — the finder
misses 20% of frames, so out-of-window hits fall back to mlp_v1 scores and the
two thresholds are calibrated jointly on val for the 99% total-eff target
(end-to-end 39.3% rejection @ 99.0% eff on test). Train stage 2 on jittered
windows (U(-25,25) ns) so it tolerates stage-1 error.

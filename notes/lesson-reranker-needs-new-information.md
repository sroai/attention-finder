Reranking the window finder's top-k works ONLY when the verifier brings
information the finder's input didn't contain — a CNN re-reading the same
occupancy image gains nothing; per-hit watt scores gain +4 points.

Measured (2026-07-13/14, phase 6b, finder2d_v1 top-10, 18x275 test holdout):
- crop-verifier CNN (64x16 image crop around each candidate + finder
  scalars, trained on ALL presets, 219k candidates): 88.3% -> 88.7%. Flat.
- hit-based reranker (11 features from watt_r2 per-hit scores, logistic
  selector, trained on 18x275 only): 88.3% -> 92.3%.

Why: the finder is already a CNN over exactly those pixels — its top-10
ranking already reflects everything the occupancy image knows. The watt
scores come from hit-level training (millions of labeled hits, fine
geometry + system identity), which the 64x88 log-occupancy image discards.

Corollary that made the all-preset reranker possible: watt_r2, trained on
18x275 hits only, transfers — true-window signal-vs-background AUC 0.99
(10x130), 0.94 (DVMP), 0.95 (DIS-NC), zero unknown system ids
(scratchpad watt_transfer_check.py).

How to apply: before building any verifier/reranker stage, ask what NEW
evidence it sees; if the answer is "the same input, looked at harder",
skip it.

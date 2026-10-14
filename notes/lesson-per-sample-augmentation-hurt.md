Per-SAMPLE cyclic time-shift augmentation trained the 2D finder WORSE than
one shared shift per batch — more augmentation "diversity" enabled
memorization instead of preventing it.

Measured (2026-07-13, w32 b3 arch, images2d_v1):
- per-batch shift (finder2d_v1): train CE plateaus ~0.70, best val@50 88.2%.
- per-sample shifts (finder2d_v5_roll): train CE collapses to 0.095, val
  peaks 87.4% at epoch 28 then degrades. Test 87.3% vs v1 87.9%.
- label smoothing on adjacent bins did not rescue it (v6_rollsmooth).
- the gather-based per-sample roll was unit-tested equal to torch.roll —
  this is a real effect, not a bug.

Interpretation: the net is fully convolutional over time, hence largely
shift-equivariant; a shared batch shift adds the same regularization at
lower optimization freedom, while independent shifts let the optimizer use
frame-specific background texture as a landmark (train CE 0.095 means it
"finds" the burst almost perfectly on train frames it cannot generalize).
Also note t0 is already uniform across frames, so the augmentation adds
less genuine diversity than it appears to.

How to apply: for translation-equivariant nets with already-uniform target
positions, don't assume finer-grained augmentation is better; check train
CE — a collapse well below the batch-shift baseline is the memorization
signature.

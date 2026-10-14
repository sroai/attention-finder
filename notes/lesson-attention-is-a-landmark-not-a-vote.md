# Attention does not "vouch" — it triangulates against the burst

Measured (script 45, watt_r2, 300 test windows), not assumed:

- Every hit — signal AND background — spends ~58% of its attention mass on
  the ~3% of window hits that form the collision burst: 18.5x chance for
  signal queries, 21.9x for background ones. Background hits actually look at
  the burst slightly HARDER than signal hits do (0.63 vs 0.53 of their mass).
- Attention received: burst hits get ~9x uniform attention, background hits
  0.05x — a ~200x ratio. Nobody ever labels the hits for the attention layer;
  it finds them.

So the story told in earlier slides and in the phase-4 write-up — "hits on the
same track up-weight each other, a lone background hit gets no votes" — is
wrong as a description of the mechanism. The burst is a LANDMARK the whole
window triangulates against; the per-hit head then classifies a hit by whether
it is compatible with that landmark. That is also why attention only works
INSIDE the window (notes/lesson-attention-needs-window.md): over the full
2000 ns frame there is no single landmark to lock onto.

Checked and discarded as a discriminator: attention-weighted distance to the
hits a query listens to. Signal hits do reach shorter (median 0.83 m vs
1.94 m), but the distributions overlap heavily — it is reported in
reports/attention_maps.json, not used as evidence.

Why it matters: the intuition drives what we build next. "Neighbours vouch for
each other" argues for local/kNN graphs; "everyone triangulates against a
global landmark" argues for exactly what we have — full self-attention over a
window, with the window (the landmark's location) being the thing worth
investing in. It also explains why the finder, not the hit classifier, is the
bottleneck.

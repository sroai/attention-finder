Origin-1/2 "signal" includes unreconstructable neutron-albedo hits (~1.2%); marking needs TWO rules — per-particle lateness (t_hit - prt_vtx_time > 100 ns) ALONE MISSES albedo whose persisted parent was itself created late (capture products get their own late vtx_time and look prompt); add the per-event rule t_hit - t0_event > 300 ns. And trk_hit_time is TOF-corrected toward the origin, so do NOT subtract a path/c term.

Why the convention check mattered: the naive lateness (with |r|/c subtracted)
went negative by up to -122 ns for far-forward hits, which is how the
TOF-correction was discovered (ZDC hits read exactly |r|/c early). Tolerance
100 ns sits above the +-25 ns prompt core AND the 30-120 ns slow-secondary
shoulder; validate on the bimodal lateness histogram before changing it.
3-state labels (y=2 albedo): weight 0 in loss, out of the efficiency
denominator, filtered fraction reported. Effect of relabeling on 18x275:
in-window rejection at the operating point 49% -> 67.5%, back-edge loss
0.69% -> 0.48%, headline e2e 39.3% -> 39.9% @ ~99% eff. Case study:
reports/06_event27_anatomy.md (K+ nuclear stop, ~3 keV neutron, +760 ns).

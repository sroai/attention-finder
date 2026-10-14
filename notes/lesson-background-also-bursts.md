Background is NOT temporally uniform at frame scale: beam-gas interactions produce their own time-bursts with per-hit properties indistinguishable from the signal burst — t0 cannot be found by occupancy peaks (21% success) or score-weighted peaks (21%); only collective burst GEOMETRY (full radial span to r~750, coherent far-forward hits at z~6m, radiating from origin) separates the ep-collision burst from beam-gas bursts.

Measured in phase 2 (reports/03_temporal_analysis.md, plots/temporal/):
signal burst median span 76 ns, 99.3% of signal within t0+200ns, purity in
[t0, t0+25ns) is 18.3% vs 0.5% baseline. Perfect-t0 time-cut alone ~ 89%
rejection @ 99% eff, 6x better than the phase-1 per-hit ceiling (14%).
So phase-3 architecture must judge WINDOWS by aggregate geometry (burst
finder), not hits in isolation — per-hit scores are burst-type blind.
Also: 2/10000 frames have zero signal tracker hits; rare signal-secondary
stragglers out to 71 us exist (loopers/neutron capture) — treat as
background by construction when defining stage-1 targets.

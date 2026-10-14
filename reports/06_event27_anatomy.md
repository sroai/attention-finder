# Event anatomy: 0022/evt27 — why signal hits appear 760 ns "too late"

*2026-07-13 — triggered by the suspicious `wf_v2_gevt27.png`; investigation
scripts: manual queries + `20_event27_anatomy.py`; plot:
`plots/window/event27_anatomy.png`. Data: trk_hits parquet extras +
`0022.mc_particles.csv.zip` (streamed, event 27 only).*

## The suspicion

The window-validation display showed generator-level (origin-1) signal hits
in the CENTRAL detector at ~1650 ns — 780 ns after the interaction
(t0 = 878.3 ns) and long after the far-forward secondaries reached the
ZDC region (z≈33 m at ~980 ns). Nothing physical flies that slowly... unless
it does.

## What actually happened (from mc_particles)

The physics event is **e p → e′ K⁺ Λ** (the meson-structure channel):

| particle | energy | fate |
|---|---|---|
| e′ (pdg 11) | 12.8 GeV | backward disks, prompt hits 874 ns |
| K⁺ (pdg 321) | 10.5 GeV | prompt barrel pass 873–892 ns, then **nuclear stop at r≈898, z=1106 mm** (just behind BarrelTOF), 7 recorded daughters |
| Λ (pdg 3122) | 239.8 GeV | flies forward, decays; daughters hit RomanPots/OffM at z≈32.5–34.3 m at ~970–1000 ns |

The K⁺'s hadronic interaction at ~884–893 ns produced (recorded daughters):
two **neutrons** (KE ≈ 5.5 MeV and 114 MeV), a ²⁸Si nuclear fragment,
gammas, and an electron — i.e. a nuclear breakup right behind the TOF.

## The late hits are neutron albedo

The three origin-1/pdg-321 hits at 1633–1653 ns (r = 557–643 mm) are NOT the
kaon: a 10.5 GeV kaon crosses the whole tracker in <5 ns. Transport check:
kaon endpoint → late hit = ~604 mm in ~750 ns = **0.0027c ⇒ a ~3 keV
neutron** — exactly a moderated (multiply-scattered) albedo neutron from the
kaon's nuclear stop. The two BarrelTOF hits at 1652 ns sit at z=1099 mm,
i.e. directly under the kaon endpoint (z=1106 mm).

They carry the K⁺'s identity because Geant4/DD4hep attributes a tracker hit
to the nearest ANCESTOR that exists as an MCParticle: the specific slow
neutrons / capture products that made these deposits were below the
persistence cuts, so the hit points back to the generator kaon
(prt_origin=1, prt_status=1).

## Follow-up: albedo from SECONDARIES (0022/evt6) — second marking rule

The first marking rule (per-particle lateness > 100 ns) missed a second
albedo population, spotted in `0022_evt0006.png`: origin-2 hits ~1200 ns
after the burst whose median per-particle lateness is 0.9 ns. Mechanism:
when the neutron capture happens late, Geant4 persists the capture products
(electron/gamma) as their own MCParticles with a LATE creation time — hits
then look prompt relative to their own parent. Fix (v3 data): a hit is also
albedo if `trk_hit_time - t0_event > 300 ns` (t0_event = earliest origin-1
hit of the frame). Measured on 0022: 679 such hits/file escaped rule (a);
the dt band (200,300] ns holds only 18 of 175k signal hits, so the 300 ns
cut sits in a genuine gap between prompt physics and albedo.

## Implications for the pipeline (why this matters beyond curiosity)

1. **These hits are unreconstructable as track hits** — a 3 keV neutron
   deposit 760 ns late has no pointing relation to the ep vertex. Yet they
   are labeled "signal" (origin 1) and define the frame's `t_last`. Our ≥99%
   efficiency target implicitly demands keeping them; that is arguably
   over-strict. A cleaner label for future iterations: treat
   origin-1/2 hits with dt > some physics bound (e.g. 500 ns) as
   "don't-care" in the loss/metrics — worth raising as a physics decision.
2. This is the population behind the phase-2 observation of signal
   stragglers out to 71 µs and the 0.69% back-edge loss (script 19): it is
   irreducible by window sizing, and the outside-model fallback path is the
   right (only) way to keep them.
3. **Data quirk noted:** several prompt hits carry times 2–6 ns BEFORE the
   particle's vtx_time (e.g. K⁺ hit at 872.8 vs vtx 878.3), and hit times
   along the e′ trajectory are ordered opposite to flight direction. Hit
   times appear to include a detector-local correction (bunch/TOF
   reference?). Harmless at our 25 ns granularity but must be understood
   before any sub-ns timing feature is built. → check csv_convert
   definitions of trk_hit_time vs prt_vtx_time with the data owners.

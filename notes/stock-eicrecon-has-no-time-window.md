# Stock EICrecon tracking uses NO time window anywhere — seeds are few because of helix-pointing combinatorics

Investigated (2026-07-14, run 8c) why unfiltered seeding yields only ~508
seeds/event from ~55k background MC particles and ~4k central rec hits.

Where time could enter and doesn't:
- SiliconTrackerDigi: integrates the WHOLE 2000 ns frame. No time cut; hits
  in the same cell are merged (earliest timestamp kept, charges summed).
  Thresholds: 0.54 keV silicon, 0.1 keV MPGD — cut only ~4% of sim hits
  (synrad hits that convert ARE above threshold).
- TrackSeeding (ACTS orthogonal seeder): consumes (x,y,z) space points only;
  even initializes every seed's track-parameter time to a FIXED 10 ns
  (TrackSeeding.cc:250) with a large timeError.
- TrackerMeasurementFromHits: stores hit time on the Measurement2D
  (meas2D.setTime), but the CKF fits only the 2 local coordinates — time is
  a passenger, never a constraint.

The per-event funnel (DIS 10x275 minQ2=1000 + background, file 1.0531):
  55,366 MC particles
   -> 3,754 particles leave >=1 central-tracker sim hit (93% never touch it:
      synrad photons absorbed in beam pipe or missing the Si/MPGD volumes)
   -> 7,604 sim hits -> 7,289 above digi threshold (-4%)
   -> 5,739 unique cells = raw hits (-21% same-cell merging over the frame)
   -> 3,975 rec hits; 3,835 inside seeder acceptance (33<r<440 mm,
      -1500<z<1700 mm) — acceptance is NOT the reducer
   -> 506 seeds  <- THE reduction: seeds are 3-hit triplets that must fit a
      helix with pT >~ 100-200 MeV pointing to |d0| < 3 mm, |z_vtx| < 250 mm,
      dphi < 0.085 between layers, plus seed-filter dedup. Isolated synrad
      hits are uncorrelated across layers and essentially never form valid
      triplets: ~3,800^3/6 ~ 1e10 possible triples -> 5e2 seeds.
   -> 9.1 CKF tracks (ambiguity-solved).

So the "window" that tames the background in stock EICrecon is GEOMETRIC
(beamline pointing + helix consistency), not temporal. That also explains why
our time-window filter helps seeding so much (508 -> 105 seeds, /4.8): it
removes the out-of-time hits that the geometry cuts alone still let through
as accidental triplets — combinatorics scale ~n^3 in local density.

Caveat: raw->rec is another -31% (5,739 -> 3,975) not explained by the 1:1
TrackerHitReconstruction; likely the MPGD strip/segmentation treatment or my
threshold assumption per collection — not load-bearing for the conclusion.

# Far-forward collections in EICrecon: B0 joins ReconstructedParticles, the Roman Pots do not

Traced in the /eic-workbench/eicrecon source (commit as built for our
ai_trkfilter work, 2026-07-14). Line numbers are from that tree.

## B0 tracker — a normal tracking subsystem, ends up in ReconstructedParticles

`src/detectors/B0TRK/B0TRK.cc` does only digitization + hit reconstruction
(`B0TrackerRecHits`). Everything else is in `src/global/tracking/tracking.cc`,
which treats B0 as a real tracking detector — ACTS measurements, its own CKF
pass, its own ambiguity solver:

    B0TrackerRecHits
      -> B0TrackerMeasurements                     (tracking.cc:332)
      -> B0TrackerCKFTracks (+ truth-seeded twin)  (tracking.cc:329-456)

and is then merged into the main particle flow:

    CombinedTracks = CentralCKFTracks + B0TrackerCKFTracks   (tracking.cc:489)
      -> TracksToParticles
      -> ReconstructedChargedWithoutPIDParticles             (tracking.cc:543)
      -> PID chain (global/pid_lut/pid_lut.cc)
      -> ReconstructedChargedParticles                       (pid_lut.cc:286)
      -> MatchClusters
      -> ReconstructedParticles                              (reco/reco.cc:79-88)

A truth-seeded twin exists throughout (`CombinedTruthSeededTracks` ->
`ReconstructedTruthSeededChargedParticles`). B0 tracks also feed vertexing via
`CentralAndB0TrackVertices` (tracking.cc:459-487). Net effect: a B0 track
becomes an ordinary edm4eic::ReconstructedParticle, indistinguishable in the
collection from a central-tracker one.

## Roman Pots / off-momentum — parallel branch, TERMINAL

The Roman Pots never enter tracking. `src/detectors/RPOTS/RPOTS.cc` runs a
TRANSFER-MATRIX reconstruction straight on the rec hits (no ACTS, no seeds, no
CKF) and emits its own edm4eic::ReconstructedParticle collections — two of
them, from two algorithms:

  * `ForwardRomanPotRecParticles`        PolynomialMatrixReconstruction (RPOTS.cc:148)
  * `ForwardRomanPotStaticRecParticles`  MatrixTransferStatic, older linear
                                         version (RPOTS.cc:50)

Both take `MCParticles` + `ForwardRomanPotRecHits` as input (MC truth supplies
the beam configuration, not the track). The off-momentum tracker is the same
pattern: `ForwardOffMRecParticles` (detectors/FOFFMTRK/FOFFMTRK.cc:46).

**Nothing in EICrecon consumes these three collections.** Grepped the whole
src/ tree: they are produced, written to the podio output
(services/io/podio/JEventProcessorPODIO.cc:238-240), and that is all. They are
NOT merged into ReconstructedParticles; an analysis has to pick them up by
name.

## Consequences for our AI hit filter

- `AiTrackerHitFilter` reads all 12 tracker rec-hit collections (incl.
  `ForwardRomanPotRecHits`, `ForwardOffMTrackerRecHits`, `B0TrackerRecHits`)
  only as CONTEXT for the finder image; it filters ONLY `CentralTrackingRecHits`.
- B0 is therefore untouched by us today — but it DOES flow into
  ReconstructedParticles, so extending the filter to B0 hits would change that
  collection. Worth knowing before anyone assumes the filter is central-only in
  its effects.
- Roman Pots / off-momentum are untouched and cannot be affected even
  indirectly: their reconstruction is a standalone matrix transform on their
  own hits.
- OPPORTUNITY: the far-forward reconstruction has no time cut either, and the
  Roman Pot matrix reconstruction is fed EVERY rec hit of the frame — 2000 ns
  of background included. The same window idea could plausibly pay off there;
  nobody is doing it (see also notes/stock-eicrecon-has-no-time-window.md).

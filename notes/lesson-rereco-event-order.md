# Local eicrecon re-recos do NOT preserve event numbering — align by EventHeader

The multithreaded local re-recos of k_lambda_18x275_5000evt_0022 wrote
events in their own order: `bkg_0022_ev1400.edm4eic.root` STARTS at sim
event 1400 (its "entry 8" = sim event 1408, the p-gas showcase) and even
swaps 1411/1410; an earlier ev1400 run (the one behind the
plots/phase9_cal_plan/csv displays) had yet another order. Any plot or CSV
labeled by entry index from those files can silently show a different
physical event than the same label elsewhere.

Consequences already caused: the phase-9 "bkg_0022_ev1400 event 8"
threshold and csv displays are different physical events under one label.

Rule: when an event identity matters, map entry → sim event through
`EventHeader.eventNumber` (script 253 does this with a RECO_ENTRY dict)
and name outputs by SIM event number. The campaign reco files
(/data/msf-2026-06/reco-background) are entry-aligned with their sim
files; only the local multithreaded re-recos are not.

# A JANA factory's parameter prefix is <plugin>:<tag> — moving the factory renames every -P flag

Found while restructuring the filter for the upstream PR (2026-07-14). The
factory tag never changed (`AiCentralTrackingHitFilter`), but moving it out of
our private `ai_trkfilter` plugin into the stock `tracking` plugin silently
changed every user-facing parameter:

    before:  -Pai_trkfilter:AiCentralTrackingHitFilter:windowMode=ideal
    after:   -Ptracking:AiCentralTrackingHitFilter:windowMode=ideal

With `jana:parameter_strictness=2` (the EICrecon default) an unrecognized -P
does not warn — it THROWS, and the run dies before processing a single event.
So a stale flag in a batch script looks like a crash, not a typo.

Updated together with the move: run_ai_eicrecon.sh, scripts 41 and 42,
reports/09, notes/lesson-eicrecon-build-env.md.

Related trap, same family: `-Pjana:nthreads=4` is NOT a parameter;
it is `-Pnthreads=4`. Same symptom — JException, zero events.

How to verify a move like this changed nothing: run N events before and after,
then compare the output collections EVENT BY EVENT after sorting on
EventHeader.eventNumber. With nthreads>1 the events land in the output file in
completion order, so a naive array comparison reports differences that are not
real. (Float sums also wobble at ~1e-7 from thread summation order — compare
integer counts and index hashes, not float sums.)

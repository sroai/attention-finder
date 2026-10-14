# Building/running modified EICrecon in this container: 4 traps

1. LD_LIBRARY_PATH order. /app/eicrecon/lib (stock install) is baked into the
   env; a locally built eicrecon loads its plugins but they resolve shared
   libs (libalgorithms_tracking.so, ...) from the STOCK path first ->
   ABI-mismatch segfault at factory construction/teardown with a confusing
   backtrace mixing /app and local paths. Always
   `export LD_LIBRARY_PATH=/eic-workbench/eicrecon-install/lib:$LD_LIBRARY_PATH`
   AND `JANA_PLUGIN_PATH=/eic-workbench/eicrecon-install/lib/EICrecon/plugins:/app/jana2/plugins`.
   Wrapper does both: /eic-workbench/run_ai_eicrecon.sh.

2. particle_flow plugin is broken at commit 82d25ea0 (references
   TrackClusterSubtractor whose .cc is not compiled into the plugin ->
   dlopen "undefined symbol"). Run with -Pplugins_to_ignore=particle_flow.

3. Container onnxruntime is 1.17: max ONNX IR version 9. torch 2.13 exports
   IR 10 (finder2d/wver) -> "Unsupported model IR version". Fix: load with
   python onnx, set model.ir_version = 9, save (opsets 17/18 themselves are
   fine); also inline external .onnx.data at that point (single-file model,
   parity diff 0.0). Deployed copies: /eic-workbench/ai-trkfilter-models/.

4. CollectionCollector_factory + optional variadic inputs ABORT on missing
   collections: VariadicPodioInput<T, true> pushes nullptr for a name that
   doesn't resolve, and the collector wraps inputs in gsl::not_null ->
   gsl::details::terminate ("terminate called without an active exception",
   no JException). Every collection name in a collector list must exist.
   Converter-era names that changed: TOFBarrel/TOFEndcapRecHits ->
   TOF*SharedRecHits; RICHEndcapNRecHits no longer exists (PFRICH raw only).

Also: JOmniFactory parameters are prefixed with the PLUGIN name:
-Ptracking:AiCentralTrackingHitFilter:windowMode=..., not
-PAiCentralTrackingHitFilter:... (the latter is silently "Unused").

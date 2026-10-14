OBSOLETE 2026-07-13: /data migrated from NTFS to btrfs (same NVMe as /work) — the D-state trap below can no longer occur. Kept for history; the generic diagnosis method (ps CPU-time + /proc/PID/wchan) is still valid.

If a writer process on /data is killed mid-np.savez, the leftover file can become un-unlinkable: any later unlink/rewrite hangs the caller in uninterruptible D-state (verified via /proc/PID/wchan = filename_unlinkat) — never delete-and-rewrite such a file; abandon the path and write to a fresh directory instead.

How it bit us: the first windows-dataset build died silently while writing
test.npz (zombie python, 188-byte file). Every retry that tried to unlink or
overwrite that file hung forever with ~0 CPU — which LOOKED like a slow
featurization. Diagnosis recipe that worked:
  ps -ef | grep python           -> tiny cumulative CPU = blocked, not slow
  cat /proc/PID/wchan            -> filename_unlinkat = stuck in filesystem op
  ls -l /proc/PID/fd             -> shows the culprit file
D-state processes cannot be killed; leave them and route around the file
(windows_v1 -> windows_v2). Also: `uv run` serializes on a project lock, so a
wedged instance makes every later `uv run` hang too — use python
directly when a prior uv process is suspect.

UPDATE (phase 6): the hazard also hits CHECKPOINT WRITERS — a training run
(watt_r3) wedged in D-state while torch.save'ing last.pt to /data and its
3.5 GB CUDA allocation stayed pinned forever, blocking the GPU for every
later job. Rule: training checkpoints go to LOCAL disk
(./runs_local/, flag --local-run-dir in script 26), copied to
/data/runs/ after training ends. Bulky immutable artifacts (parquet,
datasets) on /data are fine — it's concurrent small rewrites that wedge.

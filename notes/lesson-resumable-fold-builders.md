Any script that writes several large artifacts in sequence (per-fold npz, per-file parquet) must skip/validate already-written outputs — a build died mid-write of the last fold and the naive rerun would have redone 10 minutes of work; now 10_build_window_dataset.py validates each npz (np.load probe) and rebuilds only what's missing/truncated.

Context: the python child was killed silently while writing test.npz (left a
188-byte file and a zombie process; train/val were fine). Same pattern as
01_convert_csv.py's exists-check. Rule adopted: every multi-artifact builder
gets a per-artifact skip-if-valid check, not a single am-I-done flag.

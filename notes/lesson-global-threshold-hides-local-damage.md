A single global score threshold at 99% overall signal efficiency destroyed up to 37% of signal locally in background-dominated systems (78: 63% eff) — always check per-system efficiency, and report both global and per-system-threshold operating modes.

Why it happens: the loss and the global operating point both let the model
buy overall rejection by writing off the few signal hits in systems that are
99.9% background (69/70 TrackerEndcapN_1/2, 77/78 TrackerEndcapP_0/1).
Overall metrics look fine (98.8% eff); the damage is only visible in the
per-system table.

Trade-off measured on mlp_v1 (test fold): global threshold = 98.8% eff /
14.1% rejection but locally 63–90% eff in those endcap disks; per-system
thresholds = >=99% everywhere but only 6.2% overall rejection. Which mode is
right is a physics call — ship both (deploy_info.json carries the global
threshold, metrics.json the per-system table).

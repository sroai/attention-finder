With inputs (x,y,z,t,system_id) per hit, every architecture plateaus at AUC ~0.864 / ~14% bkg rejection @99% sig eff — don't spend more runs on capacity or global event pooling; the next gain needs NEW information (cell_id bits or local neighbourhood density).

Evidence (test fold, dataset 18x275_v1): mlp_v1 13k params AUC 0.8635;
mlp_v2 273k params AUC 0.8639; ec_v1 DeepSets global mean-pool AUC 0.8623.
20x capacity: +0.3% rejection; global event context: slightly worse.

Why it makes sense: time is class-uniform (2 microsecond integration window),
position mostly identifies the detector element, which the system embedding
already encodes. Global mean-pool context only measures event occupancy.
Background is LOCALLY clustered (showers/fans), so per-hit neighbourhood
density in (z, phi, t) is the physically-motivated next feature — not more
network.

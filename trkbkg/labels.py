"""Label conventions shared by all pipeline stages.

prt_origin (written by eic2ai, src/origin.hpp; value 5 is added later,
at conversion, by the albedo rules in scripts/110_convert_csv_to_parquet.py):
    0 = unknown            -> dropped from datasets (counted + reported)
    1 = signal particle    -> KEEP  (y = 0)
    2 = geant4 secondary of signal particle -> KEEP (y = 0)
    3 = background particle -> FILTER (y = 1)
    4 = geant4 secondary of background particle -> FILTER (y = 1)

We define the positive class y=1 as "background / filter out" so the model
output is naturally a "background score": higher = more likely junk.
The physics requirement is asymmetric: keeping signal hits matters more than
rejecting background, so operating thresholds are chosen on the score to
guarantee >=99% signal efficiency (see evaluate.py).
"""

ORIGIN_UNKNOWN = 0
ORIGIN_SIGNAL = (1, 2)
ORIGIN_BACKGROUND = (3, 4)
ORIGIN_ALBEDO = 5  # added at parquet conversion, NOT present in raw CSV:
# an origin-1/2 hit whose lateness = trk_hit_time - prt_vtx_time exceeds
# ALBEDO_TOL_NS. Physically: slow-neutron / capture products of a signal
# particle's nuclear interaction, attributed to the signal ancestor by
# Geant4 persistence cuts (case study: reports/06_event27_anatomy.md).
# Such hits are unreconstructable -> excluded from training loss and from
# the signal-efficiency denominator; reported separately.

ALBEDO_TOL_NS = 100.0  # per-particle rule: above the +-25ns prompt core and
# the 30-120ns slow-secondary shoulder.
ALBEDO_EVT_TOL_NS = 300.0  # per-event rule (t_hit - t0_event): catches albedo
# whose persisted parent was itself created late (0022/evt6); prompt physics
# ends by ~200ns, only 18/175k hits fall in (200,300]. Together the rules
# flag ~1.2% of origin-1/2 hits (18x275).

ORIGIN_NAMES = {
    0: "unknown",
    1: "signal",
    2: "signal-secondary",
    3: "background",
    4: "background-secondary",
    5: "signal-albedo",
}


def origin_to_label3(origin):
    """Vectorized origin -> y mapping (3-state).

    y = 0 signal (keep), 1 background (filter), 2 albedo (don't-care:
    excluded from loss and efficiency; reported separately).
    Caller must drop origin==0 rows *before* calling this.
    """
    y = (origin >= 3).astype("int8")
    y[origin == 5] = 2
    return y

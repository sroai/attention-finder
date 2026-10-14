#!/usr/bin/env python
"""Colored event displays: background BY SOURCE, G4 secondaries BY ANCESTRY.

Augmented version of plots/temporal/events/ (user request): the background
is not one gray blob — each cocktail source keeps its own color (status
band 2000 synrad = amber, 3000 e-gas brems = blue, 4000 e-gas coulomb =
teal, 5000 e-gas touschek = magenta, 6000 p-gas = brown; names from the
cocktail JSONs in submodules/simulation-pipeline/background_cocktails/),
and Geant4-created hits (status 0) are colored by WHO made them
(prt_origin): light green = secondary of the signal, red = secondary of
background. Signal generator hits dark green, albedo purple.

Output: plots/temporal/events_color/0022_evtNNNN.png, first 50 events of
0022 (18x275_v3 parquet), same layout as events/ for direct comparison.

Usage (from the repo root):
    python scripts/240_event_displays_colored.py
    # or plot a 50-event window centered on one event (into a subfolder):
    ... 240_event_displays_colored.py --around 2231 [--file 0022]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg import temporal as T

PARQUET = (CFG.parquet_dir / "18x275_v3")
OUT = Path(__file__).resolve().parents[1] / "plots" / "temporal" / \
    "events_color"
OUT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--around", type=int, default=None,
                    help="plot the 50 events centered on this event id "
                         "into events_color/around_evtNNNN/ and exit")
parser.add_argument("--file", default="0022")
args = parser.parse_args()

d = T.load_events(PARQUET / f"{args.file}.trk_hits.parquet")

if args.around is not None:
    sub = OUT / f"around_evt{args.around:04d}"
    sub.mkdir(exist_ok=True)
    evt_index = {int(d["evt"][s]): j for j, s in enumerate(d["starts"])}
    lo = max(0, args.around - 25)
    for e in range(lo, lo + 50):
        if e in evt_index:
            T.event_display_color(d, evt_index[e],
                                  sub / f"{args.file}_evt{e:04d}.png",
                                  args.file)
    print(f"wrote 50 displays around event {args.around} to {sub}")
    sys.exit(0)
for i in range(50):
    T.event_display_color(d, i, OUT / f"0022_evt{i:04d}.png", "0022")
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/50", flush=True)
print(f"wrote 50 colored event displays to {OUT}")

# ---- showcase: the rare sources are 0.04% of hits (synrad 34.3M vs brems
# 5.6k / coulomb 0.5k / touschek 1.3k / p-gas 6.8k per file), so the first
# 50 events show them only sparsely. Pick the 3 events RICHEST in each rare
# band (and the 3 biggest G4-background showers) so every color is easy to
# see somewhere.
import numpy as np  # noqa: E402

SHOW = OUT / "showcase"
SHOW.mkdir(exist_ok=True)
band = np.where(d["status"] >= 1000, d["status"] // 1000 * 1000, 0)
evt_of_hit = d["evt"]
picked = {}
for b in (3000, 4000, 5000, 6000):
    ue, uc = np.unique(evt_of_hit[band == b], return_counts=True)
    for e in ue[np.argsort(-uc)][:3]:
        picked.setdefault(int(e), []).append(f"band{b}")
g4bkg = (d["status"] == 0) & (d["origin"] >= 3)
ue, uc = np.unique(evt_of_hit[g4bkg], return_counts=True)
for e in ue[np.argsort(-uc)][:3]:
    picked.setdefault(int(e), []).append("g4shower")
evt_index = {int(d["evt"][s]): j for j, s in enumerate(d["starts"])}
for e, why in sorted(picked.items()):
    T.event_display_color(d, evt_index[e], SHOW / f"0022_evt{e:04d}.png",
                          "0022")
    print(f"showcase evt {e}: {'+'.join(why)}", flush=True)
print(f"wrote {len(picked)} showcase displays to {SHOW}")

"""Three honest ways to draw the background funnel (slide figures).

The funnel of script 43 is right but hard to read. Two problems, two fixes:

  1. On a log axis a 4% cut and a 56x cut look about the same, and 55 366 next
     to 3 754 looks like a factor of two. People read log scales badly.
  2. It mixes PARTICLES and HITS in one column of bars, so the eye compares
     things that are not comparable (one particle leaves several hits — the
     chain even goes UP at that step).

Same numbers (cached in reports/phase7_funnel.json by script 43), three views,
and in all of them the three families are colour-coded and never blurred:
purple = particles, blue = hits, teal = tracking objects (seeds, tracks).

  background_funnel_log.png     the log view, made countable: alternating
      decade bands, the reduction factor printed on EVERY step (including the
      ones that do nothing) and the three regimes bracketed.

  background_funnel_linear.png  the linear view with a BROKEN axis: the 55 366
      particle bar is cut in the middle ("yes, it is long") so the rest of the
      chain keeps its true proportions.

  background_funnel_hits.png    HITS ONLY, linear, starting from all tracker
      hits of the frame. No particles, no broken axis, nothing to misread:
      this is the chain our filter actually lives in.

Physics point of all three (notes/stock-eicrecon-has-no-time-window.md): stock
EICrecon has no time cut anywhere. Digitization integrates the whole 2000 ns
frame, so background survives intact all the way to the seed finder, which
then pays for it in combinatorics — the bill our filter pays off.

Run: python scripts/623_funnel_views.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

JSON = (CFG.reports_dir / "phase7_funnel.json")
DIS_CMP = (CFG.reports_dir / "phase7_tracking_compare_dis.json")
PLOTS = (CFG.plots_dir / "phase7_dis")
TEXT, MUTED, RED = "#0b0b0b", "#52514e", "#e34948"

# three families, three hues — a particle is not a hit, a hit is not a track
PARTICLE = ["#cbbfe6", "#a892d6"]
HIT = ["#cde2fb", "#a9cdf6", "#86b6ef", "#6da7ec", "#4f93e6", "#2f7ddd"]
OBJECT = ["#2a9d8f", "#1c6b62"]
LEGEND = [Patch(fc=PARTICLE[1], label="particles"),
          Patch(fc=HIT[3], label="hits"),
          Patch(fc=OBJECT[0], label="tracking objects (seeds, tracks)")]

d = json.loads(JSON.read_text())
labels = [s["label"] for s in d["stages"]]
vals = np.array([s["per_event"] for s in d["stages"]])

I_ALL_HITS = labels.index("all tracker sim hits (12 systems)")
I_CENTRAL = labels.index("central-tracker sim hits")
I_ACC = labels.index("in seeder acceptance (33<r<440 mm)")
I_SEED = labels.index("track seeds")

KIND = (["particle"] * I_ALL_HITS + ["hit"] * (I_SEED - I_ALL_HITS)
        + ["object"] * (len(vals) - I_SEED))
COLORS, n_p, n_h, n_o = [], 0, 0, 0
for k in KIND:
    if k == "particle":
        COLORS.append(PARTICLE[min(n_p, len(PARTICLE) - 1)]); n_p += 1
    elif k == "hit":
        COLORS.append(HIT[min(n_h, len(HIT) - 1)]); n_h += 1
    else:
        COLORS.append(OBJECT[min(n_o, len(OBJECT) - 1)]); n_o += 1

geo_factor = vals[0] / vals[1]
det_factor = vals[I_CENTRAL] / vals[I_ACC]
trk_factor = vals[I_ACC] / vals[-1]


def factors(idx):
    """(y-between-rows, text, is_big) for consecutive pairs of the rows `idx`."""
    yy = np.arange(len(idx))[::-1]
    out = []
    for i in range(1, len(idx)):
        f = vals[idx[i - 1]] / vals[idx[i]]
        txt = f"÷{f:,.0f}" if f >= 1.5 else (f"×{1 / f:.1f}" if f < 0.95
                                             else f"÷{f:.2f}")
        out.append(((yy[i - 1] + yy[i]) / 2, txt, f >= 1.5))
    return out


# ---------------------------------------------------------------- log view
idx = list(range(len(vals)))
ys = np.arange(len(idx))[::-1]
fig, ax = plt.subplots(figsize=(10.6, 6.6))
for k in range(0, 6):                      # alternating decade bands: the eye
    if k % 2 == 0:                         # can count factors of ten
        ax.axvspan(10.0 ** k, 10.0 ** (k + 1), color="#f4f6f9", zorder=0)
ax.barh(ys, vals, height=0.6, color=COLORS, edgecolor="none", zorder=2)
ax.set_xscale("log")
ax.set_xlim(1, 3e5)
for v, yy in zip(vals, ys):
    ax.text(v * 1.18, yy, f"{v:,.0f}".replace(",", " "), va="center",
            fontsize=12, color=TEXT, fontweight="bold", zorder=3)
for yy, txt, big in factors(idx):
    ax.text(1.25, yy, txt, va="center", fontsize=10.5,
            color=RED if big else MUTED, fontweight="bold" if big else "normal",
            zorder=3)

# the three regimes, bracketed on the right — do NOT blur them together
for y0, y1, text, col in (
        (ys[1], ys[0], f"geometry ÷{geo_factor:.0f}\n93% never touch\nthe tracker",
         MUTED),
        (ys[I_ACC], ys[I_ALL_HITS], f"detector + digi\n÷{det_factor:.1f} in total\n"
                                    f"— no time cut", MUTED),
        (ys[-1], ys[I_ACC], f"tracking ÷{trk_factor:,.0f}\nbrute-force\n"
                            f"combinatorics", RED)):
    ax.plot([2.7e5, 2.7e5], [y0, y1], color=col, lw=1.4, clip_on=False)
    ax.text(3.1e5, (y0 + y1) / 2, text, va="center", fontsize=10, color=col,
            fontweight="bold", clip_on=False)

ax.set_yticks(ys, labels, fontsize=11)
ax.set_xlabel("per frame — LOG scale: every shaded band is another ×10",
              color=TEXT)
ax.grid(True, axis="x", color="#d9d8d4", lw=0.6, zorder=1)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.legend(handles=LEGEND, loc="lower right", fontsize=9.5, framealpha=0.95)
ax.set_title("Where the background goes — stock EICrecon, per frame",
             fontsize=13, color=TEXT, loc="left")
fig.tight_layout(rect=[0, 0, 0.86, 1])
fig.savefig(PLOTS / "background_funnel_log.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------- linear view
# Linear bars keep the true proportions; the one bar that does not fit (the
# 55 366 MC particles) is cut in the middle and continues on a second panel.
BREAK = 9800.0
fig, (axl, axr) = plt.subplots(
    1, 2, figsize=(10.6, 6.8), sharey=True,
    gridspec_kw={"width_ratios": [6, 1], "wspace": 0.06})
for ax in (axl, axr):
    ax.barh(ys, vals, height=0.6, color=COLORS, edgecolor="none", zorder=2)
    ax.grid(True, axis="x", color="#d9d8d4", lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top", "left"):
        ax.spines[sp].set_visible(False)
axl.set_xlim(0, BREAK)
axr.set_xlim(53500, 57000)
axr.set_xticks([55000])
axr.tick_params(left=False)          # the y ticks belong to the left panel
axl.spines["right"].set_visible(False)
axr.spines["right"].set_visible(False)

kw = dict(transform=axl.transAxes, color="#0b0b0b", clip_on=False, lw=1.1)
axl.plot([1 - 0.008, 1 + 0.008], [-0.014, 0.014], **kw)
kw = dict(transform=axr.transAxes, color="#0b0b0b", clip_on=False, lw=1.1)
axr.plot([-0.05, 0.05], [-0.014, 0.014], **kw)

for v, yy in zip(vals, ys):
    if v > BREAK:                                  # the broken bar: label it
        axr.text(v + 400, yy, f"{v:,.0f}".replace(",", " "), va="center",
                 fontsize=12.5, color=TEXT, fontweight="bold")
        axl.text(BREAK * 0.5, yy, "the bar keeps going  ⇢", va="center",
                 ha="center", fontsize=11, color="#5b4a86", style="italic",
                 zorder=3)
    else:
        axl.text(v + 140, yy, f"{v:,.0f}".replace(",", " "), va="center",
                 fontsize=12, color=TEXT, fontweight="bold")
for yy, txt, big in factors(idx):
    axl.text(140, yy, txt, va="center", fontsize=10.5,
             color=RED if big else MUTED,
             fontweight="bold" if big else "normal", zorder=3)

axl.set_yticks(ys, labels, fontsize=11)
axl.set_xlabel("per frame — LINEAR scale (true proportions)", color=TEXT)
fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
           frameon=False, bbox_to_anchor=(0.55, -0.005))
# (no "no time cut" callout here: the ÷1.04 / ÷1.27 factors between the hit
# rows already say it, and every place it fits collides with a value label)
axl.annotate(f"the seed finder does all the work —\n"
             f"and pays ÷{trk_factor:,.0f} of combinatorics for it",
             xy=(vals[I_SEED] + 90, ys[I_SEED]), xytext=(1700, ys[-1] + 0.15),
             fontsize=10, color=RED, fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0,
                             connectionstyle="arc3,rad=-0.2"))
axl.set_title("Same numbers, linear scale — true proportions",
              fontsize=13, color=TEXT, loc="left")
fig.tight_layout()
fig.subplots_adjust(left=0.32, bottom=0.14)   # room for names + legend
fig.savefig(PLOTS / "background_funnel_linear.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------- hits only
# The same chain with the particles dropped: it starts from every tracker hit
# of the frame — exactly what the AI filter is handed — and stays on one
# linear axis all the way down to the 9 tracks. Nothing to misread.
idx = list(range(I_ALL_HITS, len(vals)))
hv = vals[idx]
hys = np.arange(len(idx))[::-1]
hcol = [COLORS[i] for i in idx]

fig, ax = plt.subplots(figsize=(10.6, 6.2))
ax.barh(hys, hv, height=0.62, color=hcol, edgecolor="none", zorder=2)
for v, yy in zip(hv, hys):
    ax.text(v + 90, yy, f"{v:,.0f}".replace(",", " "), va="center",
            fontsize=12.5, color=TEXT, fontweight="bold", zorder=3)
for yy, txt, big in factors(idx):
    ax.text(90, yy, txt, va="center", fontsize=11,
            color=RED if big else MUTED,
            fontweight="bold" if big else "normal", zorder=3)

# what our filter changes, measured on this very campaign (report 09 table)
if DIS_CMP.is_file():
    cmp_dis = json.loads(DIS_CMP.read_text())
    seeds_u = cmp_dis["unfilt"]["seeds_per_event_mean"]
    seeds_f = cmp_dis["finder"]["seeds_per_event_mean"]
    ax.annotate(f"with our filter in front of the seeder:\n"
                f"{seeds_u:.0f} → {seeds_f:.0f} seeds per frame",
                xy=(hv[-2] + 90, hys[-2]), xytext=(3900, hys[-1] + 0.45),
                fontsize=11, color="#1c6b62", fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-|>", color="#1c6b62", lw=1.2))

ax.set_yticks(hys, [labels[i] for i in idx], fontsize=11.5)
ax.set_xlabel("per frame — LINEAR scale, hits only", color=TEXT)
ax.set_xlim(0, hv.max() * 1.2)
ax.grid(True, axis="x", color="#d9d8d4", lw=0.6, zorder=1)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.legend(handles=LEGEND[1:], loc="lower right", fontsize=10, framealpha=0.95)
ax.set_title("The chain our filter lives in: every tracker hit of the frame, "
             "down to 9 tracks", fontsize=13, color=TEXT, loc="left")
fig.tight_layout()
fig.savefig(PLOTS / "background_funnel_hits.png", dpi=150)
plt.close(fig)

print(f"geometry ÷{geo_factor:.1f} | detector+digi ÷{det_factor:.2f} | "
      f"tracking ÷{trk_factor:,.0f}")
print("wrote background_funnel_{log,linear,hits}.png ->", PLOTS)

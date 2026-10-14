#!/usr/bin/env python
"""Phase 7: three clean, single-topic timing panels for slides.

Script 44 (621_onnx_timing_plot.py) makes one dense two-panel figure with an
inset zoom -- fine for a report appendix, too busy to drop into a talk. This
script re-reads the SAME numbers from reports/phase7_onnx_timing.json (it
does not recompute or hardcode anything) and re-renders them as three small,
single-message PNGs, each meant to stand alone on a slide at ~4 inches wide:

  1. plots/phase7/time_full_reco.png
     "Does the AI filter make the whole event reconstruction faster?"
     Total EICrecon CPU time per event, unfiltered vs AI-finder vs AI-ideal.

  2. plots/phase7/time_tracking.png
     "Where does that speedup actually come from?"
     Pure tracking time per event (seeding + CKF track fit only, from the
     JANA2 janatop factory report), stacked, unfiltered vs finder vs ideal.

  3. plots/phase7/time_onnx_cost.png
     "What does running the AI filter itself cost us?"
     The single-thread ONNX inference cost per event we ADD (finder2d image
     classifier + WATT attention model + MLP model), shown against the much
     larger tracking-time saving from panel 2, to make clear the added cost
     is small change compared to what it buys.

Run: python 622_timing_panels.py  (no arguments; idempotent)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

JSON = (CFG.reports_dir / "phase7_onnx_timing.json")
PLOT_DIR = (CFG.plots_dir / "phase7")

# house style, copied from 621_onnx_timing_plot.py
TEXT = "#0b0b0b"
GRID = "#d9d8d4"
COLOR_UNFILT = "#2a78d6"
COLOR_FINDER = "#1baf7a"
COLOR_IDEAL = "#eda100"
COLOR_FINDER2D = "#104281"
COLOR_WATT = "#3987e5"
COLOR_MLP = "#9ec5f4"
ACCENT = "#e34948"

MODE_LABELS = {"unfilt": "unfiltered", "finder": "AI finder", "ideal": "AI ideal"}
MODE_COLORS = {"unfilt": COLOR_UNFILT, "finder": COLOR_FINDER, "ideal": COLOR_IDEAL}


def style_axes(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_full_reco(reco):
    """Panel 1: total reconstruction CPU time per event, before vs after."""
    tags = ["unfilt", "finder", "ideal"]
    vals = [reco[t] for t in tags]
    colors = [MODE_COLORS[t] for t in tags]
    labels = [MODE_LABELS[t] for t in tags]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    xs = range(len(tags))
    ax.bar(xs, vals, 0.55, color=colors, edgecolor="white", linewidth=0.8)
    for x, v in zip(xs, vals):
        ax.text(x, v + max(vals) * 0.02, f"{v:.0f} ms", ha="center",
                 fontsize=11, color=TEXT, fontweight="bold")
    ax.set_xticks(list(xs), labels, fontsize=10)
    ax.set_ylabel("CPU time per event [ms]", color=TEXT, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_title("Full event reconstruction time\n(EICrecon CPU, whole job)",
                 fontsize=11, color=TEXT)
    style_axes(ax)

    speedup = reco["unfilt"] / reco["finder"]
    ax.annotate(f"{speedup:.1f}× faster",
                xy=(1, reco["finder"] + max(vals) * 0.03),
                xytext=(0.5, max(vals) * 0.72),
                fontsize=13, color=ACCENT, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))

    fig.tight_layout()
    out = PLOT_DIR / "time_full_reco.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, speedup


def plot_tracking(seed, ckf):
    """Panel 2: pure tracking time (seeding + CKF), stacked, before vs after."""
    tags = ["unfilt", "finder", "ideal"]
    totals = {t: seed[t] + ckf[t] for t in tags}

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    xs = range(len(tags))
    for x, t in zip(xs, tags):
        c = MODE_COLORS[t]
        ax.bar([x], [seed[t]], 0.55, color=c, alpha=0.45,
               edgecolor="white", linewidth=0.8)
        ax.bar([x], [ckf[t]], 0.55, bottom=[seed[t]], color=c,
               edgecolor="white", linewidth=0.8)
        ax.text(x, totals[t] + max(totals.values()) * 0.03,
                f"{totals[t]:.0f} ms", ha="center", fontsize=11,
                color=TEXT, fontweight="bold")
        ax.text(x, seed[t] + ckf[t] / 2, f"CKF fit\n{ckf[t]:.0f}",
                ha="center", va="center", fontsize=8.5,
                color="white" if ckf[t] > 30 else TEXT)
        ax.text(x, seed[t] / 2, f"seed {seed[t]:.1f}", ha="center",
                va="center", fontsize=7.5, color=TEXT)

    ax.set_xticks(list(xs), [MODE_LABELS[t] for t in tags], fontsize=10)
    ax.set_ylabel("CPU time per event [ms]", color=TEXT, fontsize=10)
    ax.set_ylim(0, max(totals.values()) * 1.25)
    ax.set_title("Pure tracking time: seeding + CKF fit\n"
                 "(JANA2 janatop factory report, both CKF passes)",
                 fontsize=10.5, color=TEXT)
    style_axes(ax)

    speedup = totals["unfilt"] / totals["finder"]
    ax.annotate(f"tracking alone:\n{speedup:.1f}× faster",
                xy=(1, totals["finder"] + max(totals.values()) * 0.04),
                xytext=(0.7, max(totals.values()) * 0.68),
                fontsize=12, color=ACCENT, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))

    fig.tight_layout()
    out = PLOT_DIR / "time_tracking.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, totals, speedup


def plot_onnx_cost(onnx, saving_ms):
    """Panel 3: single-thread ONNX inference cost we add, finder vs ideal."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))

    modes = [
        ("AI finder mode", [
            ("finder2d image", onnx["finder2d_image"], COLOR_FINDER2D),
            ("WATT (in-window hits)", onnx["watt_1000_inwindow_hits"], COLOR_WATT),
            ("MLP (outside hits)", onnx["mlp_4200_outside_hits"], COLOR_MLP),
        ]),
        ("AI ideal mode", [
            ("WATT (in-window hits)", onnx["watt_1000_inwindow_hits"], COLOR_WATT),
            ("MLP (outside hits)", onnx["mlp_4200_outside_hits"], COLOR_MLP),
        ]),
    ]
    xs = range(len(modes))
    seen_labels = set()
    for x, (_, parts) in zip(xs, modes):
        bottom = 0.0
        for pname, v, color in parts:
            lbl = pname if pname not in seen_labels else None
            ax.bar([x], [v], 0.5, bottom=bottom, color=color,
                   edgecolor="white", linewidth=0.8,
                   label=lbl)
            seen_labels.add(pname)
            ax.text(x, bottom + v / 2, f"{v:.1f}", ha="center", va="center",
                    fontsize=8.5, color="white" if v > 1.5 else TEXT)
            bottom += v
        ax.text(x, bottom + max(onnx["mode_finder_total"], onnx["mode_ideal_total"]) * 0.05,
                f"{bottom:.1f} ms", ha="center", fontsize=11, color=TEXT,
                fontweight="bold")

    ax.set_xticks(list(xs), [m[0] for m in modes], fontsize=10)
    ax.set_ylabel("CPU time per event [ms]\n(single ONNX thread)", color=TEXT, fontsize=10)
    ax.set_ylim(0, max(onnx["mode_finder_total"], onnx["mode_ideal_total"]) * 1.55)
    ax.set_ylim(0, max(onnx["mode_finder_total"], onnx["mode_ideal_total"]) * 1.75)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.set_title("AI filter cost added: ONNX inference only\n"
                 "(single-thread CPU, per event)", fontsize=10.5, color=TEXT)
    style_axes(ax)

    # headline sits above the shorter bar, clear of the legend on the left
    ax.text(0.97, 0.72,
            f"~{onnx['mode_finder_total']:.0f} ms added\n"
            f"vs {saving_ms:.0f} ms saved in full reco",
            ha="right", va="top", fontsize=10, color=ACCENT, fontweight="bold",
            transform=ax.transAxes)

    fig.tight_layout()
    out = PLOT_DIR / "time_onnx_cost.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    data = json.loads(JSON.read_text())
    reco = data["reco_cpu_ms_per_event"]
    seed = data["janatop_ms_per_event"]["seeding"]
    ckf = data["janatop_ms_per_event"]["ckf_fit_both_passes"]
    onnx = data["onnx_ms_per_event_1thread"]

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    out1, speedup1 = plot_full_reco(reco)
    out2, totals2, speedup2 = plot_tracking(seed, ckf)
    saving_ms = reco["unfilt"] - reco["finder"]  # whole-event reco saving
    out3 = plot_onnx_cost(onnx, saving_ms)

    print(f"saved {out1}  (unfilt={reco['unfilt']:.1f}ms finder={reco['finder']:.1f}ms "
          f"ideal={reco['ideal']:.1f}ms speedup={speedup1:.2f}x)")
    print(f"saved {out2}  (unfilt={totals2['unfilt']:.1f}ms finder={totals2['finder']:.1f}ms "
          f"ideal={totals2['ideal']:.1f}ms speedup={speedup2:.2f}x)")
    print(f"saved {out3}  (finder_total={onnx['mode_finder_total']:.2f}ms "
          f"ideal_total={onnx['mode_ideal_total']:.2f}ms saving={saving_ms:.1f}ms)")


if __name__ == "__main__":
    main()

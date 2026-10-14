#!/usr/bin/env python
"""Phase 7: AI-filter ONNX inference cost per event vs total reconstruction.

Measures the deployed ONNX models (files in /eic-workbench/ai-trkfilter-models,
single CPU thread, session options identical to the C++ plugin) at hit counts
representative of one 2000 ns frame, and compares against the TOTAL EICrecon
CPU time per event extracted from the /usr/bin/time logs of the 10k-frame
k_lambda runs (User time / 5000 events per file).

Outputs: plots/phase7/onnx_timing_context.png
         reports/phase7_onnx_timing.json
"""

import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

MODELS = Path("/eic-workbench/ai-trkfilter-models")
LOGS = (CFG.results_root / "eicrecon")
PLOT = (CFG.plots_dir / "phase7/onnx_timing_context.png")
JSON = (CFG.reports_dir / "phase7_onnx_timing.json")
N_EVT_PER_FILE = 5000

# representative per-frame hit counts (median of the phase-7 runs)
N_IN = 1000    # in-window hits -> watt (one attention call per frame)
N_OUT = 4200   # out-of-window hits -> mlp
TEXT = "#0b0b0b"


def sess(name):
    o = ort.SessionOptions()
    o.inter_op_num_threads = 1
    o.intra_op_num_threads = 1
    return ort.InferenceSession(MODELS / name, sess_options=o,
                                providers=["CPUExecutionProvider"])


def bench(fn, n=30):
    fn()  # warmup
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def cpu_ms_per_event(tag):
    vals = []
    for fno in ("0022", "0023"):
        txt = (LOGS / f"{tag}_{fno}.log").read_text()
        user = float(re.search(r"User time \(seconds\): ([\d.]+)", txt).group(1))
        vals.append(user / N_EVT_PER_FILE * 1e3)
    return float(np.mean(vals))


def factory_ms_per_event(tag, factory):
    """Per-event self-time of one JANA factory from the janatop report
    (RECORD_CALL_STACK=1 runs print it at end-of-job; time is summed over
    all calls, so /N events gives ms/event regardless of call count)."""
    vals = []
    for fno in ("0022", "0023"):
        txt = (LOGS / f"{tag}_{fno}.log").read_text()
        m = re.search(rf"^([\d.]+) s \(\s*[\d.]+%\) :{factory}$", txt, re.M)
        vals.append(float(m.group(1)) / N_EVT_PER_FILE * 1e3 if m else np.nan)
    return float(np.nanmean(vals))


def main():
    rng = np.random.default_rng(1)
    finder = sess("window_finder2d.onnx")
    watt = sess("window_model.onnx")
    mlp = sess("mlp_model.onnx")

    img = np.log1p(rng.poisson(1.5, (1, 2, 64, 88))).astype(np.float32)

    def hits(n):
        h = np.column_stack([rng.normal(0, 200, n), rng.normal(0, 200, n),
                             rng.normal(0, 1000, n),
                             rng.uniform(-50, 300, n)]).astype(np.float32)
        return {"hits": h, "system": rng.integers(0, 22, n).astype(np.int64)}

    t_finder = bench(lambda: finder.run(None, {"image": img}))
    in_w = hits(N_IN)
    t_watt = bench(lambda: watt.run(None, in_w))
    out_w = hits(N_OUT)
    t_mlp = bench(lambda: mlp.run(None, out_w))

    reco = {tag: cpu_ms_per_event(tag) for tag in ("unfilt", "finder", "ideal")}
    # pure tracking components from the janatop factory report
    ckf = {t: factory_ms_per_event(t, "CentralCKFActsTrackStatesUnfiltered")
           for t in ("unfilt", "finder", "ideal")}
    seed = {t: factory_ms_per_event(t, "CentralTrackSeeds")
            for t in ("unfilt", "finder", "ideal")}
    filt = {t: factory_ms_per_event(t, "AiSignalTrackingRecHits")
            for t in ("unfilt", "finder", "ideal")}
    summary = {
        "onnx_ms_per_event_1thread": {
            "finder2d_image": t_finder,
            f"watt_{N_IN}_inwindow_hits": t_watt,
            f"mlp_{N_OUT}_outside_hits": t_mlp,
            "mode_finder_total": t_finder + t_watt + t_mlp,
            "mode_ideal_total": t_watt + t_mlp,
        },
        "reco_cpu_ms_per_event": reco,
        "janatop_ms_per_event": {
            "ckf_fit_both_passes": ckf,
            "seeding": seed,
            "ai_filter_factory_in_situ": filt,
            "note": "factory self-time from janatop (RECORD_CALL_STACK); "
                    "CKF counts truth-seeded + real-seeded passes; in the "
                    "unfilt runs the AI factory still executes (its output "
                    "collections are persisted) but does not feed tracking",
        },
    }
    JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- plot: filter cost in context of the whole reconstruction ----
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(13.5, 4.8),
                                  width_ratios=[1.45, 1.0])
    bars = [
        ("full reco / event\nunfiltered", [("total", reco["unfilt"], "#2a78d6")]),
        ("full reco / event\nAI finder mode", [("total", reco["finder"], "#1baf7a")]),
        ("full reco / event\nAI ideal mode", [("total", reco["ideal"], "#eda100")]),
        ("AI filter ONNX only\nfinder mode", [("finder2d", t_finder, "#104281"),
                                              ("watt", t_watt, "#3987e5"),
                                              ("mlp", t_mlp, "#9ec5f4")]),
        ("AI filter ONNX only\nideal mode", [("watt", t_watt, "#3987e5"),
                                             ("mlp", t_mlp, "#9ec5f4")]),
    ]
    xs = np.arange(len(bars))
    for x, (label, parts) in zip(xs, bars):
        bottom = 0.0
        for pname, v, color in parts:
            ax.bar([x], [v], 0.58, bottom=bottom, color=color,
                   edgecolor="white", linewidth=0.8)
            if v > 3:
                ax.text(x, bottom + v / 2, pname if len(parts) > 1 else "",
                        ha="center", va="center", fontsize=8, color="white")
            bottom += v
        ax.text(x, bottom * 1.04, f"{bottom:.0f} ms", ha="center",
                fontsize=11, color=TEXT, fontweight="bold")
    ax.set_xticks(xs, [b[0] for b in bars], fontsize=9)
    ax.set_ylabel("CPU ms per event", color=TEXT)
    ax.set_ylim(0, max(reco.values()) * 1.18)
    ax.grid(True, axis="y", color="#d9d8d4", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("AI-filter inference cost vs total event reconstruction "
                 "(single-thread ONNX; reco CPU from 10k-frame k_lambda runs)",
                 fontsize=10, color=TEXT)

    # inset: zoom on the two ONNX-only bars so the components are readable
    axi = ax.inset_axes([0.58, 0.42, 0.38, 0.52])
    for xi, parts in enumerate([bars[3][1], bars[4][1]]):
        bottom = 0.0
        for pname, v, color in parts:
            axi.bar([xi], [v], 0.5, bottom=bottom, color=color,
                    edgecolor="white", linewidth=0.8)
            axi.text(xi + 0.32, bottom + v / 2, f"{pname} {v:.1f}",
                     va="center", fontsize=8, color=TEXT)
            bottom += v
        axi.text(xi, bottom + 0.4, f"{bottom:.0f} ms", ha="center",
                 fontsize=9, color=TEXT, fontweight="bold")
    axi.set_xticks([0, 1], ["finder mode", "ideal mode"], fontsize=8)
    axi.set_xlim(-0.5, 2.1)
    axi.set_ylim(0, 15)
    axi.set_ylabel("ms", fontsize=8)
    axi.tick_params(labelsize=8)
    axi.grid(True, axis="y", color="#d9d8d4", linewidth=0.5)
    axi.set_axisbelow(True)
    axi.set_title("zoom: AI filter ONNX per event", fontsize=8.5, color=TEXT)

    # ---- right panel: PURE TRACKING time (janatop factory report) ----
    tags = ["unfilt", "finder", "ideal"]
    tag_colors = {"unfilt": "#2a78d6", "finder": "#1baf7a", "ideal": "#eda100"}
    for x, t in enumerate(tags):
        axt.bar([x], [seed[t]], 0.55, color=tag_colors[t], alpha=0.45,
                edgecolor="white", linewidth=0.8)
        axt.bar([x], [ckf[t]], 0.55, bottom=[seed[t]], color=tag_colors[t],
                edgecolor="white", linewidth=0.8)
        tot = seed[t] + ckf[t]
        axt.text(x, tot + 8, f"{tot:.0f} ms", ha="center", fontsize=11,
                 color=TEXT, fontweight="bold")
        axt.text(x, seed[t] + ckf[t] / 2, f"CKF {ckf[t]:.0f}", ha="center",
                 va="center", fontsize=8,
                 color="white" if ckf[t] > 30 else TEXT)
        axt.text(x, seed[t] / 2, f"seed {seed[t]:.1f}", ha="center",
                 va="center", fontsize=7.5, color=TEXT)
    speedup = (seed["unfilt"] + ckf["unfilt"]) / (seed["finder"] + ckf["finder"])
    axt.annotate(f"tracking alone:\n{speedup:.1f}× faster",
                 xy=(1, seed["finder"] + ckf["finder"] + 40),
                 xytext=(0.85, 230), fontsize=11, color="#e34948",
                 fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#e34948"))
    axt.set_xticks(range(3), ["unfiltered", "AI finder", "AI ideal"], fontsize=9)
    axt.set_ylabel("CPU ms per event", color=TEXT)
    axt.grid(True, axis="y", color="#d9d8d4", linewidth=0.6)
    axt.set_axisbelow(True)
    for sp in ("top", "right"):
        axt.spines[sp].set_visible(False)
    axt.set_title("pure tracking: seeding + CKF fit\n(JANA2 janatop factory "
                  "report, both CKF passes)", fontsize=9.5, color=TEXT)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"saved {PLOT}")


if __name__ == "__main__":
    main()

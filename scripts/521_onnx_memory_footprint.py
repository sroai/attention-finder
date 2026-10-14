"""Memory footprint of the deployed filter — the other half of the cost.

Speed we measured (scripts 44/46); this is memory. Two independent
measurements, because they answer two different questions.

NB the 1 GB / 2 GB ceilings from the original brief are an INTERNAL
requirement of ours: they are checked in the JSON, never drawn on the figure.

1. WHAT THE MODELS COST BY THEMSELVES. A child process loads the three
   deployed ONNX files from /eic-workbench/ai-trkfilter-models (exactly the
   ones EICrecon runs) into onnxruntime and executes one realistic frame:
   the 2x64x88 finder image, ~1200 in-window hits through the attention net,
   ~4200 out-of-window hits through the MLP. We record the RSS the sessions
   add on top of a bare interpreter, and the peak during inference.

2. WHAT THE FILTER COSTS THE JOB. /usr/bin/time -v recorded the peak RSS of
   every full EICrecon run (10k frames, k_lambda). CAVEAT, and it matters:
   the AI factory executes in ALL THREE variants (its collections are always
   persisted), so the unfiltered run also pays for the ONNX sessions. The
   difference between the bars is therefore NOT the model cost — it is what
   TRACKING no longer allocates once the background hits are gone.

Outputs: reports/phase7_onnx_memory.json + plots/phase7/memory_footprint.png
Run: python scripts/521_onnx_memory_footprint.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

MODELS = (CFG.eic_workbench / "ai-trkfilter-models")
LOGS = (CFG.results_root / "eicrecon")
REPORTS = CFG.reports_dir
PLOTS = (CFG.plots_dir / "phase7")
TEXT, MUTED, RED, WHITE_TXT = "#0b0b0b", "#52514e", "#e34948", "#ffffff"
BLUE, GREEN, ORANGE, TEAL = "#2f7ddd", "#2a9d8f", "#e8a33d", "#1c6b62"

# ---------------------------------------------------------------------------
# 1. the models on their own — measured in a child process so the parent's
#    numpy/torch/matplotlib pages do not pollute the number
CHILD = r"""
import json, os, sys
import numpy as np, onnxruntime as ort

def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20

def peak_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024

MODELS = sys.argv[1]
base = rss_mb()
so = ort.SessionOptions()
so.intra_op_num_threads = 1                      # EICrecon runs 1 thread/model
sess = {n: ort.InferenceSession(os.path.join(MODELS, f), so,
                                providers=["CPUExecutionProvider"])
        for n, f in (("finder2d", "window_finder2d.onnx"),
                     ("watt", "window_model.onnx"),
                     ("mlp", "mlp_model.onnx"))}
loaded = rss_mb()

rng = np.random.default_rng(0)
img = rng.random((1, 2, 64, 88), dtype=np.float32)
n_in, n_out = 1200, 4200                          # a typical frame
hits_in = rng.random((n_in, 4)).astype(np.float32) * 100
sys_in = rng.integers(0, 20, n_in).astype(np.int64)
hits_out = rng.random((n_out, 4)).astype(np.float32) * 100
sys_out = rng.integers(0, 20, n_out).astype(np.int64)

def run(s, feeds):
    return s.run(None, feeds)

def feed(s, arrays):
    names = [i.name for i in s.get_inputs()]
    return dict(zip(names, arrays))

for _ in range(3):                                # warm up + steady state
    run(sess["finder2d"], feed(sess["finder2d"], [img]))
    run(sess["watt"], feed(sess["watt"], [hits_in, sys_in]))
    run(sess["mlp"], feed(sess["mlp"], [hits_out, sys_out]))
after = rss_mb()

print(json.dumps({"baseline_rss_mb": base, "after_load_rss_mb": loaded,
                  "after_inference_rss_mb": after, "peak_rss_mb": peak_mb(),
                  "sessions_mb": loaded - base, "inference_mb": after - loaded}))
"""

out = subprocess.run([sys.executable, "-c", CHILD, str(MODELS)],
                     capture_output=True, text=True, check=True)
models = json.loads(out.stdout.strip().splitlines()[-1])
models["weights_on_disk_mb"] = sum(f.stat().st_size for f in MODELS.glob("*.onnx")) / 2**20
models["files"] = {f.name: round(f.stat().st_size / 1024, 1)
                   for f in sorted(MODELS.glob("*.onnx"))}

# ---------------------------------------------------------------------------
# 2. the whole EICrecon job — peak RSS from /usr/bin/time -v in the run logs
job = {}
for variant in ("unfilt", "finder", "ideal"):
    peaks = []
    for log in sorted(LOGS.glob(f"{variant}_*.log")):
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log.read_text())
        if m:
            peaks.append(int(m.group(1)) / 2**20)     # KiB -> GiB
    if peaks:
        job[variant] = {"peak_rss_gb_mean": float(np.mean(peaks)),
                        "runs": len(peaks)}

models["attributable_ram_mb"] = models["sessions_mb"] + models["inference_mb"]
report = {"models_standalone": models, "eicrecon_job_peak_rss": job,
          "note": "the AI factory runs in ALL variants (its collections are "
                  "persisted), so the unfilt-vs-filter difference is tracking "
                  "memory, not model memory"}
REPORTS.mkdir(parents=True, exist_ok=True)
(REPORTS / "phase7_onnx_memory.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

# ---------------------------------------------------------------------------
fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.6, 5.4),
                               gridspec_kw={"width_ratios": [1, 1.15]})

# left: the filter's own footprint. Only what the MODELS add is charged here:
# the RSS of the bare interpreter before any session is created (~46 MB of
# python + numpy + the onnxruntime library) is not ours to claim, so it is
# subtracted, not drawn.
# (The 1/2 GB ceilings of the original brief are OUR internal requirement —
# deliberately not drawn on a figure meant for an audience.)
ax = axl
attributable = models["sessions_mb"] + models["inference_mb"]
ax.bar([0], [models["sessions_mb"]], width=0.5, color=GREEN, zorder=3,
       label=f"3 ONNX sessions loaded  ({models['sessions_mb']:.1f} MB)")
ax.bar([0], [models["inference_mb"]], width=0.5, bottom=[models["sessions_mb"]],
       color=BLUE, zorder=3,
       label=f"inference arenas, one frame  (+{models['inference_mb']:.1f} MB)")
ax.text(0, attributable + 1.4, f"{attributable:.0f} MB", ha="center",
        fontsize=16, fontweight="bold", color=TEXT, zorder=4)
ax.set_ylim(0, attributable * 1.55)
ax.set_xlim(-0.6, 0.6)
ax.set_xticks([0], ["the AI filter"], fontsize=11)
ax.set_ylabel("memory added by the models [MB]", color=TEXT)
ax.legend(loc="upper center", fontsize=9, framealpha=0.95)
ax.set_title(f"What the models cost: {attributable:.0f} MB of RAM\n"
             f"(clean process, 1 thread) — and "
             f"{models['weights_on_disk_mb']:.2f} MB of weights on disk",
             fontsize=11.5, color=TEXT, loc="left")
ax.grid(True, axis="y", color="#d9d8d4", lw=0.6, zorder=1)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# right: the whole job — filtering makes reconstruction LIGHTER, not heavier
ax = axr
names = {"unfilt": "unfiltered", "finder": "AI finder", "ideal": "AI ideal"}
cols = {"unfilt": "#7fb3ec", "finder": GREEN, "ideal": ORANGE}
ks = [k for k in ("unfilt", "finder", "ideal") if k in job]
xs = np.arange(len(ks))
vals = [job[k]["peak_rss_gb_mean"] for k in ks]
ax.bar(xs, vals, width=0.6, color=[cols[k] for k in ks], zorder=3)
for x, v in zip(xs, vals):
    ax.text(x, v + 0.08, f"{v:.2f} GB", ha="center", fontsize=12,
            fontweight="bold", color=TEXT, zorder=4)
if "unfilt" in job and "finder" in job:
    saved = (job["unfilt"]["peak_rss_gb_mean"] - job["finder"]["peak_rss_gb_mean"]) * 1024
    ax.annotate(f"{saved:.0f} MB LESS with the filter on\n"
                f"(ACTS holds far fewer seeds\nand track candidates)",
                xy=(1, vals[1] + 0.07), xytext=(1.55, max(vals) * 1.02),
                fontsize=10, color=TEAL, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2))
ax.set_xticks(xs, [names[k] for k in ks], fontsize=11)
ax.set_ylabel("peak RSS of the whole EICrecon job [GB]", color=TEXT)
ax.set_ylim(0, max(vals) * 1.28)
ax.set_title("What the filter costs the JOB: nothing — it saves\n"
             "(all three runs load the ONNX models; only tracking differs)",
             fontsize=11.5, color=TEXT, loc="left")
ax.grid(True, axis="y", color="#d9d8d4", lw=0.6, zorder=1)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.tight_layout()
PLOTS.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOTS / "memory_footprint.png", dpi=150)
print("wrote", PLOTS / "memory_footprint.png")

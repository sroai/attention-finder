"""How the 2D window finder sees a frame (illustration figure for the slides).

The finder never looks at individual hits. A whole 2000 ns frame is rendered
as a small two-channel IMAGE and a convolutional net reads it like a picture:

    channel 0   (z, t)   64 z-bins  x 88 time-bins of 25 ns, log(1 + count)
    channel 1   (r, t)   64 r-bins  x 88 time-bins            log(1 + count)

Time is the x axis of BOTH channels, so a column of the image is one 25 ns
slice of the frame. The network pools only along the vertical (space) axis,
never along time, so all 88 columns survive to the output: one logit per
column, softmax over columns = "which 25 ns slice contains the collision".

What the picture shows, and why a CNN was the right move: the ep collision is
not a bright column (background is 200x more numerous and bursts too) - it is
a SHAPE. A full radial ladder lights up at once in (r, t), z spreads
symmetrically around the interaction point, and a far-forward streak trails a
few slices later. Those are exactly the patterns 3x3 kernels are good at, and
exactly what the 11 hand-made numbers per slice of the earlier finder could
not express.

The bottom panel is the model's actual output on this frame: the per-column
probability, the column it picks, and the true t0.

Outputs: plots/phase6/finder2d_illustration.png
Run:  python scripts/712_finder2d_illustration.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.windows import BIN_NS, N_BINS, T_MIN
from trkbkg.windows2d import N_SPACE, R_RANGE, WindowFinder2D, Z_RANGE

DATA = (CFG.dataset_dir / "images2d_v1")
RUN = (CFG.runs_local_dir / "finder2d_v1")
PLOTS = (CFG.plots_dir / "phase6")
PRESET_18x275 = 0

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = json.loads((RUN / "config.json").read_text())
model = WindowFinder2D(width=cfg.get("width", 32),
                       n_blocks=cfg.get("blocks", 3),
                       batchnorm=cfg.get("batchnorm", False)).to(dev)
ckpt = torch.load(RUN / "best.pt", map_location=dev)
model.load_state_dict(ckpt.get("model_state", ckpt))
model.eval()

d = np.load(DATA / "test.npz")
img, target, t0, preset = d["img"], d["target"], d["t0"], d["preset"]

with torch.no_grad():
    logits = model(torch.from_numpy(img).to(dev))
    prob = torch.softmax(logits, dim=-1).cpu().numpy()
pred = prob.argmax(1)

# Display frame: 18x275, correctly found, and busy enough to look like the
# real thing (a nearly empty frame would be a flattering illustration).
occ = img[:, 0].sum(axis=(1, 2))
ok = np.flatnonzero((preset == PRESET_18x275) & (pred == target)
                    & (occ > np.percentile(occ, 60)))
i = int(ok[np.argsort(prob[ok, target[ok]])[len(ok) // 2]])   # typical confidence

t_edges = T_MIN + np.arange(N_BINS + 1) * BIN_NS
fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True,
                         gridspec_kw={"height_ratios": [3, 3, 2.1]})

for ax, ch, (lo, hi), name in (
        (axes[0], 0, Z_RANGE, "z"),
        (axes[1], 1, R_RANGE, "r")):
    ax.imshow(img[i, ch], origin="lower", aspect="auto", cmap="magma",
              extent=[t_edges[0], t_edges[-1], lo, hi])
    ax.set_ylabel(f"{name} [mm]")
    ax.set_title(f"channel {ch}:  ({name}, t) hit-count map, log(1+count) "
                 f"— {N_SPACE} × {N_BINS} pixels", fontsize=11, loc="left")
    ax.axvline(t0[i], color="#3ddc97", lw=1.6, ls="--")
    ax.text(t0[i] + 30, hi - 0.12 * (hi - lo), "true $t_0$", color="#3ddc97",
            fontsize=10, fontweight="bold")

ax = axes[2]
centers = t_edges[:-1] + BIN_NS / 2
ax.bar(centers, prob[i], width=BIN_NS * 0.9, color="#c9ccd1",
       label="per-column probability (softmax over the 88 columns)")
ax.bar([centers[pred[i]]], [prob[i, pred[i]]], width=BIN_NS * 0.9,
       color="#e8833a", label=f"picked column → $t_0$ estimate")
ax.axvline(t0[i], color="#3ddc97", lw=1.6, ls="--", label="true $t_0$")
ax.set_xlabel("time within the 2000 ns frame [ns]   "
              "(one pixel column = one 25 ns slice)")
ax.set_ylabel("probability")
ax.set_xlim(t_edges[0], t_edges[-1])
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.25, lw=0.5)
err = abs(centers[pred[i]] - t0[i])
ax.set_title("output of the CNN — one logit per column, softmax over the 88",
             fontsize=11, loc="left")
ax.text(0.015, 0.86, f"the collision is invisible to the eye in the image "
                     f"above:\nit is 0.5% of the hits. The net still calls it "
                     f"with p = {prob[i, pred[i]]:.2f}  (|Δt₀| = {err:.0f} ns)",
        transform=ax.transAxes, va="top", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fdf3ec", ec="#e8833a"))

fig.suptitle("The window finder reads the whole frame as a picture: "
             "2 × 64 × 88 pixels, one column per 25 ns slice", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.965])
PLOTS.mkdir(parents=True, exist_ok=True)
out = PLOTS / "finder2d_illustration.png"
fig.savefig(out, dpi=150)
print("frame index", i, "| true t0", float(t0[i]), "| pred",
      float(centers[pred[i]]), "| p =", float(prob[i, pred[i]]))
print("wrote", out)

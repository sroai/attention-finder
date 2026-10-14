"""What does the attention layer actually look at? (figure for the slides)

The window model (trkbkg/attention.py) lets every hit of a window exchange
information with every other hit of the same window. The claim we make in
talks is "a hit is judged by whether its neighbours vouch for it". This
script tests that claim on real test windows instead of asserting it.

For one window it recomputes the attention weights a_ij by hand (same maths
as _AttentionBlock.forward, but keeping the softmax matrix):

    a_ij = softmax_j ( q_i . k_j / sqrt(d) )        rows sum to 1

a_ij is "how much hit i listens to hit j".

RESULT, and it is NOT the story we used to tell ("do my neighbours vouch for
me?"): EVERY hit - signal and background alike - spends most of its attention
on the ~3% of hits that form the collision burst (18-22x more than chance).
Attention is not mutual vouching between track-mates. The burst is a LANDMARK
that the whole window triangulates against; the per-hit head then decides
whether a hit is compatible with it. The network is never told which hits are
signal, yet its attention singles them out.

(Checked and discarded: the attention-weighted DISTANCE to what a hit listens
to. Signal hits do reach shorter - median 0.8 m vs 1.9 m - but the two
distributions overlap heavily, so it is reported in the JSON, not plotted as
a discriminator.)

Plots:
  * left/middle: one real window in the (z, r) plane. Grey = background,
    orange = signal. For one QUERY hit we draw a line to every hit it listens
    to, opacity/width proportional to a_ij. Query = a signal hit (left) and
    the most confidently-rejected background hit (middle). Both stare at the
    burst.
  * right: attention RECEIVED per hit over many windows, in units of uniform
    attention (1.0 = an average, ignored hit) - the burst hits are the ones
    the window looks at.

Outputs: plots/window/attention_maps.png + reports/attention_maps.json
Run:  python scripts/710_attention_maps.py
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
from trkbkg.attention import WindowNet
from trkbkg.dataset import load_meta

DATA = (CFG.dataset_dir / "window_hits_v2")
HIT_META = (CFG.dataset_dir / "18x275_v2")
RUN = (CFG.runs_dir / "watt_r2")
PLOTS = (CFG.plots_dir / "window")
REPORTS = CFG.reports_dir
N_WINDOWS_STATS = 300      # windows used for the right-hand histogram
TOP_K_LINES = 40           # attention links drawn per query hit

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta = json.loads((DATA / "meta.json").read_text())
n_systems = len(load_meta(HIT_META)["system_ids"])

model = WindowNet(meta["norm"]["mean"], meta["norm"]["std"], n_systems,
                  d=64, n_blocks=2).to(dev)
ckpt = torch.load(RUN / "best.pt", map_location=dev)
model.load_state_dict(ckpt["model_state"])
model.eval()

d = np.load(DATA / "test.npz")
X, SYS, Y, EVT = d["X"], d["sys"].astype(np.int64), d["y"], d["evt"]
bounds = np.flatnonzero(np.diff(EVT.astype(np.int64))) + 1
starts = np.concatenate([[0], bounds])
ends = np.concatenate([bounds, [len(EVT)]])


@torch.no_grad()
def run_window(sl):
    """Forward one window, returning (attention matrices, bkg scores).

    Mirrors _AttentionBlock.forward but keeps the softmax matrix. Returns the
    attention of each block: list of [N, N] arrays, rows = queries.
    """
    x = torch.from_numpy(X[sl]).to(dev)
    sysx = torch.from_numpy(SYS[sl]).to(dev)
    h = model.encoder(x, sysx)
    atts = []
    for blk in model.blocks:
        xn = blk.norm1(h)
        a = torch.matmul(blk.q(xn), blk.k(xn).transpose(-2, -1)) * blk.scale
        a = a.softmax(dim=-1)
        atts.append(a)
        h = h + blk.out(torch.matmul(a, blk.v(xn)))
        h = h + blk.ffn(blk.norm2(h))
    score = torch.sigmoid(model.head(h).squeeze(-1))
    return [a.cpu().numpy() for a in atts], score.cpu().numpy()


# ---------------------------------------------------------------- statistics
# Per hit i we measure two things over many windows:
#   to_sig[i]  = sum_j over SIGNAL keys of a_ij  -> "how much of my attention
#                goes to the burst" (chance = the signal fraction of the window)
#   reach[i]   = sum_j a_ij * |pos_i - pos_j|    -> attention-weighted mean
#                distance to what I listen to. This is the discriminating one:
#                a signal hit listens to its own track, a background hit
#                reaches across the detector to the burst.
# Albedo (y == 2) is excluded from both classes: it is neither.
to_sig = {0: [], 1: []}
reach = {0: [], 1: []}
recv = {0: [], 1: []}
chance = []
rng = np.random.default_rng(0)
order = rng.permutation(len(starts))
used = 0
for i in order:
    s, e = starts[i], ends[i]
    n = e - s
    if not 200 <= n <= 2000:            # keep the N^2 work bounded
        continue
    y = Y[s:e]
    if (y == 0).sum() < 10:             # need some signal to measure "to_sig"
        continue
    atts, _ = run_window(slice(s, e))
    a_mean = np.mean(atts, axis=0)      # average over the 2 blocks
    is_sig = (y == 0)
    mass = a_mean[:, is_sig].sum(axis=1)
    pos = X[s:e, :3]
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    rch = (a_mean * dist).sum(axis=1)
    # attention RECEIVED, in units of "uniform attention": a hit that everyone
    # ignores scores < 1, a hit the whole window stares at scores >> 1.
    rcv = a_mean.mean(axis=0) * n
    for cls in (0, 1):
        sel = y == cls
        to_sig[cls].append(mass[sel])
        reach[cls].append(rch[sel])
        recv[cls].append(rcv[sel])
    chance.append(is_sig.mean())
    used += 1
    if used >= N_WINDOWS_STATS:
        break

sig_mass, bkg_mass = (np.concatenate(to_sig[c]) for c in (0, 1))
sig_reach, bkg_reach = (np.concatenate(reach[c]) for c in (0, 1))
sig_recv, bkg_recv = (np.concatenate(recv[c]) for c in (0, 1))
chance_level = float(np.mean(chance))
stats = {
    "windows_used": used,
    "chance_attention_to_signal": chance_level,
    "signal_hits_attention_to_signal_mean": float(sig_mass.mean()),
    "background_hits_attention_to_signal_mean": float(bkg_mass.mean()),
    "signal_over_chance": float(sig_mass.mean() / chance_level),
    "background_over_chance": float(bkg_mass.mean() / chance_level),
    # attention received, in units of uniform attention (1.0 = ignored-average)
    "signal_hits_attention_received_median": float(np.median(sig_recv)),
    "background_hits_attention_received_median": float(np.median(bkg_recv)),
    # secondary: how far a hit reaches for what it listens to (overlapping
    # distributions - reported, not used as a discriminator)
    "signal_hits_attention_reach_mm_median": float(np.median(sig_reach)),
    "background_hits_attention_reach_mm_median": float(np.median(bkg_reach)),
}
print(json.dumps(stats, indent=2))

# ------------------------------------------------------------ example window
# Pick a display window: enough signal hits to see a track, not too crowded.
best = None
for i in order:
    s, e = starts[i], ends[i]
    n = e - s
    if not 600 <= n <= 1400:
        continue
    y = Y[s:e]
    if 25 <= (y == 0).sum() <= 70:
        best = (s, e)
        break
s, e = best
atts, score = run_window(slice(s, e))
a_mean = np.mean(atts, axis=0)
x, y_, z, dt = X[s:e].T
r = np.hypot(x, y_)
lab = Y[s:e]
is_sig, is_bkg = lab == 0, lab == 1

# Query hits: the signal hit and the background hit the model is most sure of,
# both picked inside the central tracker so the display frame contains them
# (the window also holds far-forward hits out to |z| ~ 6 m; links that leave
# the frame are pointing at those).
central = (np.abs(z) < 1600) & (r < 900)
cand_sig = np.flatnonzero(is_sig & central)
cand_bkg = np.flatnonzero(is_bkg & central)
q_sig = cand_sig[np.argmin(score[cand_sig])]
q_bkg = cand_bkg[np.argmax(score[cand_bkg])]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
ZLIM = (-1900, 1900)


def draw_window(ax, q, title):
    """Scatter the window in (z, r) and draw the query hit's attention links."""
    ax.scatter(z[is_bkg], r[is_bkg], s=9, c="#c9ccd1", label="background hits",
               linewidths=0, zorder=1)
    ax.scatter(z[is_sig], r[is_sig], s=16, c="#e8833a", label="signal hits",
               linewidths=0, zorder=3)
    w = a_mean[q]
    top = np.argsort(w)[::-1][:TOP_K_LINES]
    wmax = w[top].max()
    for j in top:
        if j == q:
            continue
        rel = float(np.clip(w[j] / wmax, 0.0, 1.0))
        ax.plot([z[q], z[j]], [r[q], r[j]], lw=0.6 + 2.2 * rel, zorder=2,
                color="#1f4e79", alpha=0.25 + 0.65 * rel)
    ax.scatter([z[q]], [r[q]], s=170, facecolor="none", edgecolor="#1f4e79",
               linewidths=2.2, zorder=4, label="query hit")
    ax.set_xlim(*ZLIM)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, lw=0.5)
    frac = w[is_sig].sum()
    ax.text(0.02, 0.96, f"attention on signal hits: {frac:.0%}\n"
                        f"(signal = {is_sig.mean():.1%} of the window)",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c9ccd1"))


draw_window(axes[0], q_sig,
            "query = a SIGNAL hit → it listens to the rest of the burst")
draw_window(axes[1], q_bkg,
            "query = a BACKGROUND hit → it listens to the burst as well")
axes[0].legend(loc="lower right", fontsize=9, framealpha=0.9)

# Right panel: who gets LISTENED TO. Attention received per hit, in units of
# uniform attention (1.0 = an average, ignored hit).
ax = axes[2]
bins = np.logspace(-2.2, np.log10(300), 55)
# fraction of hits per bin (NOT density: with log bins, density divides by the
# bin width and would flatten the signal peak into invisibility)
for vals, color, lbl in ((bkg_recv, "#c9ccd1", "background hits"),
                         (sig_recv, "#e8833a", "signal hits")):
    ax.hist(np.clip(vals, bins[0], bins[-1]), bins=bins, color=color,
            alpha=0.85, label=lbl,
            weights=np.full(len(vals), 1.0 / len(vals)))
ax.axvline(1.0, color="#1f4e79", ls="--", lw=1.6,
           label="uniform attention (ignored hit)")
ax.set_xscale("log")
ax.set_xlabel("attention a hit RECEIVES  [× uniform]")
ax.set_ylabel("fraction of hits per bin")
ax.set_title("Nobody told it which hits are signal —\nyet the window stares "
             f"at them ({np.median(sig_recv):.0f}× vs "
             f"{np.median(bkg_recv):.2f}× uniform)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.25, lw=0.5)

fig.suptitle("What the attention layer actually looks at "
             f"({used} test windows): the collision burst is a LANDMARK — "
             f"every hit spends ~{100 * 0.5 * (sig_mass.mean() + bkg_mass.mean()):.0f}% "
             f"of its attention on the {chance_level:.0%} of hits that form it",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
PLOTS.mkdir(parents=True, exist_ok=True)
out = PLOTS / "attention_maps.png"
fig.savefig(out, dpi=150)
(REPORTS / "attention_maps.json").write_text(json.dumps(stats, indent=2))
print("wrote", out)

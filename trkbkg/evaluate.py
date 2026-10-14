"""Metrics, operating-point selection, and result plots.

Vocabulary (y=1 is background/"filter", y=0 is signal/"keep"; the model score
is a background probability — higher = more background-like):

* signal efficiency  = fraction of true signal hits KEPT (score < threshold).
  The physics requirement is >= 99%: throwing away real track hits damages
  reconstruction more than letting some background through.
* background rejection = fraction of true background hits REMOVED
  (score >= threshold). This is what we maximize at fixed signal efficiency.

The operating threshold is chosen on the VALIDATION fold at the target
efficiency, then applied unchanged to the TEST fold — exactly how it would be
deployed, so the reported test numbers are honest.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .systems import SYSTEM_NAMES, system_name

SIG_COLOR, BKG_COLOR = "#1a7f37", "#c0392b"


# ---------------------------------------------------------------- metrics --
def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUC via the rank-sum (Mann-Whitney) identity — no sklearn needed.

    Probability that a random background hit scores above a random signal hit.
    1.0 = perfect ranking, 0.5 = random. Albedo hits (y==2) are ignored.
    """
    if (y == 2).any():
        m = y < 2
        y, scores = y[m], scores[m]
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def efficiency_rejection_curve(y, scores, n_points=2000):
    """Sweep thresholds; return (signal_eff, bkg_rejection, thresholds).

    Uses score quantiles of the signal class as thresholds so the curve is
    dense exactly where the interesting operating points are (high signal
    efficiency).
    """
    sig_scores = scores[y == 0]
    bkg_scores = scores[y == 1]
    thr = np.quantile(sig_scores, np.linspace(0.0, 1.0, n_points))
    thr = np.unique(thr)
    # keep := score < thr
    sig_eff = np.searchsorted(np.sort(sig_scores), thr, side="left") / len(sig_scores)
    bkg_keep = np.searchsorted(np.sort(bkg_scores), thr, side="left") / len(bkg_scores)
    return sig_eff, 1.0 - bkg_keep, thr


def rejection_at_efficiency(y, scores, target_eff=0.99):
    """Background rejection at the smallest threshold with eff >= target.

    Returns (rejection, threshold).
    """
    sig_scores = np.sort(scores[y == 0])
    # threshold just above the target_eff quantile of signal keeps >= target
    k = int(np.ceil(target_eff * len(sig_scores)))
    thr = np.nextafter(sig_scores[min(k, len(sig_scores) - 1)], np.inf)
    bkg_scores = scores[y == 1]
    rejection = float((bkg_scores >= thr).mean())
    return rejection, float(thr)


def per_system_thresholds(y, scores, sys_idx, n_systems, target_eff=0.99):
    """One threshold per detector system, each guaranteeing target_eff there.

    Why: a single global threshold lets background-dominated systems trade
    away their (few) signal hits for rejection elsewhere — we observed local
    signal efficiencies as low as 63% at global eff 99%. Per-system thresholds
    make the efficiency guarantee hold in every detector. Deployment cost is
    just a lookup table next to the model.
    Systems with no signal hits in the calibration fold get threshold 1.0
    (keep everything) — the safe default.
    """
    thr = np.full(n_systems, 1.1)  # scores are <=1, so 1.1 keeps everything
    for i in range(n_systems):
        sig = np.sort(scores[(sys_idx == i) & (y == 0)])
        if len(sig):
            k = min(int(np.ceil(target_eff * len(sig))), len(sig) - 1)
            thr[i] = np.nextafter(sig[k], np.inf)
    return thr


def evaluate_fold_per_system(y, scores, sys_idx, thresholds) -> dict:
    keep = scores < thresholds[sys_idx]
    return {
        "signal_efficiency": float(keep[y == 0].mean()),
        "background_rejection": float(1.0 - keep[y == 1].mean()),
        "kept_fraction_of_all_hits": float(keep.mean()),
    }


def evaluate_fold(y, scores, threshold) -> dict:
    """Efficiency counts y==0 only; y==2 (albedo) reported separately."""
    keep = scores < threshold
    out = {
        "threshold": float(threshold),
        "signal_efficiency": float(keep[y == 0].mean()),
        "background_rejection": float(1.0 - keep[y == 1].mean()),
        "kept_fraction_of_all_hits": float(keep.mean()),
        "auc": roc_auc(y, scores),
    }
    if (y == 2).any():
        out["albedo_filtered_fraction"] = float(1.0 - keep[y == 2].mean())
        out["albedo_hits"] = int((y == 2).sum())
    return out


def per_system_table(y, scores, sys_idx, system_ids, threshold) -> dict:
    """Signal efficiency / bkg rejection per detector system at threshold."""
    out = {}
    keep = scores < threshold
    for i, sid in enumerate(system_ids):
        m = sys_idx == i
        if not m.any():
            continue
        sig, bkg = m & (y == 0), m & (y == 1)
        out[str(sid)] = {
            "name": SYSTEM_NAMES.get(int(sid), "?"),
            "hits": int(m.sum()),
            "signal_eff": float(keep[sig].mean()) if sig.any() else None,
            "bkg_rejection": float(1 - keep[bkg].mean()) if bkg.any() else None,
        }
    return out


# ------------------------------------------------------------------ plots --
def plot_training_history(history: list[dict], out_path):
    rej_key = next(k for k in history[0] if k.startswith("val_rej"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ep = [h["epoch"] for h in history]
    axes[0].plot(ep, [h["train_loss"] for h in history], "o-")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("weighted BCE loss")
    axes[0].set_title("training loss")
    axes[1].plot(ep, [h["val_auc"] for h in history], "o-", color="#2c5aa0")
    axes[1].set_xlabel("epoch"); axes[1].set_title("validation AUC")
    axes[2].plot(ep, [h[rej_key] for h in history], "o-", color=BKG_COLOR)
    axes[2].set_xlabel("epoch"); axes[2].set_title(f"validation {rej_key}")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_score_distribution(y, scores, threshold, out_path, title=""):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 120)
    ax.hist(scores[y == 1], bins=bins, density=True, alpha=0.45,
            color=BKG_COLOR, label="background")
    ax.hist(scores[y == 0], bins=bins, density=True, histtype="step",
            lw=1.8, color=SIG_COLOR, label="signal")
    ax.axvline(threshold, color="k", ls="--", lw=1,
               label=f"threshold={threshold:.3f}")
    ax.set_yscale("log"); ax.set_xlabel("background score"); ax.set_ylabel("density")
    ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_eff_rejection_curve(y, scores, target_eff, out_path, title=""):
    eff, rej, _ = efficiency_rejection_curve(y, scores)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, xlim in [(axes[0], (0.0, 1.005)), (axes[1], (0.95, 1.002))]:
        ax.plot(eff, rej, color="#2c5aa0")
        ax.axvline(target_eff, color=SIG_COLOR, ls="--", lw=1,
                   label=f"target eff={target_eff}")
        ax.set_xlim(*xlim)
        ax.set_xlabel("signal efficiency (kept)")
        ax.set_ylabel("background rejection (removed)")
        ax.grid(alpha=0.3); ax.legend()
    axes[1].set_title("zoom on operating region")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_per_system(table: dict, out_path, title=""):
    labels = [system_name(s) for s in table]
    eff = [table[s]["signal_eff"] for s in table]
    rej = [table[s]["bkg_rejection"] for s in table]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(labels, [e if e is not None else 0 for e in eff], color=SIG_COLOR)
    axes[0].axhline(0.99, color="k", ls="--", lw=1)
    axes[0].set_ylim(0.6, 1.005)
    axes[0].set_title("signal efficiency")
    axes[1].bar(labels, [r if r is not None else 0 for r in rej], color=BKG_COLOR)
    axes[1].set_title("background rejection")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def save_metrics(metrics: dict, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2))

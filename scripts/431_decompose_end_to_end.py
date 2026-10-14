#!/usr/bin/env python
"""Decompose the end-to-end result: where does background survive and why.

Usage (from the repo root):
    python scripts/431_decompose_end_to_end.py watt_v1

Answers three sanity questions about the 99% eff / 39% rejection result:
 1. The window alone rejects ~84% of background — why doesn't the pipeline?
    -> because 11.6% of signal is stranded OUTSIDE (finder misses), and
    keeping >=99% total forces a loose mlp_v1 threshold on outside hits.
 2. How much does each region contribute to kept background?
 3. What would a PERFECT finder give (same models, true t0)? -> the
    realistic upper bound that finder improvements can unlock.

Appends a "decomposition" block to runs/<run>/metrics_end_to_end.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG
from trkbkg.attention import WindowNet
from trkbkg.dataset import load_fold, load_meta
from trkbkg.models import build_model
from trkbkg.windows import BIN_NS, T_MIN, WindowFinderCNN

HIT_DATA = (CFG.dataset_dir / "18x275_v3")
WIN_DATA = (CFG.dataset_dir / "windows_v4")
WHITS = (CFG.dataset_dir / "window_hits_v3")
RUNS = CFG.runs_dir

run_name = sys.argv[1] if len(sys.argv) > 1 else "watt_v1"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hit_meta = load_meta(HIT_DATA)
wmeta = json.loads((WHITS / "meta.json").read_text())
FRONT, BACK = wmeta["front_ns"], wmeta["back_ns"]

WF_RUN = sys.argv[2] if len(sys.argv) > 2 else "wf_v2"
OUT_RUN = sys.argv[3] if len(sys.argv) > 3 else "mlp_v1"
wf_ckpt = torch.load(RUNS / WF_RUN / "best.pt", weights_only=False)
wf = WindowFinderCNN(width=wf_ckpt["config"]["width"]).to(dev).eval()
wf.load_state_dict(wf_ckpt["model_state"])
win_ckpt = torch.load(RUNS / run_name / "best.pt", weights_only=False)
wnet = WindowNet(wmeta["norm"]["mean"], wmeta["norm"]["std"],
                 len(hit_meta["system_ids"]), d=win_ckpt["config"]["dim"],
                 n_blocks=win_ckpt["config"]["blocks"]).to(dev).eval()
wnet.load_state_dict(win_ckpt["model_state"])
out_ckpt = torch.load(RUNS / OUT_RUN / "best.pt", weights_only=False)
onet = build_model(out_ckpt["config"]["model"], hit_meta,
                   **out_ckpt["config"]["model_kwargs"]).to(dev).eval()
onet.load_state_dict(out_ckpt["model_state"])


def scores_for(fold, t0_source):
    """t0_source: 'est' (wf_v2) or 'true'. Returns y, inside, scores, plus
    per-hit |finder error| (est mode) for failure attribution."""
    win = np.load(WIN_DATA / f"{fold}.npz")
    if t0_source == "est":
        with torch.no_grad():
            logits = wf(torch.from_numpy(win["frames"]).to(dev)).cpu().numpy()
        t0 = logits.argmax(axis=1) * BIN_NS + T_MIN
    else:
        t0 = win["t0"].astype(np.float64)
    err = np.abs(t0 - win["t0"]) if t0_source == "est" else np.zeros_like(t0)
    t0_map = dict(zip(win["gevt"].tolist(), t0.tolist()))
    err_map = dict(zip(win["gevt"].tolist(), err.tolist()))

    X, sysi, y, evt = load_fold(HIT_DATA, fold)
    uevt, evt_index = np.unique(evt, return_inverse=True)
    t0_hit = np.array([t0_map.get(g, np.nan) for g in uevt])[evt_index]
    err_hit = np.array([err_map.get(g, np.inf) for g in uevt])[evt_index]
    dt = X[:, 3] - t0_hit
    inside = (dt >= -FRONT) & (dt <= BACK)

    scores = np.empty(len(y), np.float32)
    with torch.no_grad():
        oi = np.flatnonzero(~inside)
        for i in range(0, len(oi), 2_000_000):
            j = oi[i:i + 2_000_000]
            scores[j] = torch.sigmoid(onet(
                torch.from_numpy(X[j]).to(dev),
                torch.from_numpy(sysi[j]).to(dev))).cpu().numpy()
        Xin = X[inside].copy(); Xin[:, 3] = dt[inside]
        sin, ein = sysi[inside], evt[inside]
        sc = np.empty(len(ein), np.float32)
        ch = np.flatnonzero(np.diff(ein.astype(np.int64))) + 1
        for s, e in zip(np.concatenate([[0], ch]), np.concatenate([ch, [len(ein)]])):
            sc[s:e] = torch.sigmoid(wnet(torch.from_numpy(Xin[s:e]).to(dev),
                                         torch.from_numpy(sin[s:e]).to(dev))).cpu().numpy()
        scores[inside] = sc
    return y, inside, scores, err_hit


def calibrate(y, inside, scores, target=0.99):
    sig_in = np.sort(scores[inside & (y == 0)])
    sig_out = np.sort(scores[~inside & (y == 0)])
    n_sig = len(sig_in) + len(sig_out)
    c = len(sig_in) / n_sig
    best = None
    for eff_in in np.linspace(max(0.9, target - 0.02), 0.9999, 40):
        need = (target - c * eff_in) / (1 - c) if c < 1 else 0.0
        if need > 1:
            continue
        need = max(need, 0.0)
        ti = np.nextafter(sig_in[min(int(np.ceil(eff_in * len(sig_in))),
                                     len(sig_in) - 1)], np.inf)
        to = (np.nextafter(sig_out[min(int(np.ceil(need * len(sig_out))),
                                       len(sig_out) - 1)], np.inf)
              if len(sig_out) else 1.1)
        keep = np.where(inside, scores < ti, scores < to)
        rej = 1 - keep[y == 1].mean()
        if best is None or rej > best[0]:
            best = (rej, ti, to)
    return best[1], best[2]


out = {}
for mode in ("est", "true"):
    yv, iv, sv, _ = scores_for("val", mode)
    ti, to = calibrate(yv, iv, sv)
    y, ins, sc, err = scores_for("test", mode)
    keep = np.where(ins, sc < ti, sc < to)
    sig, bkg = y == 0, y == 1
    d = {
        "signal_efficiency": float(keep[sig].mean()),
        "background_rejection": float(1 - keep[bkg].mean()),
        "window_alone": {
            "signal_eff_if_hard_cut": float(ins[sig].mean()),
            "bkg_rejection_if_hard_cut": float(1 - ins[bkg].mean()),
        },
        "background_budget": {
            "inside_share_of_bkg": float(ins[bkg].mean()),
            "inside_rejected": float(1 - keep[bkg & ins].mean()),
            "outside_share_of_bkg": float((~ins)[bkg].mean()),
            "outside_rejected": float(1 - keep[bkg & ~ins].mean()),
        },
        "signal_budget": {
            "inside_share_of_sig": float(ins[sig].mean()),
            "inside_kept": float(keep[sig & ins].mean()),
            "outside_share_of_sig": float((~ins)[sig].mean()),
            "outside_kept": float(keep[sig & ~ins].mean()),
        },
    }
    if mode == "est":
        far = err > 50  # hits of frames where the finder missed
        d["outside_signal_from_missed_frames"] = float(
            (far[sig & ~ins]).mean())
    out[f"finder_{mode}"] = d
    print(f"=== finder={mode} ===")
    print(json.dumps(d, indent=2))

mfile = RUNS / run_name / "metrics_end_to_end.json"
m = json.loads(mfile.read_text())
m["decomposition"] = out
mfile.write_text(json.dumps(m, indent=2))
print(f"appended decomposition to {mfile}")

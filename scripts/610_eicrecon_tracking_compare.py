#!/usr/bin/env python
"""Phase 7: compare EICrecon tracking + vertexing before/after the AI hit filter.

Reads eicrecon outputs produced by scripts/611_run_eicrecon_compare.sh
(variants: unfilt = stock tracking, finder = AI filter with ONNX window
finder, ideal = AI filter with MC-truth window; files 0022+0023, 5000 evt
each) plus the original edm4hep inputs for MC truth.

The plot set follows what the ePIC tracking WG uses to prove algorithms
under machine background (BkgImpact_PV.pdf, ePIC_Tracking_ShyamKumar PDFs,
beampipe_gold_coating_2026-2.pdf), covering all three directions:

  SEEDING      seeds_per_event
  TRACK FIT    tracks_per_event, chi2_ndf, nmeasurements, track_kinematics,
               eff_vs_eta_pbins, purity_vs_eta_pbins, signal_efficiency,
               track_truth_classes, dpp_resolution, dca_resolution
  VERTEXING    vertices_per_event, vertex_ntracks_chi2, vertex_residuals,
               vertex_resolution_vs_nmc, vertex_efficiency_vs_nmc

Definitions (matching the WG decks):
  signal primary   = MCParticle generatorStatus==1, charged, pT>0.2, |eta|<3.5
                     (background merger shifts bkg statuses to >=1000)
  track matched    = CentralCKFTrackAssociations weight > 0.5
  track purity     = matched-to-signal tracks / all tracks (per eta/p bin)
  selected vertex  = reco vertex with smallest chi2/ndf in the event
  vertex found     = selected vertex within 1 mm (3D) of the MC primary vertex
  resolutions      = robust sigma (half the central 68% interval) of residuals

Events are joined between output and input files via EventHeader.eventNumber
(multithreaded eicrecon does not preserve event order); files are processed
independently and aggregated.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkbkg.config import CFG

REPORTS = CFG.reports_dir

# --dataset selects everything path-related. "klambda" is the original
# 18x275 exclusive sample; "dis26" the 26.07.0 NC-DIS minQ2=1000 campaign
# files (99 frames/file, many charged primaries -> vs-N_MC plots enabled).
DATASETS = {
    "klambda": {
        "results": (CFG.results_root / "eicrecon"),
        "inputs": lambda: {f: (CFG.dd4hep_dir / "18x275")
                           / f"k_lambda_18x275_5000evt_{f}.edm4hep.root"
                           for f in ["0022", "0023"]},
        "plots": (CFG.plots_dir / "phase7"),
        "json": "phase7_tracking_compare.json",
        "vs_nmc": False,  # N_MC ~ 2 for every event, axis degenerate
    },
    "dis26": {
        "results": (CFG.results_root / "eicrecon-dis"),
        "inputs": lambda: {
            "1" + re.search(r"1\.(\d+)\.edm4hep", p.name).group(1): p
            for p in sorted(Path("/data/background-2026-07").glob("pythia8NCDIS_*.edm4hep.root"))},
        "plots": (CFG.plots_dir / "phase7_dis"),
        "json": "phase7_tracking_compare_dis.json",
        "vs_nmc": True,
    },
}

# set in main() from --dataset
RESULTS: Path
PLOTS: Path
FILES: list
INPUT_OF: dict
VS_NMC = False

# dataviz reference palette, fixed slot order (1 blue, 2 aqua, 3 yellow)
VARIANTS = {
    "unfilt": {"label": "unfiltered (stock)", "color": "#2a78d6"},
    "finder": {"label": "AI filter, ONNX window finder", "color": "#1baf7a"},
    "ideal": {"label": "AI filter, ideal (MC) window", "color": "#eda100"},
}
TEXT = "#0b0b0b"
GRID = dict(color="#d9d8d4", linewidth=0.6)

ETA_MAX = 3.5
PT_MIN = 0.2  # GeV
MATCH_WEIGHT = 0.5
P_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 5.0), (5.0, 1000.0)]  # Shujie's bins
ETA_REGIONS = [(-3.5, -1.0, "backward"), (-1.0, 1.0, "central"), (1.0, 3.5, "forward")]
VTX_MATCH_MM = 1.0


def b(t, name):
    return t[name].array()


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def read_variant_file(tag: str, fno: str) -> dict:
    """Per-event tracking + vertexing quantities from one eicrecon output."""
    t = uproot.open(RESULTS / f"{tag}_{fno}.root")["events"]
    d = {
        "evt_no": ak.flatten(b(t, "EventHeader/EventHeader.eventNumber")).to_numpy(),
        "n_seeds": ak.num(b(t, "CentralTrackSeeds/CentralTrackSeeds.perigee.x")).to_numpy(),
        "chi2": b(t, "CentralCKFTracks/CentralCKFTracks.chi2"),
        "ndf": b(t, "CentralCKFTracks/CentralCKFTracks.ndf"),
        "nmeas": b(t, "CentralCKFTrajectories/CentralCKFTrajectories.nMeasurements"),
        "loc_a": b(t, "CentralCKFTrackParameters/CentralCKFTrackParameters.loc.a"),
        "theta": b(t, "CentralCKFTrackParameters/CentralCKFTrackParameters.theta"),
        "phi": b(t, "CentralCKFTrackParameters/CentralCKFTrackParameters.phi"),
        "qoverp": b(t, "CentralCKFTrackParameters/CentralCKFTrackParameters.qOverP"),
        "aw": b(t, "CentralCKFTrackAssociations/CentralCKFTrackAssociations.weight"),
        "arec": b(t, "_CentralCKFTrackAssociations_rec/_CentralCKFTrackAssociations_rec.index"),
        "asim": b(t, "_CentralCKFTrackAssociations_sim/_CentralCKFTrackAssociations_sim.index"),
        "vx": b(t, "CentralTrackVertices/CentralTrackVertices.position.x"),
        "vy": b(t, "CentralTrackVertices/CentralTrackVertices.position.y"),
        "vz": b(t, "CentralTrackVertices/CentralTrackVertices.position.z"),
        "vchi2": b(t, "CentralTrackVertices/CentralTrackVertices.chi2"),
        "vndf": b(t, "CentralTrackVertices/CentralTrackVertices.ndf"),
        "vnp_b": b(t, "CentralTrackVertices/CentralTrackVertices.associatedParticles_begin"),
        "vnp_e": b(t, "CentralTrackVertices/CentralTrackVertices.associatedParticles_end"),
    }
    return d


def read_truth_file(fno: str, needed_sim: dict[int, set]) -> dict:
    """One chunked pass over one input file's MCParticles.

    Returns per-event dicts keyed by event number:
      signal[evt] = dict(idx, pt, eta) of signal primaries (for efficiency)
      status[evt] = {idx: generatorStatus} for association-referenced indices
      pmag[evt]   = {idx: |p|} for the same indices (for dp/p)
      pv[evt]     = (x, y, z) MC primary vertex (scattered-electron vertex,
                    fallback: most frequent status-1 vertex)
      nmc[evt]    = number of charged status-1 primaries with |eta|<3.5
                    (vertexing-deck denominator; no pT cut)
    """
    out = {"signal": {}, "status": {}, "pmag": {}, "pv": {}, "nmc": {}}
    B = {
        "no": "EventHeader/EventHeader.eventNumber",
        "st": "MCParticles/MCParticles.generatorStatus",
        "q": "MCParticles/MCParticles.charge",
        "pdg": "MCParticles/MCParticles.PDG",
        "px": "MCParticles/MCParticles.momentum.x",
        "py": "MCParticles/MCParticles.momentum.y",
        "pz": "MCParticles/MCParticles.momentum.z",
        "vx": "MCParticles/MCParticles.vertex.x",
        "vy": "MCParticles/MCParticles.vertex.y",
        "vz": "MCParticles/MCParticles.vertex.z",
    }
    t = uproot.open(INPUT_OF[fno])["events"]
    for chunk in t.iterate(list(B.values()), step_size=200):
        nos = ak.flatten(chunk[B["no"]]).to_numpy()
        for k in range(len(nos)):
            evt = int(nos[k])
            st = chunk[B["st"]][k].to_numpy()
            q = chunk[B["q"]][k].to_numpy()
            px = chunk[B["px"]][k].to_numpy()
            py = chunk[B["py"]][k].to_numpy()
            pz = chunk[B["pz"]][k].to_numpy()
            pt = np.hypot(px, py)
            with np.errstate(divide="ignore", invalid="ignore"):
                eta = np.arcsinh(np.where(pt > 0, pz / np.maximum(pt, 1e-12), np.inf))
            prim = st == 1
            charged = q != 0
            sel = prim & charged & (pt > PT_MIN) & (np.abs(eta) < ETA_MAX)
            idx = np.nonzero(sel)[0]
            out["signal"][evt] = {"idx": idx, "pt": pt[idx], "eta": eta[idx]}
            out["nmc"][evt] = int((prim & charged & (np.abs(eta) < ETA_MAX)).sum())

            vx = chunk[B["vx"]][k].to_numpy()
            vy = chunk[B["vy"]][k].to_numpy()
            vz = chunk[B["vz"]][k].to_numpy()
            pdg = chunk[B["pdg"]][k].to_numpy()
            e_mask = prim & (pdg == 11)
            if e_mask.any():
                j = int(np.nonzero(e_mask)[0][0])
                out["pv"][evt] = (float(vx[j]), float(vy[j]), float(vz[j]))
            elif prim.any():
                j_all = np.nonzero(prim)[0]
                key = np.round(vz[j_all], 3)
                vals, counts = np.unique(key, return_counts=True)
                j = int(j_all[np.argmax(key == vals[np.argmax(counts)])])
                out["pv"][evt] = (float(vx[j]), float(vy[j]), float(vz[j]))

            want = needed_sim.get(evt)
            if want:
                p = np.sqrt(px * px + py * py + pz * pz)
                out["status"][evt] = {int(i): int(st[i]) for i in want if i < len(st)}
                out["pmag"][evt] = {int(i): float(p[i]) for i in want if i < len(st)}
    return out


# --------------------------------------------------------------------------
# matching / flattening
# --------------------------------------------------------------------------
def build_tables(d: dict, truth: dict) -> dict:
    """Flatten one (variant, file) pair into per-particle / per-track /
    per-event arrays with truth attached."""
    T = {  # per matched-denominator signal primary
        "eff_pt": [], "eff_eta": [], "eff_hit": [],
        # per track
        "trk_eta": [], "trk_p": [], "trk_pt": [], "trk_sig": [], "trk_nmeas": [],
        "trk_chi2ndf": [], "trk_dpp": [], "trk_d0res": [],
        # per event (vertexing)
        "nvtx": [], "nmc": [], "vres": [],  # vres = (dx,dy,dz) of selected vtx or None
        "vtx_ntrk": [], "vtx_chi2ndf": [],
    }
    for k, evt in enumerate(d["evt_no"]):
        evt = int(evt)
        w = d["aw"][k].to_numpy()
        rec = d["arec"][k].to_numpy()
        sim = d["asim"][k].to_numpy()
        good = w > MATCH_WEIGHT
        matched_sim = set(sim[good].tolist())

        sig = truth["signal"][evt]
        T["eff_pt"].append(sig["pt"])
        T["eff_eta"].append(sig["eta"])
        T["eff_hit"].append(np.array([int(i) in matched_sim for i in sig["idx"]], dtype=bool))

        # per-track best association
        best: dict[int, tuple[float, int]] = {}
        for wi, ri, si in zip(w, rec, sim):
            if ri not in best or wi > best[ri][0]:
                best[ri] = (wi, si)
        st_map = truth["status"].get(evt, {})
        p_map = truth["pmag"].get(evt, {})
        pv = truth["pv"].get(evt)

        th = d["theta"][k].to_numpy()
        ph = d["phi"][k].to_numpy()
        qp = d["qoverp"][k].to_numpy()
        d0 = d["loc_a"][k].to_numpy()
        chi2 = d["chi2"][k].to_numpy()
        ndf = d["ndf"][k].to_numpy()
        nmeas = d["nmeas"][k].to_numpy()
        n_trk = len(th)
        p_rec = 1.0 / np.abs(qp)
        for ri in range(n_trk):
            T["trk_eta"].append(-np.log(np.tan(th[ri] / 2.0)))
            T["trk_p"].append(p_rec[ri])
            T["trk_pt"].append(p_rec[ri] * np.sin(th[ri]))
            T["trk_chi2ndf"].append(chi2[ri] / max(ndf[ri], 1))
            T["trk_nmeas"].append(nmeas[ri] if ri < len(nmeas) else -1)
            is_sig, dpp, d0res = False, np.nan, np.nan
            if ri in best and best[ri][0] > MATCH_WEIGHT:
                si = int(best[ri][1])
                if st_map.get(si) in (1, 2):
                    is_sig = True
                    p_true = p_map.get(si)
                    if p_true:
                        dpp = (p_rec[ri] - p_true) / p_true
                    if pv is not None:
                        # ACTS perigee convention (checked empirically on this
                        # data: wrong sign gives 25x wider residuals)
                        d0_true = pv[1] * np.cos(ph[ri]) - pv[0] * np.sin(ph[ri])
                        d0res = d0[ri] - d0_true
            T["trk_sig"].append(is_sig)
            T["trk_dpp"].append(dpp)
            T["trk_d0res"].append(d0res)

        # vertexing
        vchi2 = d["vchi2"][k].to_numpy()
        vndf = d["vndf"][k].to_numpy()
        vx = d["vx"][k].to_numpy()
        vy = d["vy"][k].to_numpy()
        vz = d["vz"][k].to_numpy()
        vnp = (d["vnp_e"][k].to_numpy() - d["vnp_b"][k].to_numpy())
        T["nvtx"].append(len(vx))
        T["nmc"].append(truth["nmc"].get(evt, 0))
        T["vtx_ntrk"].extend(vnp.tolist())
        with np.errstate(divide="ignore", invalid="ignore"):
            vq = vchi2 / np.maximum(vndf, 1)
        T["vtx_chi2ndf"].extend(vq.tolist())
        res = None
        if len(vx) and pv is not None:
            j = int(np.argmin(vq))  # WG convention: smallest chi2/ndf vertex
            res = (vx[j] - pv[0], vy[j] - pv[1], vz[j] - pv[2])
        T["vres"].append(res)
    return T


def concat_tables(parts: list[dict]) -> dict:
    out = {}
    for key in parts[0]:
        vals = []
        for p in parts:
            vals.extend(p[key])
        if key in ("eff_pt", "eff_eta", "eff_hit"):
            out[key] = np.concatenate([np.asarray(v) for v in vals]) if vals else np.array([])
        elif key == "vres":
            out[key] = vals
        else:
            out[key] = np.asarray(vals)
    return out


# --------------------------------------------------------------------------
# plotting helpers
# --------------------------------------------------------------------------
def style(ax):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def robust_sigma(x):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan, np.nan
    lo, hi = np.percentile(x, [16, 84])
    sig = 0.5 * (hi - lo)
    return sig, sig / np.sqrt(2 * max(len(x) - 1, 1))


def binned_ratio(ax, x, hit, bins, color, label, min_n=20):
    idx = np.digitize(x, bins) - 1
    centers, eff, err = [], [], []
    for i in range(len(bins) - 1):
        m = idx == i
        n = m.sum()
        if n < min_n:
            continue
        kk = hit[m].sum()
        centers.append(0.5 * (bins[i] + bins[i + 1]))
        eff.append(kk / n)
        err.append(np.sqrt(max(kk, 1) * (1 - kk / n)) / n)
    ax.errorbar(centers, eff, yerr=err, color=color, label=label,
                marker="o", markersize=4, linewidth=2, capsize=2)


def binned_sigma(ax, x, val, bins, color, label, scale=1.0, min_n=40):
    idx = np.digitize(x, bins) - 1
    centers, sig, err = [], [], []
    for i in range(len(bins) - 1):
        m = (idx == i) & np.isfinite(val)
        if m.sum() < min_n:
            continue
        s, e = robust_sigma(val[m])
        centers.append(np.sqrt(bins[i] * bins[i + 1]) if bins[i] > 0 else 0.5 * (bins[i] + bins[i + 1]))
        sig.append(s * scale)
        err.append(e * scale)
    ax.errorbar(centers, sig, yerr=err, color=color, label=label,
                marker="o", markersize=4, linewidth=2, capsize=2)


def overlay_hist(fname, series, xlabel, bins, xlog=False, ylog=False, density=False):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tag, values in series.items():
        ax.hist(values, bins=bins, histtype="step", linewidth=2, density=density,
                color=VARIANTS[tag]["color"], label=VARIANTS[tag]["label"])
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel("fraction of events" if density else "entries", color=TEXT)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    global RESULTS, PLOTS, FILES, INPUT_OF, VS_NMC
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="klambda")
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    RESULTS, PLOTS, VS_NMC = ds["results"], ds["plots"], ds["vs_nmc"]
    INPUT_OF = ds["inputs"]()
    # analyze the files for which all three variants were produced
    FILES = [f for f in INPUT_OF
             if all((RESULTS / f"{tag}_{f}.root").exists() for tag in VARIANTS)]
    print(f"dataset {args.dataset}: {len(FILES)} files: {FILES}")

    PLOTS.mkdir(parents=True, exist_ok=True)

    # ---- read all outputs, gather needed truth indices per file ----
    raw = {tag: {fno: read_variant_file(tag, fno) for fno in FILES} for tag in VARIANTS}
    truth = {}
    for fno in FILES:
        needed: dict[int, set] = {}
        for tag in VARIANTS:
            d = raw[tag][fno]
            for k, evt in enumerate(d["evt_no"]):
                needed.setdefault(int(evt), set()).update(
                    int(s) for s in d["asim"][k].to_numpy())
        print(f"truth pass over file {fno}...")
        truth[fno] = read_truth_file(fno, needed)

    tables = {tag: concat_tables([build_tables(raw[tag][fno], truth[fno]) for fno in FILES])
              for tag in VARIANTS}
    data = {tag: {"n_seeds": np.concatenate([raw[tag][f]["n_seeds"] for f in FILES]),
                  "n_tracks": np.concatenate([ak.num(raw[tag][f]["ndf"]).to_numpy() for f in FILES])}
            for tag in VARIANTS}

    # ================= SEEDING =================
    overlay_hist("seeds_per_event.png", {t: data[t]["n_seeds"] for t in VARIANTS},
                 "track seeds per event", np.arange(0, 800, 20))

    # ================= TRACKS =================
    overlay_hist("tracks_per_event.png", {t: data[t]["n_tracks"] for t in VARIANTS},
                 "CKF tracks per event", np.arange(-0.5, 15.5, 1))
    overlay_hist("chi2_ndf.png", {t: tables[t]["trk_chi2ndf"] for t in VARIANTS},
                 r"track $\chi^2$/ndf", np.linspace(0, 10, 50))
    overlay_hist("nmeasurements.png", {t: tables[t]["trk_nmeas"] for t in VARIANTS},
                 "measurements per track", np.arange(-0.5, 20.5, 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, tb in tables.items():
        axes[0].hist(tb["trk_pt"], bins=np.logspace(-1, 1.3, 40), histtype="step",
                     linewidth=2, color=VARIANTS[tag]["color"], label=VARIANTS[tag]["label"])
        axes[1].hist(tb["trk_eta"], bins=np.linspace(-4, 4, 40), histtype="step",
                     linewidth=2, color=VARIANTS[tag]["color"], label=VARIANTS[tag]["label"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("track $p_T$ [GeV]", color=TEXT)
    axes[1].set_xlabel(r"track $\eta$", color=TEXT)
    for ax in axes:
        ax.set_ylabel("tracks", color=TEXT)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "track_kinematics.png", dpi=150)
    plt.close(fig)

    # ---- efficiency & purity vs eta in momentum bins (Shujie-style) ----
    eta_bins = np.linspace(-3.5, 3.5, 15)
    for kind in ("eff", "purity"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
        for (plo, phi_), ax in zip(P_BINS, axes.ravel()):
            for tag, tb in tables.items():
                if kind == "eff":
                    # denominator: signal primaries; p from pt/eta
                    p_all = tb["eff_pt"] * np.cosh(tb["eff_eta"])
                    m = (p_all >= plo) & (p_all < phi_)
                    binned_ratio(ax, tb["eff_eta"][m], tb["eff_hit"][m], eta_bins,
                                 VARIANTS[tag]["color"], VARIANTS[tag]["label"])
                else:
                    m = (tb["trk_p"] >= plo) & (tb["trk_p"] < phi_)
                    binned_ratio(ax, tb["trk_eta"][m], tb["trk_sig"][m], eta_bins,
                                 VARIANTS[tag]["color"], VARIANTS[tag]["label"])
            ax.set_title(f"{plo} < p < {phi_} GeV", fontsize=9, color=TEXT)
            ax.set_ylim(0, 1.1)
            style(ax)
        for ax in axes[1]:
            ax.set_xlabel(r"$\eta$", color=TEXT)
        lab = "track efficiency (signal primaries)" if kind == "eff" else "track purity (signal fraction)"
        for ax in axes[:, 0]:
            ax.set_ylabel(lab, color=TEXT)
        axes[0, 0].legend(frameon=False, fontsize=8, loc="lower center")
        fig.tight_layout()
        fig.savefig(PLOTS / f"{kind}_vs_eta_pbins.png", dpi=150)
        plt.close(fig)

    # ---- overall efficiency vs pT / eta (kept from v1) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    pt_bins = np.array([0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0, 7.0, 12.0])
    for tag, tb in tables.items():
        binned_ratio(axes[0], tb["eff_pt"], tb["eff_hit"], pt_bins,
                     VARIANTS[tag]["color"], VARIANTS[tag]["label"])
        binned_ratio(axes[1], tb["eff_eta"], tb["eff_hit"], eta_bins,
                     VARIANTS[tag]["color"], VARIANTS[tag]["label"])
    axes[0].set_xlabel("signal primary $p_T$ [GeV]", color=TEXT)
    axes[0].set_xscale("log")
    axes[1].set_xlabel(r"signal primary $\eta$", color=TEXT)
    for ax in axes:
        ax.set_ylabel("track efficiency", color=TEXT)
        ax.set_ylim(0, 1.05)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "signal_efficiency.png", dpi=150)
    plt.close(fig)

    # ---- dp/p and DCA_r resolution vs p / pT in eta regions ----
    p_res_bins = np.array([0.3, 0.6, 1.0, 2.0, 4.0, 8.0, 15.0])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for (elo, ehi, name), ax in zip(ETA_REGIONS, axes):
        for tag, tb in tables.items():
            m = (tb["trk_eta"] >= elo) & (tb["trk_eta"] < ehi) & tb["trk_sig"]
            binned_sigma(ax, tb["trk_p"][m], tb["trk_dpp"][m], p_res_bins,
                         VARIANTS[tag]["color"], VARIANTS[tag]["label"], scale=100.0)
        ax.set_title(f"{name}  ({elo} < $\\eta$ < {ehi})", fontsize=9, color=TEXT)
        ax.set_xlabel("p [GeV]", color=TEXT)
        ax.set_xscale("log")
        style(ax)
    axes[0].set_ylabel(r"$\sigma(\Delta p/p)$ [%]", color=TEXT)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "dpp_resolution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for (elo, ehi, name), ax in zip(ETA_REGIONS, axes):
        for tag, tb in tables.items():
            m = (tb["trk_eta"] >= elo) & (tb["trk_eta"] < ehi) & tb["trk_sig"]
            binned_sigma(ax, tb["trk_pt"][m], tb["trk_d0res"][m], p_res_bins,
                         VARIANTS[tag]["color"], VARIANTS[tag]["label"], scale=1000.0)
        ax.set_title(f"{name}  ({elo} < $\\eta$ < {ehi})", fontsize=9, color=TEXT)
        ax.set_xlabel("$p_T$ [GeV]", color=TEXT)
        ax.set_xscale("log")
        style(ax)
    axes[0].set_ylabel(r"$\sigma(DCA_r)$ [$\mu$m]", color=TEXT)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "dca_resolution.png", dpi=150)
    plt.close(fig)

    # ---- track truth classes (kept from v1) ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = np.arange(2)
    width = 0.26
    for j, (tag, tb) in enumerate(tables.items()):
        n = len(tb["trk_sig"])
        vals = [tb["trk_sig"].sum() / n, (~tb["trk_sig"]).sum() / n]
        bars = ax.bar(xs + (j - 1) * width, vals, width * 0.92,
                      color=VARIANTS[tag]["color"], label=VARIANTS[tag]["label"])
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.0%}",
                    ha="center", fontsize=7, color=TEXT)
    ax.set_xticks(xs, ["signal (matched, status 1/2)", "other (bkg / secondary / unmatched)"])
    ax.set_ylabel("fraction of CKF tracks", color=TEXT)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "track_truth_classes.png", dpi=150)
    plt.close(fig)

    # ================= VERTEXING =================
    overlay_hist("vertices_per_event.png", {t: tables[t]["nvtx"] for t in VARIANTS},
                 "reconstructed vertices per event", np.arange(-0.5, 8.5, 1), density=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, tb in tables.items():
        axes[0].hist(tb["vtx_ntrk"], bins=np.arange(-0.5, 30.5, 1), histtype="step",
                     linewidth=2, color=VARIANTS[tag]["color"], label=VARIANTS[tag]["label"])
        axes[1].hist(np.clip(tb["vtx_chi2ndf"], 0, 49), bins=np.linspace(0, 50, 50),
                     histtype="step", linewidth=2,
                     color=VARIANTS[tag]["color"], label=VARIANTS[tag]["label"])
    axes[0].set_xlabel("tracks used in vertex fit", color=TEXT)
    axes[1].set_xlabel(r"vertex $\chi^2$/ndf", color=TEXT)
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_ylabel("vertices", color=TEXT)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "vertex_ntracks_chi2.png", dpi=150)
    plt.close(fig)

    # residuals of the selected (smallest chi2/ndf) vertex
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    comps = ["x", "y", "z"]
    for ci, ax in enumerate(axes):
        for tag, tb in tables.items():
            res = np.array([r[ci] for r in tb["vres"] if r is not None])
            ax.hist(np.clip(res, -0.5, 0.5), bins=np.linspace(-0.5, 0.5, 80),
                    histtype="step", linewidth=2, density=True,
                    color=VARIANTS[tag]["color"], label=VARIANTS[tag]["label"])
        ax.set_xlabel(f"vertex {comps[ci]} (reco - MC) [mm]", color=TEXT)
        ax.set_yscale("log")
        style(ax)
    axes[0].set_ylabel("normalized", color=TEXT)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "vertex_residuals.png", dpi=150)
    plt.close(fig)

    # vertex summary: efficiency + core resolutions per variant.
    # (The WG decks plot these vs N_MC charged primaries, but the exclusive
    # k_lambda signal has N_MC ~ 2 in every event, so the axis is degenerate
    # here - single numbers per variant are the honest presentation.)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    xs = np.arange(len(VARIANTS))
    labels4 = [f"P(vtx within {VTX_MATCH_MM} mm of MC)",
               r"$\sigma(v_x)$ [$\mu$m]", r"$\sigma(v_y)$ [$\mu$m]", r"$\sigma(v_z)$ [$\mu$m]"]
    for j, (tag, tb) in enumerate(tables.items()):
        found = np.array([r is not None and np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) < VTX_MATCH_MM
                          for r in tb["vres"]])
        vals = [found.mean()]
        errs = [np.sqrt(found.mean() * (1 - found.mean()) / len(found))]
        for ci in range(3):
            res = np.array([r[ci] for r in tb["vres"] if r is not None])
            s, e = robust_sigma(res[np.abs(res) < 0.2])  # WG: fit core within +-0.2 mm
            vals.append(s * 1000.0)
            errs.append(e * 1000.0)
        for ax, v, e in zip(axes, vals, errs):
            bar = ax.bar([j], [v], 0.7, yerr=[e], capsize=3,
                         color=VARIANTS[tag]["color"], label=VARIANTS[tag]["label"])
            fmt = f"{v:.1%}" if ax is axes[0] else f"{v:.1f}"
            ax.text(j, v * 1.01, fmt, ha="center", fontsize=8, color=TEXT)
    for ax, lab in zip(axes, labels4):
        ax.set_ylabel(lab, color=TEXT)
        ax.set_xticks(xs, ["unfilt", "finder", "ideal"])
        style(ax)
    axes[0].set_ylim(0, 1.1)
    fig.suptitle("selected vertex (smallest $\\chi^2$/ndf) vs MC primary vertex",
                 fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(PLOTS / "vertex_summary.png", dpi=150)
    plt.close(fig)

    # WG-style vertex performance vs N_MC (only for datasets with real
    # multiplicity spread, e.g. DIS)
    if VS_NMC:
        nmc_bins = np.array([0, 2, 4, 6, 8, 10, 13, 16, 20, 26, 34])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for tag, tb in tables.items():
            found = np.array([r is not None and
                              np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) < VTX_MATCH_MM
                              for r in tb["vres"]])
            binned_ratio(axes[0], tb["nmc"], found, nmc_bins,
                         VARIANTS[tag]["color"], VARIANTS[tag]["label"], min_n=10)
            resz = np.array([r[2] if r is not None else np.nan for r in tb["vres"]])
            ok = np.isfinite(resz) & (np.abs(resz) < 0.2)
            binned_sigma(axes[1], tb["nmc"][ok], resz[ok], nmc_bins,
                         VARIANTS[tag]["color"], VARIANTS[tag]["label"],
                         scale=1000.0, min_n=20)
        axes[0].set_ylabel(f"P(vtx within {VTX_MATCH_MM} mm of MC)", color=TEXT)
        axes[0].set_ylim(0, 1.1)
        axes[1].set_ylabel(r"$\sigma(v_z)$ core [$\mu$m]", color=TEXT)
        for ax in axes:
            ax.set_xlabel(r"$N_{MC}$ charged primaries ($|\eta|<3.5$)", color=TEXT)
            style(ax)
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(PLOTS / "vertex_perf_vs_nmc.png", dpi=150)
        plt.close(fig)

    # ================= summary =================
    summary = {}
    for tag, tb in tables.items():
        found = np.array([r is not None and np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) < VTX_MATCH_MM
                          for r in tb["vres"]])
        has_v = np.array([r is not None for r in tb["vres"]])
        resz = np.array([r[2] for r in tb["vres"] if r is not None])
        sz, _ = robust_sigma(resz[np.abs(resz) < 0.2])
        dpp = tb["trk_dpp"][tb["trk_sig"] & np.isfinite(tb["trk_dpp"])]
        sdpp, _ = robust_sigma(dpp)
        d0 = tb["trk_d0res"][tb["trk_sig"] & np.isfinite(tb["trk_d0res"])]
        sd0, _ = robust_sigma(d0)
        summary[tag] = {
            "events": int(len(tb["nvtx"])),
            "seeds_per_event_mean": float(data[tag]["n_seeds"].mean()),
            "tracks_per_event_mean": float(data[tag]["n_tracks"].mean()),
            "signal_primaries": int(len(tb["eff_hit"])),
            "signal_track_efficiency": float(tb["eff_hit"].mean()),
            "track_purity_signal_fraction": float(tb["trk_sig"].mean()),
            "nmeas_mean": float(np.mean(tb["trk_nmeas"][tb["trk_nmeas"] >= 0])),
            "dpp_sigma_percent": float(sdpp * 100),
            "dca_r_sigma_um": float(sd0 * 1000),
            "vertices_per_event_mean": float(tb["nvtx"].mean()),
            "events_with_vertex": float(has_v.mean()),
            "vertex_eff_1mm": float(found.mean()),
            "vertex_z_sigma_um": float(sz * 1000),
        }
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / ds["json"], "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

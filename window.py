#!/usr/bin/env python3
"""Where the 8B-versus-dark-matter degeneracy in xenon is exact, by halo class.

Self-consistent targets: the 8B recoil spectrum folded through the same
response that builds the dark-matter generators, so a residual measures halo
geometry and nothing else.

  released   XENONnT SR0+SR1 cS1-cS2 templates, ROI 0.51-5 keV.
  ideal      an energy-resolving xenon detector with bin width dE and threshold E_thr.

Outputs results/window.json and results/window.png.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import analysis as A
from isotropic import generator_matrix, l2_residual, poisson_fit

OUT = A.OUT
V_SHELL = np.concatenate([[0.0], np.geomspace(2.0, A.V_ESC, 400)])
V_STREAM = np.geomspace(10.0, A.V_MAX, 400)
E_MAX = A.E_ROI[1]
TOL = 1e-3


def ideal_response(E, E_thr, dE):
    edges = np.arange(E_thr, E_MAX + 1e-9, dE)
    if edges[-1] < E_MAX - 1e-9:
        edges = np.append(edges, E_MAX)
    T = np.zeros((len(E), len(edges) - 1))
    inside = (E >= edges[0]) & (E < edges[-1])
    which = np.clip(np.digitize(E, edges) - 1, 0, len(edges) - 2)
    T[np.where(inside)[0], which[inside]] = 1.0
    return T


def mass_for(E_keV, v_kms, A_nuc=131.293):
    """Mass whose v_min at E_keV equals v_kms (xenon)."""
    m_N = A_nuc * A.AMU
    mu = np.sqrt(m_N * E_keV * 1e-6 / 2.0) * A.C_KM / v_kms
    return float(mu * m_N / (m_N - mu)) if mu < m_N else float("inf")


def window(rows, tol=TOL):
    inside = [r["mass"] for r in rows if r["residual"] < tol]
    return (min(inside), max(inside)) if inside else None


def scan_masses(E, T, dR, masses, kind, vgrid, with_deviance=False):
    phi = A.fold_spectrum(E, dR, E, T)
    rows = []
    for m in masses:
        G = generator_matrix(E, T, float(m), A.XE_ISOTOPES, vgrid, kind, A.V_EARTH)
        resid, _ = l2_residual(G, phi)
        row = dict(mass=float(m), residual=resid)
        if with_deviance:
            dev1, _ = poisson_fit(G, phi)
            row["deviance_per_ty"] = float(dev1) if np.isfinite(dev1) else 1e300
        rows.append(row)
    return rows


def _plain_label(v):
    if v <= 0 or not math.isfinite(v):
        return ""
    if abs(v - round(v)) < 1e-6 * max(v, 1):
        return str(int(round(v)))
    return f"{v:.3g}"


def _decade_label(v):
    if v <= 0 or not math.isfinite(v):
        return ""
    exp = int(round(math.log10(v)))
    if abs(v - 10.0 ** exp) > 1e-6 * v:
        return ""
    if -2 <= exp <= 2:
        return _plain_label(10.0 ** exp)
    return rf"$10^{{{exp}}}$"


def apply_log_axis(ax, which, lo, hi, ticks=None):
    """Log axis whose labels do not collide.

    A span of about one decade gets a few plain numbers (0.2, 0.5, 1, 10).
    A wider span labels decades only, as 1, 10, 100 or 10^n. Minor ticks are
    drawn on wide spans and never labelled. Pass `ticks` to force plain labels.
    """
    import matplotlib.ticker as mticker

    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    (ax.set_xlim if which == "x" else ax.set_ylim)(lo, hi)
    decades = math.log10(hi / lo)
    axis.set_minor_formatter(mticker.NullFormatter())
    axis.get_offset_text().set_visible(False)

    if ticks is not None:
        mode = "plain"
    elif decades <= 1.55:
        plain = [0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5, 6, 8, 10, 15, 20, 30, 40, 50, 60, 100]
        ticks = [t for t in plain if lo <= t <= hi]
        min_sep = max(decades / 6.0, 0.09)
        kept = []
        for t in ticks:
            if not kept or math.log10(t / kept[-1]) >= min_sep:
                kept.append(t)
        ticks = kept or [lo, hi]
        mode = "plain"
    else:
        e0 = math.ceil(math.log10(lo) - 1e-9)
        e1 = math.floor(math.log10(hi) + 1e-9)
        cap = 5 if which == "x" else 8
        step = 1
        while step < 8 and len(list(range(e0, e1 + 1, step))) > cap:
            step += 1
        ticks = [10.0 ** e for e in range(e0, e1 + 1, step)]
        mode = "decade"

    axis.set_major_locator(mticker.FixedLocator(list(ticks)))
    if mode == "plain":
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: _plain_label(v)))
        axis.set_minor_locator(mticker.NullLocator())
    else:
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: _decade_label(v)))
        axis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))
    return ticks


def warn_overlaps(fig):
    """Print tick, title, annotation, and legend collisions in display space."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    arts = []
    for ax in fig.axes:
        texts = list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        texts += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        for t in texts:
            label = t.get_text().replace("\n", " ").strip()
            if not t.get_visible() or not label:
                continue
            arts.append((label[:48], t.get_window_extent(rend).expanded(1.05, 1.18)))
        leg = ax.get_legend()
        if leg is not None:
            lb = leg.get_window_extent(rend)
            if lb.overlaps(ax.get_window_extent(rend)):
                print("LEGEND OVERLAPS AXES")
            arts.append(("legend", lb))
    for i in range(len(arts)):
        for j in range(i + 1, len(arts)):
            if arts[i][1].overlaps(arts[j][1]):
                print(f"OVERLAP [{arts[i][0]}] x [{arts[j][0]}]")


def plot_window(summary):
    """Diagnostic figure for the exact-degeneracy window. Reads a window.json dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parent / "config"))
    import figstyle as fs

    fs.set_style()
    iso, any_h = fs.role("isotropic"), fs.role("anyhalo")
    edge, shade = fs.role("edge"), fs.role("window")

    analytic = summary["analytic_edges_GeV"]
    m_lo = float(analytic["lower_both"])
    m_hi = float(analytic["upper_isotropic_by_threshold"]["0.51"])
    v_e = float(summary["v_earth"])
    v_max = float(summary["v_esc"]) + v_e
    e_end = float(summary["b8_support_end_keV"])
    e_c = e_end * (v_e / v_max) ** 2
    tol = float(summary["tolerance"])
    win_i = summary["released_windows_GeV"]["released|isotropic"]
    thresholds = [0.2, 0.3, 0.4, 0.51, 0.7, 1.0]
    mass_ticks = [3, 5, 10, 20, 40]

    fig, ax = plt.subplots(1, 3, figsize=(15.8, 5.15))

    # (a) shape residual. The shaded band is where the released observable is
    # exactly degenerate for an isotropic halo; m_lo and m_hi(0.51) are the
    # kinematic edges at the XENONnT threshold.
    rel_i = summary["released_scan"]["released|isotropic"]
    rel_s = summary["released_scan"]["released|streams"]
    ideal_key = "ideal|dE=0.1|thr=0.51|isotropic"
    ideal_rows = summary["ideal_scan"][ideal_key]
    ax[0].axvspan(win_i[0], win_i[1], color=shade, zorder=0, lw=0)
    ax[0].plot([r["mass"] for r in rel_s], [max(r["residual"], 1e-6) for r in rel_s],
               color=any_h, ls="--", lw=1.15, label="any halo", zorder=3)
    ax[0].plot([r["mass"] for r in rel_i], [max(r["residual"], 1e-6) for r in rel_i],
               color=iso, lw=1.45, label="isotropic", zorder=3)
    ax[0].plot([r["mass"] for r in ideal_rows], [max(r["residual"], 1e-6) for r in ideal_rows],
               color=iso, ls=":", lw=1.15, label="isotropic, 0.1 keV bins", zorder=3)
    ax[0].axhline(tol, color=edge, lw=0.55, zorder=1)
    ax[0].axvline(m_lo, color=edge, lw=0.6, ls="-.", zorder=2)
    ax[0].axvline(m_hi, color=edge, lw=0.6, ls="-.", zorder=2)
    ax[0].text(m_lo, 2.2e-3, rf"$m_{{\rm lo}}={m_lo:.2f}$", rotation=90, ha="right", va="bottom",
               color=edge, fontsize=6.5, zorder=4)
    ax[0].text(m_hi, 2.2e-3, rf"$m_{{\rm hi}}(0.51)={m_hi:.2f}$", rotation=90, ha="right", va="bottom",
               color=edge, fontsize=6.5, zorder=4)
    ax[0].text(math.sqrt(win_i[0] * win_i[1]), 0.28, "exact degeneracy", ha="center", va="center",
               color=iso, fontsize=7, zorder=4)
    ax[0].text(36, tol * 2.4, r"tolerance $10^{-3}$", color=edge, fontsize=6.5, ha="right", va="bottom")
    apply_log_axis(ax[0], "x", 3, 40, mass_ticks)
    apply_log_axis(ax[0], "y", 1e-6, 2)
    ax[0].set_xlabel(r"$m_\chi$ (GeV)")
    ax[0].set_ylabel(r"$^8$B residual outside the DM cone")
    ax[0].set_title(rf"Isotropic window {win_i[0]:.2f}–{win_i[1]:.1f} GeV")
    fs.legend_below(ax[0], ncol=2, y=-0.36)
    fs.panel_label(ax[0], "a", dx=-0.02, dy=1.08)

    # (b) ideal 0.1 keV bins. Shade the kinematic window m_lo to m_hi(E_thr),
    # which opens at E_c. Markers are the numerical edges at tolerance 1e-3.
    tt = np.linspace(e_c, 1.16, 240)
    mhi_line = np.array([mass_for(float(t), v_e) for t in tt])
    ax[1].fill_between(tt, np.full_like(tt, m_lo), np.maximum(mhi_line, m_lo),
                       color=shade, lw=0, zorder=0)
    ax[1].plot(tt, mhi_line, color=edge, ls=":", lw=0.9, zorder=2)
    ax[1].axhline(m_lo, color=edge, lw=0.7, zorder=2)
    ax[1].axvline(e_c, color=edge, lw=0.6, ls=":", zorder=2)

    def edges(cname):
        lo, hi = [], []
        for t in thresholds:
            w = summary["ideal_windows_GeV"][f"ideal|dE=0.1|thr={t}|{cname}"]
            lo.append(w[0] if w else np.nan)
            hi.append(w[1] if w else np.nan)
        return np.array(thresholds), np.array(lo), np.array(hi)

    tx, lo_i, hi_i = edges("isotropic")
    _tx, lo_s, hi_s = edges("streams")
    ax[1].plot(tx, lo_i, "o", color=iso, ms=4.2, label="isotropic, lower edge", zorder=3)
    ax[1].plot(tx, hi_i, "s", color=iso, ms=4.2, mfc="none", label="isotropic, upper edge", zorder=3)
    ax[1].plot(tx, lo_s, "o", color=any_h, ms=3.6, label="any halo, lower edge", zorder=3)
    open_to = next((h for h in hi_s if np.isfinite(h)), 40.0)
    ax[1].text(0.27, 12.4, rf"$E_c={e_c:.2f}$ keV", color=edge, fontsize=6.5, ha="left", va="bottom")
    ax[1].text(1.08, m_lo * 0.90, rf"$m_{{\rm lo}}={m_lo:.2f}$", color=edge, fontsize=6.5, ha="right", va="top")
    ax[1].text(0.46, 6.35, r"$m_{\rm hi}(E_{\rm thr})$", color=edge, fontsize=6.5, ha="left", va="center")
    ax[1].text(0.205, 8.6, "0.1 keV bins\nstill closed", color=iso, fontsize=6.5, ha="left", va="center")
    ax[1].text(1.10, 15.45, f"any halo open to {open_to:.0f} GeV", color=any_h, fontsize=6.5, ha="right", va="top")
    apply_log_axis(ax[1], "x", 0.18, 1.18, [0.2, 0.3, 0.5, 1])
    apply_log_axis(ax[1], "y", 4, 16, [5, 8, 10, 15])
    ax[1].set_xlabel(r"xenon threshold (keV)")
    ax[1].set_ylabel(r"$m_\chi$ (GeV)")
    ax[1].set_title(rf"Exact window opens at $E_c={e_c:.2f}$ keV")
    fs.legend_below(ax[1], ncol=2, y=-0.36)
    fs.panel_label(ax[1], "b", dx=-0.02, dy=1.08)

    # (c) exposure to separate 8B from a profiled halo. Reference labels sit
    # at m=8.4 GeV, in the gap where both curves are off the top of the axes.
    exp_i = summary["exposure_released"]["isotropic"]
    exp_s = summary["exposure_released"]["streams"]
    ax[2].axvspan(win_i[0], win_i[1], color=shade, zorder=0, lw=0)
    ax[2].plot([r["mass"] for r in exp_s], [min(r["exposure_ty_3sigma"], 1e6) for r in exp_s],
               color=any_h, ls="--", lw=1.15, label="any halo", zorder=3)
    ax[2].plot([r["mass"] for r in exp_i], [min(r["exposure_ty_3sigma"], 1e6) for r in exp_i],
               color=iso, lw=1.45, label="isotropic", zorder=3)
    ty_sr = float(A.SR_LIVETIME["sr0"] + A.SR_LIVETIME["sr1"])
    for name, ty in (("XENONnT SR0+SR1", ty_sr), ("20 t y", 20.0), ("XLZD 1000 t y", 1000.0)):
        ax[2].axhline(ty, color=edge, lw=0.5, zorder=1)
        ax[2].text(8.4, ty * 1.45, name, fontsize=6.5, color=edge, ha="center", va="bottom", zorder=4)
    ax[2].text(8.4, 4.0e4, "exact degeneracy", ha="center", va="center", color=iso, fontsize=7, zorder=4)
    apply_log_axis(ax[2], "x", 3, 40, mass_ticks)
    apply_log_axis(ax[2], "y", 0.35, 3e6)
    ax[2].set_xlabel(r"$m_\chi$ (GeV)")
    ax[2].set_ylabel(r"exposure for $3\sigma$ separation (t y)")
    ax[2].set_title("Separation diverges inside the window")
    fs.legend_below(ax[2], ncol=2, y=-0.36)
    fs.panel_label(ax[2], "c", dx=-0.02, dy=1.08)

    fig.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=0.24, wspace=0.32)
    warn_overlaps(fig)
    path = OUT / "window.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("wrote", path)


def main():
    if "--plot-only" in sys.argv:
        plot_window(json.loads((OUT / "window.json").read_text()))
        return
    # --- released observable
    energies, ly_axis, qy_axis, g0, g1, _b0, _b1 = A.load_mono_grid()
    yields = A.load_yield()
    ly = A.yield_curve(yields, "ly", 0.0)
    qy = A.yield_curve(yields, "qy", 0.0)
    E_rel = np.linspace(A.E_ROI[0], A.E_ROI[1], 241)
    T_rel = A.released_response(energies, ly_axis, qy_axis, g0, g1, E_rel, ly, qy)

    Es, dR_rel, _ = A.spectrum_from_release(A.XE_DATA / "signal" / "cevns_spectrum.json")

    masses = np.unique(np.concatenate([np.geomspace(3.0, 40.0, 140), [5.3, 6.0, 7.0, 7.66, 8.0, 9.0, 10.0]]))
    classes = {"streams": ("streams", V_STREAM), "isotropic": ("isotropic", V_SHELL)}

    scan = {}
    for cname, (kind, vgrid) in classes.items():
        rows = scan_masses(E_rel, T_rel, np.interp(E_rel, Es, dR_rel), masses, kind, vgrid, with_deviance=True)
        scan[f"released|{cname}"] = rows
        print(f"released|{cname:10s} window: {window(rows)}", flush=True)

    # --- ideal detector: fine grid from the lowest threshold studied
    thresholds = [0.2, 0.3, 0.4, 0.51, 0.7, 1.0]
    widths = [0.5, 0.25, 0.1, 0.05]
    E_fine = np.linspace(0.2, E_MAX, 4801)
    dR_fine = np.interp(E_fine, Es, dR_rel)

    # Convexity of the 8B halo-function target on the support of the spectrum.
    K = (131**2) * A.helm2(E_fine, 131)
    eta_t = dR_fine / K
    support = dR_fine > 1e-3 * dR_fine.max()
    E_end = float(E_fine[support][-1])
    w6 = A.vmin_kms(E_fine, 6.0, 131)
    d2 = np.gradient(np.gradient(eta_t, w6), w6)
    convex_fraction = float(np.mean(d2[support][5:-5] > 0))
    print(f"8B support ends at {E_end:.2f} keV; target convex on {convex_fraction:.4f} of its support")

    ideal = {}
    for dE in widths:
        for E_thr in thresholds:
            T = ideal_response(E_fine, E_thr, dE)
            for cname, (kind, vgrid) in classes.items():
                rows = scan_masses(E_fine, T, dR_fine, masses, kind, vgrid)
                key = f"ideal|dE={dE}|thr={E_thr}|{cname}"
                ideal[key] = dict(window=window(rows), rows=rows)
                print(f"{key:45s} window: {ideal[key]['window']}", flush=True)

    analytic = {
        "upper_isotropic_by_threshold": {f"{t}": mass_for(t, A.V_EARTH) for t in thresholds},
        "lower_both": mass_for(E_end, A.V_MAX),
        "note": "upper: v_min(E_thr)=v_E (concavity of isotropic halo functions below v_E); lower: v_min(E_end)=v_esc+v_E",
    }
    print("analytic:", json.dumps(analytic, indent=1))

    exposure = {
        cname: [dict(mass=r["mass"], exposure_ty_3sigma=(9.0 / r["deviance_per_ty"] if r["deviance_per_ty"] > 1e-12 else 1e300))
                for r in scan[f"released|{cname}"]]
        for cname in classes
    }

    summary = dict(
        halo_classes=dict(
            streams="nonnegative lab-frame speed distributions on [0, v_esc + v_E]; generators are lab streams",
            isotropic="isotropic Galactic-frame speed distributions on [0, v_esc]; generators are boosted shells",
        ),
        v_esc=A.V_ESC, v_earth=A.V_EARTH, tolerance=TOL,
        b8_support_end_keV=E_end, b8_target_convex_fraction=convex_fraction,
        analytic_edges_GeV=analytic,
        released_windows_GeV={k: window(v) for k, v in scan.items()},
        ideal_windows_GeV={k: v["window"] for k, v in ideal.items()},
        exposure_released=exposure,
        released_scan=scan,
        ideal_scan={k: v["rows"] for k, v in ideal.items() if "dE=0.1" in k},
    )
    (OUT / "window.json").write_text(json.dumps(summary, indent=2))
    plot_window(summary)


if __name__ == "__main__":
    main()

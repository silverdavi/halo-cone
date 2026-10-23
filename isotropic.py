#!/usr/bin/env python3
"""Exact-degeneracy window of 8B CEvNS versus SI dark matter in xenon, by halo class.

Two convex cones of dark-matter spectra, both with free cross section:

  streams    any nonnegative lab-frame speed distribution on [0, v_esc + v_E].
             Generators: lab-frame mono-speed streams, eta_v(w) = 1[w < v] / v.
  isotropic  any isotropic Galactic-frame speed distribution on [0, v_esc].
             Generators: Galactic shells boosted by v_E,
             eta_v(w) = (v + v_E - max(w, |v - v_E|))^+ / (2 v v_E).

The 8B template is exactly degenerate with dark matter of mass m iff it lies
in the cone at that mass. Outside the cone the residual shape fixes the xenon
exposure needed to separate the two hypotheses with the halo profiled.

Writes results/isotropic.json and results/isotropic.png.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, nnls

import analysis as A

OUT = A.OUT


def eta_shell(w, v, v_e):
    """Mean inverse lab speed of an isotropic Galactic shell at speed v, w = v_min."""
    w = np.asarray(w, float)
    if v < 1e-9:
        return np.where(w < v_e, 1.0 / v_e, 0.0)
    lo = abs(v - v_e)
    hi = v + v_e
    return np.clip(hi - np.maximum(w, lo), 0.0, None) / (2.0 * v * v_e)


def eta_stream(w, v):
    w = np.asarray(w, float)
    return np.where(w < v, 1.0 / v, 0.0)


def generator_matrix(E, T, m_chi, isotopes, v_grid, kind, v_e):
    """Columns: observable spectra of one generator each (free overall scale)."""
    dE = np.gradient(E)
    v_grid = np.asarray(v_grid, float)
    G = np.zeros((T.shape[1], len(v_grid)))
    for Aiso, _Z, _n, w in A.mass_fractions(isotopes):
        K = w * (Aiso**2) * A.helm2(E, Aiso) * dE
        vmin = A.vmin_kms(E, m_chi, Aiso)
        if kind == "isotropic":
            lo = np.abs(v_grid - v_e)[None, :]
            hi = (v_grid + v_e)[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                eta = np.clip(hi - np.maximum(vmin[:, None], lo), 0.0, None) / (2.0 * v_grid[None, :] * v_e)
            zero = v_grid < 1e-9
            if zero.any():
                eta[:, zero] = np.where(vmin[:, None] < v_e, 1.0 / v_e, 0.0)
        else:
            with np.errstate(divide="ignore"):
                eta = np.where(vmin[:, None] < v_grid[None, :], 1.0 / v_grid[None, :], 0.0)
        G += T.T @ (K[:, None] * eta)
    return G


def poisson_fit(G, mu, x0=None):
    """Nonnegative fit minimizing the Poisson deviance of G x against mu."""
    col = np.linalg.norm(G, axis=0)
    if col.max() <= 0:
        return np.inf, np.zeros(G.shape[1])
    keep = col > 1e-14 * col.max()
    Gk = G[:, keep] / col[keep]
    if x0 is None:
        x0, _ = nnls(Gk, mu)
    x0 = np.maximum(x0, 1e-12)

    def dev(x):
        mh = np.maximum(Gk @ x, 1e-300)
        pos = mu > 0
        return 2.0 * (np.sum(mu[pos] * np.log(mu[pos] / mh[pos]) - (mu[pos] - mh[pos])) + np.sum(mh[~pos]))

    def grad(x):
        mh = np.maximum(Gk @ x, 1e-300)
        return Gk.T @ (1.0 - mu / mh) * 2.0

    res = minimize(dev, x0, jac=grad, method="L-BFGS-B", bounds=[(0, None)] * Gk.shape[1],
                   options=dict(maxiter=2000, ftol=1e-12, gtol=1e-9))
    x = np.zeros(G.shape[1])
    x[keep] = res.x / col[keep]
    return float(res.fun), x


def l2_residual(G, phi):
    col = np.linalg.norm(G, axis=0)
    if col.max() <= 0:
        return 1.0, np.zeros_like(phi)
    keep = col > 1e-14 * col.max()
    x, _ = nnls(G[:, keep] / col[keep], phi)
    recon = (G[:, keep] / col[keep]) @ x
    return float(np.linalg.norm(phi - recon) / np.linalg.norm(phi)), recon


def plot_isotropic(summary):
    """Residual and separation exposure by halo class. `summary` is isotropic.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sys.path.insert(0, str(Path(__file__).resolve().parent / "config"))
    import figstyle as fs
    from window import apply_log_axis, warn_overlaps

    fs.set_style()
    iso, any_h = fs.role("isotropic"), fs.role("anyhalo")
    edge, shade = fs.role("edge"), fs.role("window")
    scan = summary["scan"]
    win = summary["windows"]["isotropic"]["practical"]
    thr = 0.039  # yield-morph shape radius used to define the practical window
    analytic = json.loads((OUT / "window.json").read_text())["analytic_edges_GeV"]
    m_lo = float(analytic["lower_both"])
    m_hi = float(analytic["upper_isotropic_by_threshold"]["0.51"])

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.85))
    for name in ("isotropic_vE220", "isotropic_vE250", "isotropic_vesc500", "isotropic_vesc600"):
        rows = scan[name]
        ms = [r["mass"] for r in rows]
        ax[0].plot(ms, [r["residual"] for r in rows], color=iso, alpha=0.35, lw=0.8, zorder=2)
        ax[1].plot(ms, [r["exposure_ty_3sigma"] for r in rows], color=iso, alpha=0.35, lw=0.8, zorder=2)
    for name, color, ls, lw in (("streams", any_h, "--", 1.15), ("isotropic", iso, "-", 1.45)):
        rows = scan[name]
        ms = [r["mass"] for r in rows]
        ax[0].plot(ms, [r["residual"] for r in rows], color=color, ls=ls, lw=lw, zorder=3)
        ax[1].plot(ms, [r["exposure_ty_3sigma"] for r in rows], color=color, ls=ls, lw=lw, zorder=3)

    handles = [
        Line2D([], [], color=any_h, ls="--", label="any halo"),
        Line2D([], [], color=iso, label="isotropic"),
        Line2D([], [], color=iso, alpha=0.45, label=r"isotropic, $v_E$ and $v_{\rm esc}$ varied"),
    ]
    for a in ax:
        a.axvspan(win[0], win[1], color=shade, zorder=0, lw=0)
        a.axvline(m_lo, color=edge, lw=0.6, ls="-.", zorder=1)
        a.axvline(m_hi, color=edge, lw=0.6, ls="-.", zorder=1)
        apply_log_axis(a, "x", 2, 60, [2, 5, 10, 20, 50])

    ax[0].axhline(thr, color=edge, lw=0.55, zorder=1)
    ax[0].text(m_lo, 0.55, rf"$m_{{\rm lo}}={m_lo:.2f}$", rotation=90, ha="right", va="top",
               color=edge, fontsize=6.5)
    ax[0].text(m_hi, 0.55, rf"$m_{{\rm hi}}={m_hi:.2f}$", rotation=90, ha="right", va="top",
               color=edge, fontsize=6.5)
    ax[0].text(math.sqrt(win[0] * win[1]), 0.22, "practical window", ha="center", va="center",
               color=iso, fontsize=7)
    ax[0].text(52, thr * 1.25, "yield-morph radius", color=edge, fontsize=6.5, ha="right", va="bottom")
    apply_log_axis(ax[0], "y", 0.008, 1.6)
    ax[0].set_xlabel(r"$m_\chi$ (GeV)")
    ax[0].set_ylabel(r"$^8$B residual outside the DM cone")
    ax[0].set_title(rf"Practical isotropic window {win[0]:.2f}–{win[1]:.1f} GeV")
    fs.legend_below(ax[0], ncol=2, y=-0.36, handles=handles)
    fs.panel_label(ax[0], "a", dx=-0.02, dy=1.08)

    ty_sr = float(A.SR_LIVETIME["sr0"] + A.SR_LIVETIME["sr1"])
    ax[1].axhline(ty_sr, color=edge, lw=0.55, zorder=1)
    ax[1].text(math.sqrt(win[0] * win[1]), ty_sr * 1.7, "XENONnT SR0+SR1", color=edge, fontsize=6.5,
               ha="center", va="bottom")
    ax[1].text(math.sqrt(win[0] * win[1]), 8e3, "practical window", ha="center", va="center", color=iso, fontsize=7)
    ax[1].text(m_lo, 40, rf"$m_{{\rm lo}}$", rotation=90, ha="right", va="bottom", color=edge, fontsize=6.5)
    ax[1].text(m_hi, 40, rf"$m_{{\rm hi}}$", rotation=90, ha="right", va="bottom", color=edge, fontsize=6.5)
    apply_log_axis(ax[1], "y", 0.4, 2e4)
    ax[1].set_xlabel(r"$m_\chi$ (GeV)")
    ax[1].set_ylabel(r"exposure for $3\sigma$ separation (t y)")
    ax[1].set_title("Any halo stays degenerate; isotropic window closes")
    fs.legend_below(ax[1], ncol=2, y=-0.36, handles=handles)
    fs.panel_label(ax[1], "b", dx=-0.02, dy=1.08)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.26, wspace=0.30)
    warn_overlaps(fig)
    path = OUT / "isotropic.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("wrote", path)


def main():
    if "--plot-only" in sys.argv:
        plot_isotropic(json.loads((OUT / "isotropic.json").read_text()))
        return
    energies, ly_axis, qy_axis, g0, g1, _b0, _b1 = A.load_mono_grid()
    yields = A.load_yield()
    ly = A.yield_curve(yields, "ly", 0.0)
    qy = A.yield_curve(yields, "qy", 0.0)
    E = np.linspace(A.E_ROI[0], A.E_ROI[1], 241)
    T0 = A.response_on_grid(energies, ly_axis, qy_axis, g0, E, ly, qy)
    T1 = A.response_on_grid(energies, ly_axis, qy_axis, g1, E, ly, qy)
    T = np.hstack([T0, T1])

    b8_ref = np.concatenate(
        [A.load_template(A.XE_DATA / "b8" / f"template_XENONnT_{sr}_b8_cevns_tly_0.0_tqy_0.0.h5") for sr in ("sr0", "sr1")]
    )
    Es, dR, _rate = A.spectrum_from_release(A.XE_DATA / "signal" / "cevns_spectrum.json")
    folded = A.fold_spectrum(Es, dR, E, T)
    exposure_ty = float(np.dot(folded, b8_ref) / np.dot(folded, folded))
    phi_xe = b8_ref / exposure_ty  # events per tonne-year, released observable bins
    print(f"8B per tonne-year in ROI: {phi_xe.sum():.2f}; release exposure {exposure_ty:.3f} ty")

    # Ideal energy detector, 16 bins on the ROI, no S1/S2 migration.
    edges_ideal = np.linspace(A.E_ROI[0], A.E_ROI[1], 17)
    T_ideal = np.zeros((len(E), 16))
    which = np.clip(np.digitize(E, edges_ideal) - 1, 0, 15)
    T_ideal[np.arange(len(E)), which] = 1.0
    E_check = np.linspace(0.45, 10.0, 400)
    dR_8B, _ = A.compound_spectrum(E_check, A.XE_ISOTOPES, "8B")
    phi_ideal = A.fold_spectrum(E, np.interp(E, E_check, dR_8B), E, T_ideal)

    masses = np.unique(np.concatenate([np.geomspace(2.0, 60.0, 73), [5.3, 6.0, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0]]))
    classes = {
        "streams": dict(v_grid=np.geomspace(20.0, A.V_MAX, 120), v_e=A.V_EARTH),
        "isotropic": dict(v_grid=np.concatenate([[0.0], np.geomspace(5.0, A.V_ESC, 120)]), v_e=A.V_EARTH),
    }
    # Astrophysical variations for the isotropic class.
    variants = {
        "isotropic_vE220": dict(v_grid=np.concatenate([[0.0], np.geomspace(5.0, A.V_ESC, 120)]), v_e=220.0),
        "isotropic_vE250": dict(v_grid=np.concatenate([[0.0], np.geomspace(5.0, A.V_ESC, 120)]), v_e=250.0),
        "isotropic_vesc500": dict(v_grid=np.concatenate([[0.0], np.geomspace(5.0, 500.0, 120)]), v_e=A.V_EARTH),
        "isotropic_vesc600": dict(v_grid=np.concatenate([[0.0], np.geomspace(5.0, 600.0, 120)]), v_e=A.V_EARTH),
    }
    all_classes = {**classes, **variants}

    rows = {k: [] for k in all_classes}
    for name, spec in all_classes.items():
        kind = "isotropic" if name.startswith("isotropic") else "streams"
        print(f"== {name}")
        for m in masses:
            G = generator_matrix(E, T, float(m), A.XE_ISOTOPES, spec["v_grid"], kind, spec["v_e"])
            resid, _ = l2_residual(G, phi_xe)
            dev1, _ = poisson_fit(G, phi_xe)  # deviance per tonne-year (linear in exposure)
            G_i = generator_matrix(E, T_ideal, float(m), A.XE_ISOTOPES, spec["v_grid"], kind, spec["v_e"])
            resid_i, _ = l2_residual(G_i, phi_ideal)
            # Exposure for a profiled deviance of 9 (3 sigma, one-sided) and 25.
            eps9 = 9.0 / dev1 if dev1 > 1e-12 else 1e300
            if not np.isfinite(dev1):
                dev1, eps9 = 1e300, 0.0
            rows[name].append(dict(mass=float(m), residual=resid, residual_ideal=resid_i,
                                   deviance_per_ty=dev1, exposure_ty_3sigma=eps9))
            if name in classes:
                print(f"  m={m:6.2f}  resid={resid:.4f}  ideal={resid_i:.4f}  dev/ty={dev1:.3e}  ty(3s)={eps9:.3g}")

    # Window edges: first/last mass with released-observable residual below the yield-morph radius (0.039)
    # and below a strict numerical threshold.
    def window(rs, thr):
        ms = [r["mass"] for r in rs if r["residual"] < thr]
        return (min(ms), max(ms)) if ms else None

    thr_practical = 0.039
    thr_strict = 0.005
    summary = dict(
        halo_classes=dict(
            streams="nonnegative lab-frame speed distributions on [0, v_esc+v_E]",
            isotropic="isotropic Galactic-frame speed distributions on [0, v_esc], boosted by v_E",
        ),
        v_esc=A.V_ESC, v_earth=A.V_EARTH,
        b8_per_tonne_year=float(phi_xe.sum()),
        windows={k: dict(practical=window(v, thr_practical), strict=window(v, thr_strict)) for k, v in rows.items()},
        scan=rows,
    )
    (OUT / "isotropic.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["windows"], indent=1))
    plot_isotropic(summary)


if __name__ == "__main__":
    main()

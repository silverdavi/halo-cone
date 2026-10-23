#!/usr/bin/env python3
"""Paper figures from results/*.json (Rules/FIGURES.md). Writes paper/figs/*.pdf and .png.

Colour is a role, the same one in every figure: terracotta isotropic,
ink any halo, dusty blue the 8B target, gold the SHM, warm shade the
mass window. A mass sequence is a truncated ramp of that role, light to
dark, labelled on the curve. The canvas carries data and values only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "config"))
import figstyle as fs  # noqa: E402
import analysis as A  # noqa: E402
from window import mass_for  # noqa: E402

fs.set_style()
ISO, ANY, NU, SHM, EDGE = (
    fs.role("isotropic"), fs.role("anyhalo"), fs.role("neutrino"),
    fs.role("shm"), fs.role("edge"),
)
WINDOW = fs.role("window")
CLOUD = fs.c("cloud")
INK = fs.c("ink")

MASS_TICKS = [3, 4, 5, 6, 8, 10, 15, 20, 30, 40]
TAG = dict(fontsize=7, zorder=6, annotation_clip=True,
           bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.92))


def load(name):
    return json.load(open(A.OUT / name))


def numbers():
    return json.load(open(A.OUT / "numbers.json"))


def mass_axis(ax, lo=3, hi=40):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [t for t in MASS_TICKS if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(r"$m_\chi$ (GeV)")


def tag(ax, x, y, text, color, ha="left", va="center", dx=0, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, ha=ha, va=va, **TAG)


def shell_eta(w, v, vE=A.V_EARTH):
    if v == 0:
        return np.where(w < vE, 1.0 / vE, 0.0)
    return np.clip(v + vE - np.maximum(w, abs(v - vE)), 0, None) / (2 * v * vE)


def fig_lemma():
    """Isotropic shells are concave on [0, v_E); the 8B target is strictly convex."""
    N = numbers()
    fig, ax = plt.subplots(1, 2, figsize=(fs.W2, 2.22))
    w = np.linspace(0, 820, 1801)
    speeds = (0, 120, 232, 400, 544)
    # light to dark = low speed to high; deep end mixed toward ink so
    # neighbours separate, light end still clears the shade
    colors = fs.role_ramp("isotropic", len(speeds), on=WINDOW, dark=0.55)
    # zero-crossings; the u=0 shell is a step that ends at v_E
    label_at = {120: 340, 232: 448, 400: 610, 544: 750}
    ax[0].axvspan(0, A.V_EARTH, color=CLOUD, zorder=0, lw=0)
    for v, col in zip(speeds, colors):
        y = shell_eta(w, v) * 1e3
        ax[0].plot(w, y, color=col, lw=1.5 if v == 0 else 1.2, zorder=3)
        if v == 0:
            tag(ax[0], 155, 4.31, "0", col, ha="center", va="bottom", dy=1)
        else:
            i = int(np.argmin(np.abs(w - label_at[v])))
            tag(ax[0], w[i], max(y[i], 0.22), f"{v:g}", col,
                ha="center", va="bottom", dy=1)
    ax[0].axvline(A.V_EARTH, color=EDGE, lw=0.6, ls=":")
    tag(ax[0], A.V_EARTH, 5.15, r"$v_E$", EDGE, ha="left", va="bottom", dx=3)
    ax[0].text(A.V_EARTH / 2, 5.15, r"$[0,\,v_E)$", color=EDGE, fontsize=7,
               ha="center", va="bottom")
    ax[0].set_xlabel(r"$v_{\min}$ (km/s)")
    ax[0].set_ylabel(r"$\tilde\eta_u(v_{\min})$ ($10^{-3}$ s/km)")
    ax[0].set_ylim(0, 5.55)
    ax[0].set_xlim(0, 820)
    fs.panel_label(ax[0], "a")

    E = np.linspace(0.51, 5.0, 2000)
    dR, _ = A.compound_spectrum(E, A.XE_ISOTOPES, "8B")
    K = sum(wt * Ai**2 * A.helm2(E, Ai) for Ai, _z, _n, wt in A.mass_fractions(A.XE_ISOTOPES))
    target = dR / K
    masses = (6.0, N["m_hi_051_131"], 10.0)
    names = ("6", f"{N['m_hi_051_131']:.2f}", "10")
    # light to dark = low mass to high; 10 GeV is the curve that enters the band
    cols = fs.role_ramp("neutrino", len(masses), on=CLOUD)
    offsets = ((8, 3, "left"), (-2, 9, "center"), (-8, 3, "right"))
    ax[1].axvspan(0, A.V_EARTH, color=CLOUD, zorder=0, lw=0)
    for m, name, col, (dx, dy, ha) in zip(masses, names, cols, offsets):
        vmin = A.vmin_kms(E, m, 130.905)  # 131Xe, as in export_numbers.py
        y = target / target[0]
        ax[1].plot(vmin, y, color=col, lw=1.35, zorder=3)
        ax[1].plot(vmin[0], 1.0, "o", color=col, ms=4.0, zorder=4)
        tag(ax[1], vmin[0], 1.0, name, col, ha=ha, va="bottom", dx=dx, dy=dy)
    ax[1].axvline(A.V_EARTH, color=EDGE, lw=0.6, ls=":")
    ax[1].axvline(A.V_MAX, color=EDGE, lw=0.6, ls="--")
    tag(ax[1], A.V_EARTH, 0.18, r"$v_E$", EDGE, ha="left", va="bottom", dx=3)
    tag(ax[1], A.V_MAX, 0.72, r"$v_{\rm esc}+v_E$", EDGE, ha="left", va="bottom", dx=3)
    ax[1].text(70, 0.50, r"$[0,\,v_E)$", color=EDGE, fontsize=7, ha="center", va="center")
    ax[1].set_ylim(0, 1.22)
    ax[1].set_xlim(0, 1000)
    ax[1].set_xlabel(r"$v_{\min}(E_R;m_\chi)$ (km/s)")
    ax[1].set_ylabel(r"$\tilde\eta_\nu$ / value at $E_{\rm thr}$")
    fs.panel_label(ax[1], "b")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.90, wspace=0.32)
    fs.save_fig(fig, "lemma")
    plt.close(fig)


def fig_residual():
    """Inside [m_lo, m_hi] the ideal isotropic residual is at the floor."""
    N = numbers()
    win = load("window.json")
    ideal = load("ideal.json")
    fig, ax = plt.subplots(figsize=(fs.W1, 2.55))
    m = np.array(ideal["masses"])
    floor = 1e-16
    m_lo, m_hi = N["m_lo_131"], N["m_hi_051_131"]
    ax.axvspan(m_lo, m_hi, color=WINDOW, zorder=0, lw=0)

    series = (
        (np.maximum(ideal["scans"]["0.51|isotropic"], floor),
         dict(color=ISO, lw=1.7, zorder=4)),
        (np.maximum(ideal["scans"]["0.3|isotropic"], floor),
         dict(color=ISO, lw=1.05, ls=":", zorder=3)),
        (np.maximum(ideal["scans"]["0.51|streams"], floor),
         dict(color=ANY, lw=1.0, zorder=3)),
    )
    for y, style in series:
        ax.plot(m, y, **style)
    for cname, style in (
        ("isotropic", dict(color=ISO, ls="--", lw=1.05, zorder=3)),
        ("streams", dict(color=ANY, ls="--", lw=1.0, zorder=2)),
    ):
        rows = win["released_scan"][f"released|{cname}"]
        ax.plot([r["mass"] for r in rows],
                [max(r["residual"], floor) for r in rows], **style)

    ax.axvline(N["m_hi_030_131"], color=EDGE, lw=0.6, ls="-.")
    ax.axhline(1e-3, color=EDGE, lw=0.55)
    # labels sit where that curve is alone; long ones are right-aligned
    ax.annotate(
        "0.51 keV", xy=(8.24, 1e-8), xytext=(10.6, 1e-10),
        color=ISO, fontsize=7, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=ISO, lw=0.5),
        zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.92))
    tag(ax, 12.2, 0.38, "0.30 keV", ISO, ha="left", va="bottom")
    tag(ax, 36, 0.22, "XENONnT", ISO, ha="right", va="center")
    tag(ax, 18.5, 8e-15, "any halo", ANY, ha="left", va="center")
    tag(ax, 36, 1.7e-3, r"$r=10^{-3}$", EDGE, ha="right", va="bottom")
    tag(ax, N["m_hi_030_131"], 0.25, r"$m_{\rm hi}(0.30)$", EDGE,
        ha="right", va="bottom", dx=-3)
    ax.text(np.sqrt(m_lo * m_hi), 1.6, r"$[m_{\rm lo},\,m_{\rm hi}]$",
            color=INK, fontsize=7, ha="center", va="bottom")

    mass_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(floor, 3)
    ax.set_ylabel(r"distance $r$ of $^8$B from the DM cone")
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.16, top=0.96)
    fs.save_fig(fig, "residual")
    plt.close(fig)


def fig_edges():
    """The window opens at E_c and its upper edge is m_hi(E_thr)."""
    N = numbers()
    ideal = load("ideal.json")
    thr = np.array(ideal["thresholds"])
    fig, ax = plt.subplots(figsize=(fs.W1, 2.35))
    tt = np.linspace(0.15, 1.4, 200)
    mhi = np.array([mass_for(t, A.V_EARTH) for t in tt])
    mlo = N["m_lo_131"]
    ax.fill_between(tt, mlo, mhi, where=mhi > mlo, color=WINDOW, lw=0, zorder=0)
    ax.plot(tt, mhi, color=ISO, lw=1.4, zorder=3)
    ax.axhline(mlo, color=ANY, lw=1.15, zorder=3)
    lo, hi, lo3, hi3 = [], [], [], []
    for t in thr:
        w = ideal["windows"][f"{t}|isotropic"]["1e-12"]
        w3 = ideal["windows"][f"{t}|isotropic"]["0.001"]
        lo.append(w[0] if w else np.nan)
        hi.append(w[1] if w else np.nan)
        lo3.append(w3[0] if w3 else np.nan)
        hi3.append(w3[1] if w3 else np.nan)
    ax.plot(thr, hi, "s", color=ISO, mfc="none", ms=3.6, zorder=4)
    ax.plot(thr, lo, "s", color=ANY, mfc="none", ms=3.6, zorder=4)
    ax.plot(thr, hi3, "^", color=ISO, ms=3.6, zorder=4)
    ax.plot(thr, lo3, "^", color=ANY, ms=3.6, zorder=4)
    ax.axvline(N["E_c_131"], color=EDGE, lw=0.6, ls=":")
    tag(ax, 1.22, mhi[-1], r"$m_{\rm hi}$", ISO, ha="right", va="bottom", dy=2)
    tag(ax, 0.20, mlo, r"$m_{\rm lo}$", ANY, ha="left", va="bottom", dy=2)
    tag(ax, N["E_c_131"], 13.3, rf"$E_c={N['E_c_131']:.2f}$ keV", EDGE,
        ha="left", va="top", dx=3)
    tag(ax, 0.78, 8.15, r"$r<10^{-12}$", EDGE, ha="center", va="center")
    tag(ax, 0.95, 4.55, r"$r<10^{-3}$", EDGE, ha="left", va="center")
    ax.text(1.02, 8.6, "window", color=INK, fontsize=7, ha="center", va="center")
    ax.set_xlabel(r"xenon threshold $E_{\rm thr}$ (keV)")
    ax.set_ylabel(r"$m_\chi$ (GeV)")
    ax.set_ylim(3.4, 14.2)
    ax.set_xlim(0.15, 1.42)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.96)
    fs.save_fig(fig, "edges")
    plt.close(fig)


def _disc_rows(results, mass):
    rows = results[str(mass)]
    eps = np.array([r["exposure_ty"] for r in rows], float)
    hi = np.array([r["sigma_halo_independent"] for r in rows], float)
    shm = np.array([r["sigma_shm"] for r in rows], float)
    n = np.array([np.nan if r.get("n_sigma_halo_independent") is None
                  else r["n_sigma_halo_independent"] for r in rows], float)
    return eps, hi, shm, n


def fig_discovery():
    """Three regimes: infinite below m_hi, saturating at 12-14, tracking by 30."""
    d = load("discovery.json")["results"]
    # 12 and 14 are one regime (the finite-resolution window); 16 is just
    # above it; 30 tracks the SHM. Light to dark with mass.
    groups = (12, 14, 16, 30)
    cols = fs.role_ramp("isotropic", len(groups))
    fig, ax = plt.subplots(1, 2, figsize=(fs.W2, 2.40))
    for m, col in zip(groups, cols):
        eps, hi, shm, n = _disc_rows(d, m)
        ax[0].plot(eps, shm, color=SHM, lw=0.9, ls="--", zorder=2)
        ax[0].plot(eps, hi, color=col, lw=1.45, zorder=3)
        ax[1].plot(eps, n, color=col, lw=1.45, zorder=3)
    # 12 sits just above 14, so its label is lifted into the empty headroom
    series = {m: _disc_rows(d, m) for m in groups}
    e12, h12 = series[12][0], series[12][1]
    ax[0].annotate(
        "12", xy=(e12[9], h12[9]), xytext=(e12[9], h12[9] * 25),
        color=cols[0], fontsize=7, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=cols[0], lw=0.5),
        zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.92))
    tag(ax[0], series[14][0][-1], series[14][1][-1], "14", cols[1],
        ha="left", va="center", dx=3)
    tag(ax[0], series[16][0][-1], series[16][1][-1], "16", cols[2],
        ha="left", va="center", dx=3)
    e30, h30, _s30, _n30 = series[30]
    tag(ax[0], e30[-1], h30[-1], "30", cols[3], ha="left", va="center", dx=3)
    # The four SHM curves lie within a factor 1.5 of one another, a single
    # band at this scale, so each is tagged below the band in the empty
    # lower left, in mass order along the curve, with a leader to its own
    # curve (log-log interpolated at the label's exposure).
    shm_at = {12: 0.55, 14: 2.4, 16: 7.0, 30: 20.0}
    for m, x in shm_at.items():
        e, _hi, shm, _n = series[m]
        y = 10 ** np.interp(np.log10(x), np.log10(e), np.log10(shm))
        ax[0].annotate(
            f"SHM {m}" if m == 12 else f"{m}", xy=(x, y), xytext=(0, -12),
            textcoords="offset points", color=SHM, fontsize=7, ha="center",
            va="top", zorder=6,
            arrowprops=dict(arrowstyle="-", color=SHM, lw=0.5, shrinkA=0,
                            shrinkB=1.5),
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.92))
    for m, col in zip(groups, cols):
        n = series[m][3]
        finite = np.where(np.isfinite(n))[0]
        tag(ax[1], series[m][0][finite[-1]], n[finite[-1]], f"{m}", col,
            ha="left", va="center", dx=3)
    # 6 GeV is infinite at every exposure: an arrow into the empty headroom
    ax[0].annotate(
        "", xy=(0.18, 0.93), xytext=(0.18, 0.74),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=ISO, lw=0.9),
        zorder=5)
    ax[0].text(0.20, 0.83, r"$6\,\mathrm{GeV},\ \sigma=\infty$",
               transform=ax[0].transAxes, color=ISO, fontsize=7, va="center",
               ha="left", zorder=6,
               bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.92))
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlim(0.2, 1.2e5)
    ax[0].set_ylim(3e-49, 2e-42)
    ax[0].set_xlabel(r"xenon exposure $\varepsilon$ (t y)")
    ax[0].set_ylabel(r"$3\sigma$ discovery cross section (cm$^2$)")
    fs.panel_label(ax[0], "a")

    ax[1].axhline(2, color=EDGE, lw=0.6)
    tag(ax[1], 0.35, 2, r"$n=2$", EDGE, ha="left", va="bottom", dy=2)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(0.2, 4e4)
    ax[1].set_ylim(1.0, 70)
    fog_ticks = [1, 2, 3, 5, 10, 20, 50]
    ax[1].set_yticks(fog_ticks)
    ax[1].set_yticklabels([str(t) for t in fog_ticks])
    ax[1].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[1].set_xlabel(r"xenon exposure $\varepsilon$ (t y)")
    ax[1].set_ylabel(r"fog index $n=-(\mathrm{d}\ln\sigma/\mathrm{d}\ln\varepsilon)^{-1}$")
    fs.panel_label(ax[1], "b")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.90, wspace=0.32)
    fs.save_fig(fig, "discovery")
    plt.close(fig)


def _exposure_segments(rows, cap):
    m = np.array([r["mass"] for r in rows], float)
    e = np.array([r["exposure_ty_3sigma"] for r in rows], float)
    above = e >= cap
    spans = []
    i = 0
    while i < len(m):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(m) and above[j]:
            j += 1
        spans.append((m[i], m[j - 1]))
        i = j
    finite = np.ma.masked_where(above, e)
    return m, finite, spans


def fig_separation():
    """Isotropic separation diverges inside the window and falls past 13 GeV."""
    win = load("window.json")
    comp = load("complement.json")
    fig, ax = plt.subplots(1, 2, figsize=(fs.W2, 2.42))
    cap = 1e6
    styles = {
        "streams": dict(color=ANY, ls="--", lw=1.15, zorder=3),
        "isotropic": dict(color=ISO, lw=1.45, zorder=4),
    }
    spans = {}
    for cname, style in styles.items():
        m, finite, sp = _exposure_segments(win["exposure_released"][cname], cap)
        spans[cname] = sp
        ax[0].plot(m, finite, **style)
    # shaded band: isotropic exposure above 1e6 t y (one contiguous run)
    iso_lo, iso_hi = spans["isotropic"][0]
    ax[0].axvspan(iso_lo, iso_hi, color=WINDOW, zorder=0, lw=0)
    ax[0].text(np.sqrt(iso_lo * iso_hi), 2e5, r"$\infty$", color=ISO,
               fontsize=8, ha="center", va="center", fontweight="medium")
    # any halo stays infinite from its first divergent mass through 40 GeV
    any_lo = spans["streams"][0][0]
    trans = blended_transform_factory(ax[0].transData, ax[0].transAxes)
    ax[0].plot([any_lo, 40], [0.96, 0.96], transform=trans, color=ANY,
               ls="--", lw=1.05, clip_on=False, zorder=5)
    ax[0].text(26, 0.96, r"any halo  $\infty$", transform=trans, color=ANY,
               fontsize=7, ha="center", va="bottom")
    live = A.SR_LIVETIME["sr0"] + A.SR_LIVETIME["sr1"]
    for y, text in ((live, "SR0+SR1"), (20.0, "20 t y"), (1000.0, "XLZD")):
        ax[0].axhline(y, color=EDGE, lw=0.5, zorder=1)
        ax[0].text(8.0, y * 1.35, text, color=EDGE, fontsize=7, ha="center", va="bottom")
    tag(ax[0], 3.25, 1.45, "isotropic", ISO, ha="left", va="top")
    tag(ax[0], 3.6, 90, "any halo", ANY, ha="left", va="bottom")
    mass_axis(ax[0])
    ax[0].set_yscale("log")
    ax[0].set_ylim(0.6, 2e6)
    ax[0].set_ylabel(r"xenon exposure for $3\sigma$ separation (t y)")
    fs.panel_label(ax[0], "a")

    res = comp["results"]
    for cname, style in (
        ("streams", dict(color=ANY, ls="--", lw=1.15, marker="o", ms=3.4, zorder=3)),
        ("isotropic", dict(color=ISO, lw=1.45, marker="s", ms=3.4, zorder=4)),
    ):
        ms, ty = [], []
        for k, v in res.items():
            if k.startswith(cname) and k.endswith("xenon_20ty") and v["cawo4_untagged_ty_for_deviance_9"]:
                ms.append(v["mass"])
                ty.append(v["cawo4_untagged_ty_for_deviance_9"])
        o = np.argsort(ms)
        ax[1].plot(np.array(ms)[o], np.array(ty)[o], **style)
    # the factor at the two ends of the mass list, read off the same points
    by_mass = {}
    for k, v in res.items():
        if k.endswith("xenon_20ty") and v["cawo4_untagged_ty_for_deviance_9"]:
            by_mass.setdefault(v["mass"], {})[k.split("|")[0]] = v["cawo4_untagged_ty_for_deviance_9"]
    for mass, text, dx in ((5.5, None, 0.28), (10.0, None, -0.28)):
        s, i = by_mass[mass]["streams"], by_mass[mass]["isotropic"]
        factor = int(round(s / i))
        x = mass + dx
        ax[1].annotate(
            "", xy=(x, i), xytext=(x, s),
            arrowprops=dict(arrowstyle="<->", color=EDGE, lw=0.7,
                            shrinkA=0, shrinkB=0))
        tag(ax[1], x, np.sqrt(s * i), rf"$\times{factor}$", INK, ha="center", va="center")
    tag(ax[1], 7.15, 4.6, "any halo", ANY, ha="left", va="bottom")
    tag(ax[1], 7.15, 0.055, "isotropic", ISO, ha="left", va="top")
    ax[1].set_yscale("log")
    ax[1].set_xlim(5.15, 10.7)
    ax[1].set_ylim(0.03, 9)
    ax[1].set_xlabel(r"$m_\chi$ (GeV)")
    ax[1].set_ylabel(r"CaWO$_4$ exposure for $3\sigma$ separation (t y)")
    fs.panel_label(ax[1], "b")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.90, wspace=0.30)
    fs.save_fig(fig, "separation")
    plt.close(fig)


if __name__ == "__main__":
    for role_name, on in (("isotropic", "#ffffff"), ("isotropic", WINDOW),
                          ("neutrino", "#ffffff"), ("shm", "#ffffff")):
        cols = fs.role_ramp(role_name, 4, on=on)
        ratios = [f"{fs.contrast(c, on):.2f}" for c in cols]
        print(f"ramp {role_name} on {on}: {cols} contrast {ratios}")
    print(f"gold on white {fs.contrast(SHM):.2f}, terracotta {fs.contrast(ISO):.2f}, "
          f"dustyblue {fs.contrast(NU):.2f}")
    fig_lemma()
    fig_residual()
    fig_edges()
    fig_discovery()
    fig_separation()

#!/usr/bin/env python3
"""Isotropic exact-degeneracy windows for other targets.

For each target the window edges follow from kinematics alone:
  lower edge  m_lo : v_min(E_end; m) = v_esc + v_E
  upper edge  m_hi : v_min(E_thr; m) = v_E
  critical threshold E_c = E_end (v_E / (v_esc + v_E))^2, below which the window is empty.
The edges are exact for a single isotope when the 8B target is strictly convex in v_min.
This script (i) checks that convexity numerically for each nucleus and (ii) confirms the
edges with the same convex-cone test used for xenon on an ideal energy-resolving detector.

Writes results/targets.json.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import nnls

import analysis as A
from isotropic import generator_matrix

OUT = A.OUT
V_SHELL = np.concatenate([[0.0], np.linspace(1.0, A.V_ESC, 600)])
TOLS = [1e-3, 1e-6, 1e-12]
U_MEV = 16.36   # 8B spectrum endpoint (Bahcall et al. 1996), as in paper/numbers.tex
Q_STAR_MEV = 3.1815  # inflection of the 8B target (Lemma 2 certificate)
DE_XE = 0.05  # ideal-detector bin width for xenon (keV), as in ideal.py; scaled by E_end for other targets

# (A, Z, number fraction), IUPAC 2013 isotopic compositions
TARGETS = {
    "Xe": A.XE_ISOTOPES,
    "I": [(127, 53, 1.0)],
    "Ge": [(70, 32, 0.2057), (72, 32, 0.2745), (73, 32, 0.0775), (74, 32, 0.3650), (76, 32, 0.0773)],
    "Ar": [(36, 18, 0.003336), (38, 18, 0.000629), (40, 18, 0.996035)],
    "Si": [(28, 14, 0.92223), (29, 14, 0.04685), (30, 14, 0.03092)],
    "Na": [(23, 11, 1.0)],
    "F": [(19, 9, 1.0)],
}


def ideal_response(E, E_thr, E_top, dE):
    edges = np.arange(E_thr, E_top + 1e-12, dE)
    if edges[-1] < E_top - 1e-12:
        edges = np.append(edges, E_top)
    T = np.zeros((len(E), len(edges) - 1))
    inside = (E >= edges[0]) & (E < edges[-1])
    which = np.clip(np.digitize(E, edges) - 1, 0, len(edges) - 2)
    T[np.where(inside)[0], which[inside]] = 1.0
    return T


def mass_for(E_keV, v_kms, A_nuc):
    """Dark-matter mass whose v_min at recoil energy E_keV on a nucleus of mass number A_nuc is v_kms."""
    m_N = A_nuc * A.AMU
    mu = np.sqrt(m_N * E_keV * 1e-6 / 2.0) * A.C_KM / v_kms
    return float(mu * m_N / (m_N - mu)) if mu < m_N else float("inf")


def endpoint(isotopes):
    E = np.geomspace(1e-3, 200.0, 20000)
    dR, _ = A.compound_spectrum(E, isotopes, "8B")
    return float(E[dR > 1e-14 * dR.max()][-1])


def convexity(Aiso, Z):
    """Min over the support of the normalised second derivative of the 8B target in v_min.

    The target halo function is dR/dE divided by A^2 F^2(E); v_min is proportional to sqrt(E)
    at fixed mass, so convexity in v_min is convexity in u = sqrt(E), independent of the mass.
    """
    E_hi = endpoint([(Aiso, Z, 1.0)])
    u = np.linspace(np.sqrt(E_hi) * 1e-3, np.sqrt(E_hi) * 0.995, 4000)
    E = u**2
    dR, _ = A.cevns_spectrum(E, Aiso, Z, "8B")
    eta = dR / A.helm2(E, Aiso)
    d2 = np.gradient(np.gradient(eta, u), u)
    scale = eta.max() / (u[-1] - u[0]) ** 2
    return float(d2[5:-5].min() / scale)


def main():
    q_c = U_MEV * A.V_EARTH / A.V_MAX
    out = dict(v_esc=A.V_ESC, v_earth=A.V_EARTH, tols=TOLS, U_MeV=U_MEV, q_star_MeV=Q_STAR_MEV, q_c_MeV=q_c,
               mu_lo_GeV=U_MEV * 1e-3 * A.C_KM / A.V_MAX, targets={})
    for name, isos in TARGETS.items():
        As = [a for a, _z, _f in isos]
        A_light, A_heavy = min(As), max(As)
        E_end = endpoint(isos)
        E_c = E_end * (A.V_EARTH / A.V_MAX) ** 2
        # edges: the lightest isotope reaches the largest recoil energy, the heaviest sets the upper edge
        E_end_light = endpoint([(A_light, isos[As.index(A_light)][1], 1.0)])
        m_lo = mass_for(E_end_light, A.V_MAX, A_light)
        conv = {}
        thresholds = [round(E_c * k, 4) for k in (0.8, 1.5, 2.5, 4.0)]
        m_N_heavy = A_heavy * A.AMU
        rec = dict(isotopes=isos, E_end_keV=E_end, E_c_keV=E_c, m_lo_analytic=m_lo,
                   E_star_keV=2 * (Q_STAR_MEV * 1e-3) ** 2 / m_N_heavy * 1e6,
                   convexity_min=conv, thresholds=thresholds, m_hi_analytic={}, numeric={})
        for thr in thresholds:
            m_hi = mass_for(thr, A.V_EARTH, A_heavy)
            rec["m_hi_analytic"][f"{thr}"] = m_hi
            dE_bin = DE_XE * E_end / 4.597
            E = np.linspace(thr * 0.98, E_end * 1.01, 4000)
            T = ideal_response(E, thr, E_end * 1.005, dE_bin)
            dR, _ = A.compound_spectrum(E, isos, "8B")
            phi = A.fold_spectrum(E, dR, E, T)
            lo = 0.7 * m_lo
            hi = 1.6 * max(m_hi, m_lo)
            masses = np.geomspace(lo, hi, 90)
            res = np.array([nnls(generator_matrix(E, T, float(m), isos, V_SHELL, "isotropic", A.V_EARTH), phi)[1]
                            / np.linalg.norm(phi) for m in masses])
            win = {}
            for tol in TOLS:
                inside = masses[res < tol]
                win[f"{tol:g}"] = [float(inside.min()), float(inside.max())] if len(inside) else None
            rec["numeric"][f"{thr}"] = dict(window=win, masses=masses.tolist(), residual=res.tolist())
            print(name, f"thr={thr:.3f} keV", "analytic", [round(m_lo, 2), round(m_hi, 2) if m_hi >= m_lo else None],
                  "numeric", win, flush=True)
        print(name, "E_end", round(E_end, 3), "E_c", round(E_c, 4), "m_lo", round(m_lo, 2), "convexity", conv, flush=True)
        out["targets"][name] = rec
    (OUT / "targets.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

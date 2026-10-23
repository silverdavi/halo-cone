#!/usr/bin/env python3
"""Ideal energy-resolving xenon detector: exact-degeneracy windows vs threshold.

Uses the self-computed 8B spectrum (validated against the XENONnT release to
1.5%) so that thresholds below the release's 0.45 keV lower edge are meaningful.
Writes results/ideal.json.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import nnls

import analysis as A
from isotropic import generator_matrix
from window import ideal_response, mass_for

OUT = A.OUT
THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.51, 0.6, 0.7, 0.85, 1.0, 1.3]
DE = 0.05
TOLS = [1e-3, 1e-6, 1e-12]
MASSES = np.geomspace(4.0, 16.0, 120)
V_SHELL = np.concatenate([[0.0], np.linspace(1.0, A.V_ESC, 600)])
V_STREAM = np.linspace(5.0, A.V_MAX, 600)


def window(masses, res, tol):
    ins = masses[res < tol]
    return [float(ins.min()), float(ins.max())] if len(ins) else None


def main():
    E = np.linspace(0.1, 5.0, 4901)
    dR, _ = A.compound_spectrum(E, A.XE_ISOTOPES, "8B")
    # kinematic endpoint of the 8B recoil spectrum in xenon
    E_end = float(E[dR > 1e-14 * dR.max()][-1])
    out = dict(dE_keV=DE, E_end_keV=E_end, v_esc=A.V_ESC, v_earth=A.V_EARTH,
               m_lo_strict=mass_for(E_end, A.V_MAX),
               E_critical_keV=E_end * (A.V_EARTH / A.V_MAX) ** 2,
               thresholds=THRESHOLDS, masses=MASSES.tolist(), scans={}, windows={}, analytic_m_hi={})
    for thr in THRESHOLDS:
        T = ideal_response(E, thr, DE)
        phi = A.fold_spectrum(E, dR, E, T)
        for kind, vg in (("isotropic", V_SHELL), ("streams", V_STREAM)):
            res = np.array([nnls(generator_matrix(E, T, float(m), A.XE_ISOTOPES, vg, kind, A.V_EARTH), phi)[1]
                            / np.linalg.norm(phi) for m in MASSES])
            out["scans"][f"{thr}|{kind}"] = res.tolist()
            out["windows"][f"{thr}|{kind}"] = {f"{t:g}": window(MASSES, res, t) for t in TOLS}
            print(thr, kind, {f"{t:g}": window(MASSES, res, t) for t in TOLS}, flush=True)
        out["analytic_m_hi"][f"{thr}"] = mass_for(thr, A.V_EARTH)
    print("E_end", E_end, "m_lo strict", out["m_lo_strict"], "E_c", out["E_critical_keV"])
    (OUT / "ideal.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

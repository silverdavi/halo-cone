#!/usr/bin/env python3
"""Window edges under variations of yields, v_E, v_esc, and target spectrum.

Writes results/robustness.json.
"""

from __future__ import annotations

import json

import numpy as np

import analysis as A
from isotropic import generator_matrix, l2_residual
from window import V_STREAM, TOL, mass_for

OUT = A.OUT
MASSES = np.unique(np.concatenate([np.geomspace(3.0, 40.0, 140), [5.3, 6.0, 7.0, 7.66, 8.0, 9.0, 10.0]]))  # same grid as window.py


def window(rows, tol=TOL):
    inside = [m for m, r in rows if r < tol]
    return (round(min(inside), 2), round(max(inside), 2)) if inside else None


def scan(E, T, phi, kind, vgrid, v_e):
    rows = []
    for m in MASSES:
        G = generator_matrix(E, T, float(m), A.XE_ISOTOPES, vgrid, kind, v_e)
        rows.append((float(m), l2_residual(G, phi)[0]))
    return rows


def main():
    energies, ly_axis, qy_axis, g0, g1, _b0, _b1 = A.load_mono_grid()
    yields = A.load_yield()
    E = np.linspace(A.E_ROI[0], A.E_ROI[1], 241)
    Es, dR_rel, _ = A.spectrum_from_release(A.XE_DATA / "signal" / "cevns_spectrum.json")
    dR = np.interp(E, Es, dR_rel)
    E_chk = np.linspace(0.45, 10.0, 400)
    dR_own, _ = A.compound_spectrum(E_chk, A.XE_ISOTOPES, "8B")
    dR_own = np.interp(E, E_chk, dR_own)

    def response(t_ly, t_qy):
        ly = A.yield_curve(yields, "ly", t_ly)
        qy = A.yield_curve(yields, "qy", t_qy)
        return np.hstack([A.response_on_grid(energies, ly_axis, qy_axis, g, E, ly, qy) for g in (g0, g1)])

    T0 = response(0.0, 0.0)
    out = {}

    # 1. yields at +-1 sigma, both classes
    for tly, tqy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
        T = response(float(tly), float(tqy))
        phi = A.fold_spectrum(E, dR, E, T)
        vsh = np.concatenate([[0.0], np.geomspace(2.0, A.V_ESC, 400)])
        out[f"yield tly={tly:+d} tqy={tqy:+d} | isotropic"] = window(scan(E, T, phi, "isotropic", vsh, A.V_EARTH))
        out[f"yield tly={tly:+d} tqy={tqy:+d} | streams"] = window(scan(E, T, phi, "streams", V_STREAM, A.V_EARTH))
        print(tly, tqy, out[f"yield tly={tly:+d} tqy={tqy:+d} | isotropic"], out[f"yield tly={tly:+d} tqy={tqy:+d} | streams"], flush=True)

    # 2. astrophysics
    phi0 = A.fold_spectrum(E, dR, E, T0)
    for v_e in (218.0, 232.0, 248.0):
        for v_esc in (500.0, 544.0, 600.0):
            vsh = np.concatenate([[0.0], np.geomspace(2.0, v_esc, 400)])
            w_iso = window(scan(E, T0, phi0, "isotropic", vsh, v_e))
            vst = np.geomspace(10.0, v_esc + v_e, 400)
            w_st = window(scan(E, T0, phi0, "streams", vst, v_e))
            out[f"vE={v_e:g} vesc={v_esc:g} | isotropic"] = w_iso
            out[f"vE={v_e:g} vesc={v_esc:g} | streams"] = w_st
            out[f"vE={v_e:g} vesc={v_esc:g} | analytic m_hi(0.51 keV)"] = round(mass_for(0.51, v_e), 2)
            print(v_e, v_esc, w_iso, w_st, round(mass_for(0.51, v_e), 2), flush=True)

    # 3. our own 8B spectrum as target instead of the released one
    phi_own = A.fold_spectrum(E, dR_own, E, T0)
    vsh = np.concatenate([[0.0], np.geomspace(2.0, A.V_ESC, 400)])
    out["own 8B spectrum | isotropic"] = window(scan(E, T0, phi_own, "isotropic", vsh, A.V_EARTH))
    out["own 8B spectrum | streams"] = window(scan(E, T0, phi_own, "streams", V_STREAM, A.V_EARTH))
    # 4. tolerance sensitivity
    rows_iso = scan(E, T0, phi0, "isotropic", vsh, A.V_EARTH)
    rows_st = scan(E, T0, phi0, "streams", V_STREAM, A.V_EARTH)
    for tol in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        out[f"tolerance {tol:g} | isotropic"] = window(rows_iso, tol)
        out[f"tolerance {tol:g} | streams"] = window(rows_st, tol)
    (OUT / "robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

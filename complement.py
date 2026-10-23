#!/usr/bin/env python3
"""Second target inside the exact-degeneracy window: CaWO4 exposure that separates
8B neutrinos from dark matter when xenon alone cannot, by halo class.

Xenon: XENONnT released observable, 8B folded through the same response as the
dark-matter generators (self-consistent). CaWO4: CRESST-III Detector A cut
survival and band acceptance (forecast, not a measurement). The halo and the
cross section are refit jointly to both datasets; the statistic is the Asimov
Poisson deviance of the best dark-matter-only fit to the neutrino expectation.

Writes results/complement.json.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import brentq

import analysis as A
from isotropic import generator_matrix, poisson_fit
from window import V_SHELL, V_STREAM

OUT = A.OUT
XE_EXPOSURES = {"xenonnt_sr0sr1": 3.517, "xenon_20ty": 20.0}
MASSES = [5.5, 6.0, 7.0, 8.0, 10.0]
CLASSES = {"streams": ("streams", V_STREAM), "isotropic": ("isotropic", V_SHELL)}


def main():
    energies, ly_axis, qy_axis, g0, g1, _b0, _b1 = A.load_mono_grid()
    yields = A.load_yield()
    ly = A.yield_curve(yields, "ly", 0.0)
    qy = A.yield_curve(yields, "qy", 0.0)
    E = np.linspace(A.E_ROI[0], A.E_ROI[1], 241)
    T_xe = A.released_response(energies, ly_axis, qy_axis, g0, g1, E, ly, qy)
    Es, dR_rel, _ = A.spectrum_from_release(A.XE_DATA / "signal" / "cevns_spectrum.json")
    phi_xe = A.fold_spectrum(E, np.interp(E, Es, dR_rel), E, T_xe)  # per tonne-year

    # CaWO4 forecast
    E_c = np.linspace(0.02, 16.0, 3000)
    eff = A.cresst_efficiencies(E_c)
    M_O, M_Ca, M_W = 15.999 * 4, 40.078, 183.84
    M_tot = M_O + M_Ca + M_W
    stoich = {"O": M_O / M_tot, "Ca": M_Ca / M_tot, "W": M_W / M_tot}
    nuclei = {"O": A.O_ISOTOPES, "Ca": A.CA_ISOTOPES, "W": A.W_ISOTOPES}
    edges_c = np.geomspace(A.CRESST_THRESHOLD, A.CRESST_EMAX, 25)
    which = np.clip(np.digitize(E_c, edges_c) - 1, 0, len(edges_c) - 2)
    T_c = np.zeros((len(E_c), len(edges_c) - 1))
    inside = (E_c >= edges_c[0]) & (E_c < edges_c[-1])
    T_c[np.where(inside)[0], which[inside]] = 1.0

    phi_c = np.zeros(T_c.shape[1])
    phi_c_b8 = np.zeros(T_c.shape[1])
    for el, isos in nuclei.items():
        for flux in A.SOLAR:
            dR, _ = A.compound_spectrum(E_c, isos, flux, stoich[el])
            binned = A.fold_spectrum(E_c, dR * eff[el], E_c, T_c)
            phi_c += binned
            if flux == "8B":
                phi_c_b8 += binned
    print(f"CaWO4 neutrino events per tonne-year (all solar): {phi_c.sum():.3f}; 8B only: {phi_c_b8.sum():.3f}")

    def cawo4_generators(m, kind, vgrid):
        G = np.zeros((T_c.shape[1], len(vgrid)))
        for el, isos in nuclei.items():
            G += generator_matrix(E_c, T_c * (eff[el] * stoich[el])[:, None], m, isos, vgrid, kind, A.V_EARTH)
        return G

    results = {}
    for cname, (kind, vgrid) in CLASSES.items():
        for m in MASSES:
            G_xe = generator_matrix(E, T_xe, m, A.XE_ISOTOPES, vgrid, kind, A.V_EARTH)
            G_c = cawo4_generators(m, kind, vgrid)
            dev_xe_only, _ = poisson_fit(G_xe, phi_xe)
            for xname, eps_xe in XE_EXPOSURES.items():
                def deviance(eps_c, rows=None):
                    Gc = G_c if rows is None else G_c[rows]
                    pc = phi_c if rows is None else phi_c[rows]
                    Gj = np.vstack([G_xe * eps_xe, Gc * eps_c])
                    pj = np.concatenate([phi_xe * eps_xe, pc * eps_c])
                    d, _ = poisson_fit(Gj, pj)
                    return d

                base = dev_xe_only * eps_xe
                # exposure for joint deviance 9, untagged CaWO4
                f = lambda le: deviance(10**le) - 9.0
                lo, hi = -6.0, 3.0
                eps_star = 10**brentq(f, lo, hi, xtol=1e-3) if f(lo) < 0 < f(hi) else (0.0 if f(lo) >= 0 else None)
                # best single bin at that exposure scale: deviance gain per bin at eps_c = 1 t y
                gains = []
                for k in range(T_c.shape[1]):
                    if phi_c[k] <= 0:
                        continue
                    gains.append((float(deviance(1.0, rows=[k]) - base), k))
                gains.sort(reverse=True)
                best_k = gains[0][1]
                fb = lambda le: deviance(10**le, rows=[best_k]) - 9.0
                eps_bin = 10**brentq(fb, lo, hi, xtol=1e-3) if fb(lo) < 0 < fb(hi) else None
                key = f"{cname}|m={m}|{xname}"
                results[key] = dict(
                    halo_class=cname, mass=m, xenon_exposure_ty=eps_xe,
                    xenon_only_deviance=float(base),
                    cawo4_untagged_ty_for_deviance_9=eps_star,
                    best_single_bin_keV=[float(edges_c[best_k]), float(edges_c[best_k + 1])],
                    best_single_bin_neutrinos_per_ty=float(phi_c[best_k]),
                    best_single_bin_ty_for_deviance_9=eps_bin,
                )
                print(f"{key:40s} xe-only dev={base:.2e}  CaWO4 t·y={eps_star}  best bin {edges_c[best_k]:.2f}-{edges_c[best_k+1]:.2f} keV t·y={eps_bin}", flush=True)

    (OUT / "complement.json").write_text(json.dumps(dict(
        note="forecast; CaWO4 efficiencies from CRESST-III Detector A release; xenon self-consistent fold",
        cawo4_neutrinos_per_ty=float(phi_c.sum()), cawo4_8b_per_ty=float(phi_c_b8.sum()),
        cresst_iii_detA_exposure_ty=A.CRESST_EXPOSURE_KG_DAY / (1000 * 365.25),
        results=results), indent=2))


if __name__ == "__main__":
    main()

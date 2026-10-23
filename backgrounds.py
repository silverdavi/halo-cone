#!/usr/bin/env python3
"""Non-degenerate backgrounds and the separation exposure of Sec. V.A.

window.py quotes the exposure at which the 8B expectation separates from the
closest isotropic dark-matter spectrum with 8B Poisson fluctuations only. Here
the released XENONnT background templates (accidental coincidences, electronic
recoils, radiogenic neutrons; SR0 and SR1 at median yield) enter both
hypotheses with a fixed normalisation,

  truth       nu_b  = eps (phi_b + beta_b)
  hypothesis  mu_b  = eps (G c + beta_b),  c >= 0,

and the Asimov deviance is minimised over the cone exactly as in window.py.
The backgrounds add variance in every bin and lengthen the exposure; whether
any of them lies inside the cone is not asked here.

Outputs results/backgrounds.json. Imports the response and generator machinery
of analysis.py, isotropic.py and window.py; does not rerun the scan.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize, nnls

import analysis as A
from isotropic import generator_matrix, poisson_fit
from window import OUT, V_SHELL

MASSES = (13.0, 15.0, 16.0, 18.0, 20.0, 25.0)
TEMPLATES = {
    "ac": "ac/template_XENONnT_{sr}_ac_cevns.h5",
    "er": "er/template_XENONnT_{sr}_er_cevns.h5",
    "rg": "rg/template_XENONnT_{sr}_rg_cevns_tly_0.0_tqy_0.0.h5",
}


def background_vector():
    """Background expectation per tonne-year of combined SR0+SR1 exposure.

    Release templates are events per tonne-year of their own science run, so
    each is weighted by its livetime share (as in analysis.released_response)
    and the two runs are concatenated in the order of the response matrix.
    Returns the summed vector and per-component, per-run bookkeeping.
    """
    tot = A.SR_LIVETIME["sr0"] + A.SR_LIVETIME["sr1"]
    parts, per_run = [], {}
    for sr in ("sr0", "sr1"):
        f = A.SR_LIVETIME[sr] / tot
        v = 0.0
        per_run[sr] = {}
        for name, pat in TEMPLATES.items():
            t = A.load_template(A.XE_DATA / pat.format(sr=sr))
            per_run[sr][name] = dict(per_ty_of_run=float(t.sum()),
                                     expected_counts=float(t.sum() * A.SR_LIVETIME[sr]))
            v = v + f * t
        parts.append(v)
    return np.concatenate(parts), per_run


def poisson_fit_offset(G, mu, beta, x0=None):
    """poisson_fit with a fixed additive expectation beta in every bin."""
    col = np.linalg.norm(G, axis=0)
    keep = col > 1e-14 * col.max()
    Gk = G[:, keep] / col[keep]
    if x0 is None:
        x0, _ = nnls(Gk, np.maximum(mu - beta, 0.0))
    x0 = np.maximum(x0, 1e-12)
    pos = mu > 0

    def dev(x):
        mh = np.maximum(Gk @ x + beta, 1e-300)
        return 2.0 * (np.sum(mu[pos] * np.log(mu[pos] / mh[pos]) - (mu[pos] - mh[pos])) + np.sum(mh[~pos]))

    def grad(x):
        mh = np.maximum(Gk @ x + beta, 1e-300)
        return Gk.T @ (1.0 - mu / mh) * 2.0

    res = minimize(dev, x0, jac=grad, method="L-BFGS-B", bounds=[(0, None)] * Gk.shape[1],
                   options=dict(maxiter=2000, ftol=1e-12, gtol=1e-9))
    x = np.zeros(G.shape[1])
    x[keep] = res.x / col[keep]
    return float(res.fun), x


def main():
    energies, ly_axis, qy_axis, g0, g1, _b0, _b1 = A.load_mono_grid()
    yields = A.load_yield()
    ly = A.yield_curve(yields, "ly", 0.0)
    qy = A.yield_curve(yields, "qy", 0.0)
    E = np.linspace(A.E_ROI[0], A.E_ROI[1], 241)
    T = A.released_response(energies, ly_axis, qy_axis, g0, g1, E, ly, qy)
    Es, dR, _ = A.spectrum_from_release(A.XE_DATA / "signal" / "cevns_spectrum.json")
    phi = A.fold_spectrum(E, np.interp(E, Es, dR), E, T)

    beta, per_run = background_vector()
    assert beta.shape == phi.shape
    tot = A.SR_LIVETIME["sr0"] + A.SR_LIVETIME["sr1"]
    counts = {name: sum(per_run[sr][name]["expected_counts"] for sr in per_run) for name in TEMPLATES}
    print(f"8B {phi.sum():.3f} /t y; backgrounds {beta.sum():.3f} /t y "
          f"(AC {counts['ac']:.1f}, ER {counts['er']:.2f}, NR {counts['rg']:.2f} events in {tot:.2f} t y)")

    rows = []
    for m in MASSES:
        G = generator_matrix(E, T, m, A.XE_ISOTOPES, V_SHELL, "isotropic", A.V_EARTH)
        dev0, c0 = poisson_fit(G, phi)
        dev1, _ = poisson_fit_offset(G, phi + beta, beta, x0=None)
        # Start the background fit from the 8B-only minimiser as well; keep the smaller deviance.
        col = np.linalg.norm(G, axis=0)
        keep = col > 1e-14 * col.max()
        dev1b, _ = poisson_fit_offset(G, phi + beta, beta, x0=(c0 * col)[keep])
        dev1 = min(dev1, dev1b)
        e0, e1 = 9.0 / dev0, 9.0 / dev1
        rows.append(dict(mass=m, deviance_per_ty_8b_only=dev0, deviance_per_ty_with_bkg=dev1,
                         exposure_ty_8b_only=e0, exposure_ty_with_bkg=e1, ratio=e1 / e0))
        print(f"m={m:5.1f} GeV  8B-only {e0:10.2f} t y   with bkg {e1:10.2f} t y   ratio {e1 / e0:.3f}", flush=True)

    ratios = [r["ratio"] for r in rows]
    out = dict(
        source="XENONnT light-WIMP data release (XENON:2024hup), templates ac, er, rg at median yield, SR0 and SR1",
        model="fixed background in both hypotheses: truth 8B+bkg, hypothesis DM cone+bkg; isotropic class, Asimov deviance 9",
        background_per_ty=float(beta.sum()),
        b8_per_ty=float(phi.sum()),
        livetime_ty=tot,
        expected_counts=counts,
        expected_counts_total=float(sum(counts.values())),
        per_run=per_run,
        rows=rows,
        ratio_range=[min(ratios), max(ratios)],
    )
    (OUT / "backgrounds.json").write_text(json.dumps(out, indent=2))
    print("wrote", OUT / "backgrounds.json")


if __name__ == "__main__":
    main()

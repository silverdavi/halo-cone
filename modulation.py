#!/usr/bin/env python3
"""Annual modulation inside the exact window: the exposure at which timing alone
separates solar 8B from its isotropic dark-matter mimic.

Inside the window the time-averaged spectrum does not distinguish the two, so
the only handle is the time series of the ROI-integrated rate.

  8B   rate follows (a/d)^2 of the Earth-Sun distance: Kepler orbit with
       eccentricity 0.0167, perihelion about 3 January, amplitude 2e = 3.3%.
  DM   an isotropic halo depends on time only through the Earth speed
       v_E(t) = 232 + 15 cos[2 pi (t - 2 June)/yr] km/s.  For the halo that
       mimics 8B at v_E = 232 (NNLS cone weights on the boosted shells, or the
       shell inversion of Theorem 1 for a single isotope) the rate at time t is
       the same shell weights re-evaluated at v_E(t).

Asimov Poisson deviance between the two binned time series with total counts
matched (the cross section is free), per tonne-year; the separation exposure is
9/deviance (3 sigma) and 25/deviance (5 sigma).

Writes results/modulation.json.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
from scipy.optimize import nnls

import analysis as A
from isotropic import generator_matrix
from window import ideal_response

OUT = A.OUT
MASSES = [6.7, 7.0, 7.6]
E_THR, E_MAX = A.E_ROI
DE_IDEAL = 0.05
V_SHELL = np.concatenate([[0.0], np.linspace(1.0, A.V_ESC, 600)])

YEAR = 365.25
ECC = 0.0167              # orbital eccentricity
T_PERI = 2.0              # perihelion, days after 1 January 00:00 (about 3 January)
V_E_MEAN = A.V_EARTH      # 232 km/s, as everywhere else in the paper
V_E_AMP = 15.0            # annual amplitude of the Earth speed (217-247 km/s)
T_VMAX = 152.5            # Earth speed maximum, about 2 June
N_TIME = 24
N_SUB = 25                # sub-samples per time bin
V_E_TABLE = np.linspace(V_E_MEAN - V_E_AMP - 1.0, V_E_MEAN + V_E_AMP + 1.0, 65)


# ----------------------------------------------------------------------------- time dependence
def earth_speed(t):
    return V_E_MEAN + V_E_AMP * np.cos(2.0 * np.pi * (t - T_VMAX) / YEAR)


def earth_sun_factor(t):
    """(a/d)^2 at time t (days), Kepler orbit, normalised to unit time average.

    The time average of (a/d)^2 is (1 - e^2)^(-1/2).
    """
    M = 2.0 * np.pi * (np.asarray(t, float) - T_PERI) / YEAR
    Ea = M.copy()
    for _ in range(30):
        Ea = Ea - (Ea - ECC * np.sin(Ea) - M) / (1.0 - ECC * np.cos(Ea))
    return (1.0 - ECC * np.cos(Ea)) ** -2 * np.sqrt(1.0 - ECC**2)


def harmonic_fit(t, y):
    """y ~ A0 + Ac cos(wt) + As sin(wt). Returns relative amplitude, day of maximum, signed June amplitude."""
    w = 2.0 * np.pi / YEAR
    X = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    a0, ac, as_ = np.linalg.lstsq(X, y, rcond=None)[0]
    amp = float(np.hypot(ac, as_) / a0)
    t_max = float((np.arctan2(as_, ac) / w) % YEAR)
    signed_june = float(amp * np.cos(w * (t_max - T_VMAX)))
    resid = float(np.max(np.abs(y - X @ [a0, ac, as_])) / a0)
    return dict(amplitude=amp, day_of_max=t_max, date_of_max=day_to_date(t_max),
                signed_amplitude_june=signed_june, residual_of_harmonic_fit=resid)


def day_to_date(d):
    return (dt.datetime(2023, 1, 1) + dt.timedelta(days=float(d))).strftime("%-d %b")


def time_bins():
    edges = np.linspace(0.0, YEAR, N_TIME + 1)
    sub = np.linspace(0.0, 1.0, N_SUB + 1)
    sub = 0.5 * (sub[1:] + sub[:-1])
    return edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * sub[None, :]   # (N_TIME, N_SUB)


# ----------------------------------------------------------------------------- deviance
def asimov_deviance(mu_true, mu_model):
    """Poisson deviance of a model with free normalisation against Asimov data mu_true."""
    mu_true = np.asarray(mu_true, float).ravel()
    mu_model = np.asarray(mu_model, float).ravel() * (mu_true.sum() / mu_model.sum())
    # Floor as in isotropic.poisson_fit: bins where the model vanishes but the data do not
    # (the 8B tail beyond the dark-matter end point at the annual minimum of v_E) hold
    # ~1e-10 of the counts and contribute 2 mu ln(mu/1e-300), which is negligible.
    mu_model = np.maximum(mu_model, 1e-300)
    pos = mu_true > 0
    return float(2.0 * (np.sum(mu_true[pos] * np.log(mu_true[pos] / mu_model[pos]) - (mu_true[pos] - mu_model[pos]))
                        + np.sum(mu_model[~pos])))


# ----------------------------------------------------------------------------- mimics
def nnls_mimic(E, T, phi, m, v_grid, isotopes, v_e=V_E_MEAN):
    G = generator_matrix(E, T, m, isotopes, v_grid, "isotropic", v_e)
    col = np.linalg.norm(G, axis=0)
    keep = col > 1e-14 * col.max()
    x, rn = nnls(G[:, keep] / col[keep], phi)
    w = np.zeros(G.shape[1])
    w[keep] = x / col[keep]
    return w, float(rn / np.linalg.norm(phi))


def min_slope_mimic(E, T, phi, m, v_grid, isotopes, tol=0.0, v_e=V_E_MEAN, dv=1.0):
    """The mimic with the smallest dR/dv_E at v_E = 232.

    dR/dv_E is linear in the shell weights, so over the polytope of exact mimics
    {x >= 0 : G x = phi} (or |G x - phi| <= tol phi per bin when exact equality
    is out of reach, as in the released observable) its minimum is a linear
    program.  The maximum is not reported: a shell at u slightly above w_1 - v_E
    adds O(delta^2) to the spectrum and O(delta) to its v_E derivative, so the
    supremum is unbounded and any computed value only measures the speed grid.
    Returns (x_min, slope_min) with the slope as d ln R / d ln v_E."""
    from scipy.optimize import linprog

    G = generator_matrix(E, T, m, isotopes, v_grid, "isotropic", v_e)
    col = np.linalg.norm(G, axis=0)
    keep = col > 1e-14 * col.max()
    Gk = G[:, keep] / col[keep]
    dG = (generator_matrix(E, T, m, isotopes, v_grid, "isotropic", v_e + dv)
          - generator_matrix(E, T, m, isotopes, v_grid, "isotropic", v_e - dv))[:, keep] / col[keep] / (2 * dv)
    c = dG.sum(axis=0) * v_e / phi.sum()          # d ln R / d ln v_E for unit-norm columns, R = phi.sum()
    scale = np.linalg.norm(phi)
    if tol == 0.0:
        res = linprog(c, A_eq=Gk, b_eq=phi / scale, bounds=(0, None), method="highs")
    else:
        A_ub = np.vstack([Gk, -Gk])
        b_ub = np.concatenate([(1 + tol) * phi, -(1 - tol) * phi]) / scale
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"linprog failed at m={m}: {res.message}")
    x = np.zeros(G.shape[1])
    x[keep] = res.x * scale / col[keep]
    return x, float(res.fun * scale)


def shell_inversion(m, a=V_E_MEAN, A_iso=131, n_w=6000, dv=2.0):
    """Shell construction of Theorem 1 for one isotope: the minimal isotropic halo
    (no shells below w_1 - a, which contribute nothing at v_E = a) whose boosted
    halo function equals the 8B target on [E_thr, E_end].

    The speed measure is F(u) du = -u dR, so the weight of a shell bin is the
    bin's mean speed times the drop of R across it.  Bins of dv = 2 km/s average
    over the kinks that the trapezoid rule over the 1000 flux nodes puts into
    the second derivative of the target.  Returns shell speeds, weights, and
    diagnostics."""
    E_nu, _shape, _ = A.load_flux("8B")
    U = float(E_nu.max())                                    # neutrino end point (MeV)
    m_N_MeV = A_iso * A.AMU * 1e3
    E_end = 2.0 * U**2 / m_N_MeV * 1e3                       # keV
    E = np.linspace(E_THR, E_end, n_w)
    dR, _ = A.cevns_spectrum(E, A_iso, 54, "8B")
    eta_t = dR / (A_iso**2 * A.helm2(E, A_iso))               # target halo function, up to a constant
    w = A.vmin_kms(E, m, A_iso)
    q = -2.0 * a * np.gradient(eta_t, w)                     # q_nu(w) = -2 a eta_nu'(w), zero beyond w_2
    w1, w2 = float(w[0]), float(w[-1])
    edges = np.arange(w1 - a, w2 - a + dv, dv)
    R = np.zeros_like(edges)
    j = 0
    while edges[0] + (2 * j + 1) * a <= w2:                  # Eq. (series): finite sum
        R += np.interp(edges + (2 * j + 1) * a, w, q, left=0.0, right=0.0)
        j += 1
    v = 0.5 * (edges[1:] + edges[:-1])
    weights = v * (R[:-1] - R[1:])
    neg = float(np.abs(weights[weights < 0].sum()) / weights[weights > 0].sum())
    return v, np.clip(weights, 0.0, None), dict(w1=w1, w2=w2, u_min=float(edges[0]), u_max=float(edges[-1]),
                                                E_end_keV=E_end, negative_weight_fraction=neg)


# ----------------------------------------------------------------------------- rates vs v_E
def rate_table(E, T, m, v_grid, weights, isotopes):
    """Expected ROI spectrum (per t y, up to the free scale) for each v_E in V_E_TABLE: (n_vE, n_bins)."""
    return np.array([generator_matrix(E, T, m, isotopes, v_grid, "isotropic", ve) @ weights for ve in V_E_TABLE])


def analyse(E, T, phi, m, v_grid, weights, isotopes, label):
    spec = rate_table(E, T, m, v_grid, weights, isotopes)          # (n_vE, n_bins)
    tot = spec.sum(axis=1)
    g = lambda ve: np.interp(ve, V_E_TABLE, tot)                   # noqa: E731
    tt = time_bins()
    ve = earth_speed(tt)
    t_fine = np.linspace(0.0, YEAR, 2000, endpoint=False)
    dm_fit = harmonic_fit(t_fine, g(earth_speed(t_fine)))
    b8_fit = harmonic_fit(t_fine, earth_sun_factor(t_fine))
    # ROI-integrated time series per tonne-year (bin width / year)
    rate_b8 = float(phi.sum())
    mu_b8 = rate_b8 * earth_sun_factor(tt).mean(axis=1) / N_TIME
    mu_dm = g(ve).mean(axis=1) / N_TIME
    dev_t = asimov_deviance(mu_b8, mu_dm)
    # time x energy bins (the DM spectrum shape also moves with v_E; 8B shape does not)
    mu_b8_2d = np.outer(earth_sun_factor(tt).mean(axis=1) / N_TIME, phi)
    spec_t = np.array([[np.interp(v, V_E_TABLE, spec[:, b]) for b in range(spec.shape[1])] for v in ve.ravel()])
    mu_dm_2d = spec_t.reshape(N_TIME, N_SUB, -1).mean(axis=1) / N_TIME
    dev_te = asimov_deviance(mu_b8_2d, mu_dm_2d)
    i0 = np.argmin(np.abs(V_E_TABLE - V_E_MEAN))
    dlng = float(np.gradient(np.log(tot), np.log(V_E_TABLE))[i0])
    out = dict(mass=m, rate_b8_per_ty=rate_b8, b8=b8_fit, dm=dm_fit, dlnR_dlnvE_at_232=dlng,
               deviance_per_ty_time=dev_t, exposure_3sigma_ty=9.0 / dev_t, exposure_5sigma_ty=25.0 / dev_t,
               deviance_per_ty_time_energy=dev_te, exposure_3sigma_ty_time_energy=9.0 / dev_te,
               rate_vs_vE=dict(v_E=V_E_TABLE.tolist(), rate=(tot / tot[i0]).tolist()))
    print(f"{label:28s} m={m:4.1f}  R8B={rate_b8:5.2f}/ty  "
          f"DM amp={100 * dm_fit['amplitude']:5.2f}% max {dm_fit['date_of_max']:>6s} (dlnR/dlnvE={dlng:+.2f})  "
          f"8B amp={100 * b8_fit['amplitude']:4.2f}% max {b8_fit['date_of_max']}  "
          f"D/ty={dev_t:.3e}  eps(3s)={9 / dev_t:8.0f} ty  [time x energy {9 / dev_te:8.0f} ty]", flush=True)
    return out


def worst_case(E, T, phi, m, v_grid, isotopes, tol, label):
    """The mimic with the smallest modulation, run through the full time-series analysis.

    Every mimic has dR/dv_E > 0 (maximum near 2 June) while 8B peaks near 3 January, five
    months away, so the deviance grows with the modulation and the smallest-slope mimic is
    the hardest to separate."""
    x, slope = min_slope_mimic(E, T, phi, m, v_grid, isotopes, tol)
    row = analyse(E, T, phi, m, v_grid, x, isotopes, f"{label}, min dR/dv_E")
    row.pop("rate_vs_vE")
    row.pop("b8")
    row["tolerance"] = tol
    row["n_shells"] = int(np.sum(x > 0))
    row["dlnR_dlnvE_linear_program"] = slope
    return row


RELEASED_TOL = 1e-3   # per-bin relative band for the released observable, where exact equality is out of reach


def main():
    results = dict(
        model=dict(eccentricity=ECC, perihelion_day=T_PERI, perihelion_date=day_to_date(T_PERI),
                   v_E_mean=V_E_MEAN, v_E_amplitude=V_E_AMP, v_E_max_day=T_VMAX, v_E_max_date=day_to_date(T_VMAX),
                   year_days=YEAR, time_bins=N_TIME, roi_keV=[E_THR, E_MAX], ideal_bin_keV=DE_IDEAL,
                   deviance="Asimov Poisson, 8B as data, isotropic mimic as model with free normalisation",
                   note="shells below w_1 - v_E contribute nothing at v_E = 232 and are absent from all mimics; "
                        "the NNLS mimic is a vertex of the cone, the inversion is the unique minimal single-isotope halo, "
                        "worst_case is the exact mimic with the smallest dR/dv_E (linear program) and hence the "
                        "largest separation exposure; the largest dR/dv_E over mimics is unbounded"),
        ideal={}, released={}, inversion_131={})

    # --- ideal energy-resolving detector, natural xenon, 0.51 keV threshold, 0.05 keV bins
    E = np.linspace(0.1, E_MAX, 4901)
    dR, _ = A.compound_spectrum(E, A.XE_ISOTOPES, "8B")
    T = ideal_response(E, E_THR, DE_IDEAL)
    phi = A.fold_spectrum(E, dR, E, T)
    for m in MASSES:
        w, resid = nnls_mimic(E, T, phi, m, V_SHELL, A.XE_ISOTOPES)
        row = analyse(E, T, phi, m, V_SHELL, w, A.XE_ISOTOPES, "ideal 0.51 keV, NNLS")
        row["mimic_residual"] = resid
        row["n_shells"] = int(np.sum(w > 0))
        row["worst_case"] = worst_case(E, T, phi, m, V_SHELL, A.XE_ISOTOPES, 0.0, "ideal 0.51 keV")
        results["ideal"][f"{m}"] = row

    # --- shell inversion of Theorem 1, single isotope 131Xe (unique minimal halo)
    E1 = np.linspace(0.1, E_MAX, 4901)
    iso131 = [(131, 54, 1.0)]
    dR1, _ = A.compound_spectrum(E1, iso131, "8B")
    T1 = ideal_response(E1, E_THR, DE_IDEAL)
    phi1 = A.fold_spectrum(E1, dR1, E1, T1)
    for m in MASSES:
        v, wts, diag = shell_inversion(m)
        G = generator_matrix(E1, T1, m, iso131, v, "isotropic", V_E_MEAN)
        pred = G @ wts
        resid = float(np.linalg.norm(phi1 - pred * (phi1 @ pred) / (pred @ pred)) / np.linalg.norm(phi1))
        row = analyse(E1, T1, phi1, m, v, wts, iso131, "131Xe shell inversion")
        row["mimic_residual"] = resid
        row.update(diag)
        results["inversion_131"][f"{m}"] = row

    # --- released XENONnT observable (SR0+SR1 cS1-cS2 templates)
    energies, ly_axis, qy_axis, g0, g1, _b0, _b1 = A.load_mono_grid()
    yields = A.load_yield()
    E_rel = np.linspace(E_THR, E_MAX, 241)
    T_rel = A.released_response(energies, ly_axis, qy_axis, g0, g1, E_rel,
                                A.yield_curve(yields, "ly", 0.0), A.yield_curve(yields, "qy", 0.0))
    dR_rel, _ = A.compound_spectrum(E_rel, A.XE_ISOTOPES, "8B")
    phi_rel = A.fold_spectrum(E_rel, dR_rel, E_rel, T_rel)
    v_rel = np.concatenate([[0.0], np.geomspace(2.0, A.V_ESC, 400)])
    for m in MASSES:
        w, resid = nnls_mimic(E_rel, T_rel, phi_rel, m, v_rel, A.XE_ISOTOPES)
        row = analyse(E_rel, T_rel, phi_rel, m, v_rel, w, A.XE_ISOTOPES, "released observable, NNLS")
        row["mimic_residual"] = resid
        row["n_shells"] = int(np.sum(w > 0))
        row["worst_case"] = worst_case(E_rel, T_rel, phi_rel, m, v_rel, A.XE_ISOTOPES, RELEASED_TOL, "released observable")
        results["released"][f"{m}"] = row

    (OUT / "modulation.json").write_text(json.dumps(results, indent=1))
    print("wrote", OUT / "modulation.json")


if __name__ == "__main__":
    main()

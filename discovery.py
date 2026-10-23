#!/usr/bin/env python3
"""Halo-independent discovery limit in the XENONnT observable.

For a given mass and exposure, the worst case over isotropic Galactic-frame
halos of the Asimov significance for rejecting "8B only" (flux nuisance with a
3.81% Gaussian prior) in favour of "8B + dark matter". The discovery limit is
the cross section at which that worst-case significance reaches 3 sigma.
Compared with the Standard Halo Model at the same exposure.

Writes results/discovery.json and results/discovery.png.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import minimize

import analysis as A
from isotropic import generator_matrix

OUT = A.OUT
RHO = 0.3          # GeV / cm^3
AMU_G = 1.66054e-24
FLUX_UNC = 0.0381
Q_DISC = 9.0       # 3 sigma Asimov
V_SHELL = np.concatenate([[0.0], np.geomspace(3.0, A.V_ESC, 90)])
V0_SHM = 238.0
SEC_PER_YEAR = 365.25 * 86400


def prefactor(m_chi):
    """Events per tonne-year per cm^2, multiplying generator_matrix output
    (which carries w A^2 F^2 eta dE with eta in s/km)."""
    mu_p = m_chi * 0.938272 / (m_chi + 0.938272)
    n_chi = RHO / m_chi
    per_tonne = 1e6 / AMU_G
    return n_chi * per_tonne * (A.AMU / (2.0 * mu_p**2)) * 1e-6 * A.C_KM**2 * 1e5 * SEC_PER_YEAR


def shm_weights(v_grid, v0=V0_SHM, v_esc=A.V_ESC):
    f = v_grid**2 * np.exp(-(v_grid / v0) ** 2)
    f[v_grid > v_esc] = 0.0
    # trapezoid weights on an irregular grid
    dv = np.gradient(v_grid)
    w = f * dv
    return w / w.sum()


def alpha_star(mu, n, delta):
    """Profile the 8B flux scale alpha with Gaussian prior (alpha-1)^2/delta^2 (times 1, i.e. 2*0.5)."""
    S_mu, S_n = mu.sum(), n.sum()
    # d/dalpha [ 2(alpha S_mu - S_n ln alpha) + (alpha-1)^2/delta^2 ] = 0
    a2 = 2.0 / delta**2
    a1 = 2.0 * S_mu - 2.0 / delta**2
    a0 = -2.0 * S_n
    return (-a1 + np.sqrt(a1 * a1 - 4 * a2 * a0)) / (2 * a2)


def q_asimov(mu, s, delta):
    """Asimov -2 ln Lambda for H0: alpha*mu vs truth mu+s, alpha profiled. Returns q and dq/ds."""
    n = mu + s
    al = alpha_star(mu, n, delta)
    mod = al * mu
    pos = n > 0
    q = 2.0 * (np.sum(mod - n) + np.sum(n[pos] * np.log(n[pos] / mod[pos]))) + (al - 1) ** 2 / delta**2
    grad = np.zeros_like(n)
    grad[pos] = 2.0 * np.log(n[pos] / mod[pos])
    return q, grad


def worst_case_q(mu_eps, Gp_eps, f0=None):
    """min over f on the simplex of q(mu, G f). Convex in f. SLSQP with analytic gradient."""
    k = Gp_eps.shape[1]

    def fun(f):
        s = Gp_eps @ f
        q, g = q_asimov(mu_eps, s, FLUX_UNC)
        return q, Gp_eps.T @ g

    # Vertices first: the minimum of a convex function over the simplex is often at or near one.
    vq = np.array([q_asimov(mu_eps, Gp_eps[:, j], FLUX_UNC)[0] for j in range(k)])
    jbest = int(np.argmin(vq))
    if vq[jbest] <= 1e-12:
        f = np.zeros(k); f[jbest] = 1.0
        return 0.0, f
    starts = [np.eye(k)[jbest]]
    if f0 is not None:
        starts.append(f0)
    best = (vq[jbest], np.eye(k)[jbest])
    for s0 in starts:
        res = minimize(fun, s0, jac=True, method="SLSQP",
                       bounds=[(0.0, 1.0)] * k,
                       constraints=[{"type": "eq", "fun": lambda f: f.sum() - 1.0, "jac": lambda f: np.ones(k)}],
                       options=dict(maxiter=400, ftol=1e-12))
        if res.fun < best[0]:
            best = (float(res.fun), np.clip(res.x, 0, None) / max(res.x.sum(), 1e-300))
    return best


def diagnose_minimizer(mu, G, eps, sigma):
    """f* of worst_case_q at the discovery cross section. Not used by the bisection."""
    infinite = not np.isfinite(sigma)
    sig = 1e-36 if infinite else float(sigma)
    mu_e = mu * eps
    Gp = G * sig * eps
    k = Gp.shape[1]
    vq = np.array([q_asimov(mu_e, Gp[:, j], FLUX_UNC)[0] for j in range(k)])
    j_vertex = int(np.argmin(vq))
    q_opt, f = worst_case_q(mu_e, Gp)
    f = np.asarray(f, float)
    j_hat = int(np.argmax(f))
    q_drop = float(vq[j_vertex] - q_opt)
    # Solver noise can shed 1e-4 of the weight without lowering q. A move is a
    # lower deviance, or a different shell carrying the mass.
    left = bool(q_drop > 1e-6 or (j_hat != j_vertex and float(f[j_vertex]) < 0.999))
    support = [
        dict(index=int(j), speed_kms=float(V_SHELL[j]), weight=float(f[j]))
        for j in range(k) if f[j] > 1e-4
    ]
    second = float(np.partition(vq, 1)[1])
    spec = Gp @ f
    mu_e = mu * eps
    shape_tv = None
    if spec.sum() > 0 and mu_e.sum() > 0:
        scaled = spec * (mu_e.sum() / spec.sum())
        shape_tv = float(0.5 * np.abs(scaled - mu_e).sum() / mu_e.sum())
    return dict(
        sigma_cm2=None if infinite else float(sigma),
        diagnosed_at_cm2=float(sig),
        sigma_infinite=bool(infinite),
        vertex_index=j_vertex,
        vertex_speed_kms=float(V_SHELL[j_vertex]),
        vertex_q=float(vq[j_vertex]),
        second_vertex_q=second,
        opt_q=float(q_opt),
        q_drop=q_drop,
        sqp_left_vertex=left,
        weight_on_u0=float(f[0]),
        argmax_index=j_hat,
        argmax_speed_kms=float(V_SHELL[j_hat]),
        argmax_weight=float(f[j_hat]),
        n_above_1em3=int(np.sum(f > 1e-3)),
        shape_tv=shape_tv,
        counts_per_ty_per_cm2=float(spec.sum() / (sig * eps)) if sig * eps else None,
        support=support,
    )


def shape_vs_b8(mu, G, f_shm):
    """u=0 shell against the 8B histogram, and its rate relative to the SHM."""
    s = np.asarray(G[:, 0], float)
    shm = np.asarray(G @ f_shm, float)
    out = dict(
        u0_counts_per_ty_per_cm2=float(s.sum()),
        shm_counts_per_ty_per_cm2=float(shm.sum()),
    )
    if s.sum() <= 0 or shm.sum() <= 0:
        out["u0_over_shm"] = 0.0
        out["shape_tv"] = None
        out["cosine"] = None
        return out
    out["u0_over_shm"] = float(s.sum() / shm.sum())
    scaled = s * (mu.sum() / s.sum())
    out["shape_tv"] = float(0.5 * np.abs(scaled - mu).sum() / mu.sum())
    out["cosine"] = float(np.dot(s, mu) / (np.linalg.norm(s) * np.linalg.norm(mu)))
    return out


def discovery_sigma(mu, G, eps, worst=True, f_fixed=None, lo=-50.0, hi=-36.0):
    """Cross section (cm^2) where the (worst-case) Asimov q reaches Q_DISC at exposure eps."""
    mu_e = mu * eps
    f_prev = None

    def Q(logsig):
        Gp = G * (10.0**logsig) * eps
        nonlocal f_prev
        if worst:
            q, f_prev = worst_case_q(mu_e, Gp, f_prev)
            return q
        q, _ = q_asimov(mu_e, Gp @ f_fixed, FLUX_UNC)
        return q

    if Q(hi) < Q_DISC:
        return np.inf
    a, b = lo, hi
    for _ in range(40):
        mid = 0.5 * (a + b)
        if Q(mid) >= Q_DISC:
            b = mid
        else:
            a = mid
        if b - a < 0.001:
            break
    return 10.0 ** (0.5 * (a + b))


def plot_discovery(payload):
    """Discovery limit and fog index. `payload` is a discovery.json dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "config"))
    import figstyle as fs
    from window import apply_log_axis, warn_overlaps

    fs.set_style()
    edge, shm = fs.role("edge"), fs.role("shm")
    results = payload["results"]
    masses = [m for m in ("6", "8", "10", "12", "13", "14", "16", "20", "30", "50") if m in results]
    colors = fs.sequential(len(masses))
    # Released isotropic window, so the 6 GeV degeneracy sits in context.
    win = json.loads((OUT / "window.json").read_text())["released_windows_GeV"]["released|isotropic"]

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.9))
    for m, c in zip(masses, colors):
        rows = results[m]
        x = [r["exposure_ty"] for r in rows]
        yh = [r["sigma_halo_independent"] if np.isfinite(r["sigma_halo_independent"]) else np.nan for r in rows]
        ys = [r["sigma_shm"] if np.isfinite(r["sigma_shm"]) else np.nan for r in rows]
        finite = any(np.isfinite(v) for v in yh)
        ax[0].plot(x, yh, color=c, lw=1.35, label=(m if finite else None), zorder=3)
        ax[0].plot(x, ys, color=(shm if m == "6" else c), ls=":", lw=1.05 if m == "6" else 0.9,
                   label=("6 SHM" if m == "6" else None), zorder=2)
        nh = [r.get("n_sigma_halo_independent", np.nan) for r in rows]
        ns = [r.get("n_sigma_shm", np.nan) for r in rows]
        nh = [v if np.isfinite(v) and v < 500 else np.nan for v in nh]
        ns = [v if np.isfinite(v) and v < 500 else np.nan for v in ns]
        if any(np.isfinite(v) for v in nh):
            ax[1].plot(x, nh, color=c, lw=1.35, label=m, zorder=3)
        ax[1].plot(x, ns, color=(shm if m == "6" else c), ls=":", lw=1.05 if m == "6" else 0.9,
                   label=("6 SHM" if m == "6" else None), zorder=2)

    ax[0].text(0.98, 0.97,
               "solid: worst isotropic halo\ndotted: SHM at that mass\nno 6 GeV solid curve:\nworst halo is exactly degenerate",
               transform=ax[0].transAxes, fontsize=6.5, ha="right", va="top", color=edge)
    apply_log_axis(ax[0], "x", 0.25, 4e4)
    apply_log_axis(ax[0], "y", 8e-49, 5e-42)
    ax[0].set_xlabel(r"xenon exposure (t y)")
    ax[0].set_ylabel(r"$3\sigma$ discovery cross section (cm$^2$)")
    ax[0].set_title(rf"6 GeV lies in the {win[0]:.1f}–{win[1]:.0f} GeV degeneracy window")
    fs.legend_below(ax[0], ncol=5, y=-0.34, title=r"$m_\chi$ (GeV)")
    fs.panel_label(ax[0], "a", dx=-0.02, dy=1.08)

    ax[1].axhline(2, color=edge, lw=0.6, zorder=1)
    ax[1].axhline(1, color=edge, lw=0.6, zorder=1)
    ax[1].text(0.40, 2.55, r"$n=2$ Poisson", color=edge, fontsize=6.5, va="bottom")
    ax[1].text(0.40, 1.15, r"$n=1$ background-free", color=edge, fontsize=6.5, va="bottom")
    apply_log_axis(ax[1], "x", 0.25, 4e4)
    apply_log_axis(ax[1], "y", 0.7, 80, [1, 2, 10, 30])
    ax[1].set_xlabel(r"xenon exposure (t y)")
    ax[1].set_ylabel(r"fog index $n=-(\mathrm{d}\ln\sigma/\mathrm{d}\ln\varepsilon)^{-1}$")
    ax[1].set_title("Isotropic search saturates; SHM stays near Poisson")
    fs.legend_below(ax[1], ncol=5, y=-0.34, title=r"$m_\chi$ (GeV)")
    fs.panel_label(ax[1], "b", dx=-0.02, dy=1.08)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.26, wspace=0.32)
    warn_overlaps(fig)
    path = OUT / "discovery.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("wrote", path)


def main():
    if "--plot-only" in sys.argv:
        plot_discovery(json.loads((OUT / "discovery.json").read_text()))
        return
    energies, ly_axis, qy_axis, g0, g1, _b0, _b1 = A.load_mono_grid()
    yields = A.load_yield()
    ly = A.yield_curve(yields, "ly", 0.0)
    qy = A.yield_curve(yields, "qy", 0.0)
    E = np.linspace(A.E_ROI[0], A.E_ROI[1], 241)
    T = A.released_response(energies, ly_axis, qy_axis, g0, g1, E, ly, qy)
    Es, dR_rel, _ = A.spectrum_from_release(A.XE_DATA / "signal" / "cevns_spectrum.json")
    mu = A.fold_spectrum(E, np.interp(E, Es, dR_rel), E, T)  # 8B per tonne-year
    print(f"8B per t y in observable: {mu.sum():.3f}")

    # Normalization check against the released 6 GeV SHM spectrum (rate at 1e-45 cm^2).
    rel6 = json.load(open(A.XE_DATA / "signal" / "wimp_si_spectrum_6.json"))
    E_fine = np.linspace(0.3, 5.0, 2000)  # the release integrates its spectra over 0.3-5 keV
    T_all = np.ones((len(E_fine), 1))
    G6 = generator_matrix(E_fine, T_all, 6.0, A.XE_ISOTOPES, V_SHELL, "isotropic", A.V_EARTH)
    rate6 = prefactor(6.0) * 1e-45 * float((G6 @ shm_weights(V_SHELL))[0])
    print(f"6 GeV SHM total rate at 1e-45: ours {rate6:.1f} / release {rel6['rate']:.1f} per t y")

    masses = [6.0, 8.0, 10.0, 12.0, 13.0, 14.0, 16.0, 20.0, 30.0, 50.0]
    exposures = np.geomspace(0.3, 3e4, 11)
    f_shm = shm_weights(V_SHELL)
    out = {}
    fstar = {}
    for m in masses:
        G = prefactor(m) * generator_matrix(E, T, m, A.XE_ISOTOPES, V_SHELL, "isotropic", A.V_EARTH)
        shape = shape_vs_b8(mu, G, f_shm)
        rows = []
        diags = []
        for eps in exposures:
            s_hi = discovery_sigma(mu, G, eps, worst=True)
            s_shm = discovery_sigma(mu, G, eps, worst=False, f_fixed=f_shm)
            rows.append(dict(exposure_ty=float(eps), sigma_halo_independent=float(s_hi), sigma_shm=float(s_shm)))
            diag = diagnose_minimizer(mu, G, eps, s_hi)
            diags.append(diag)
            rate = diag["counts_per_ty_per_cm2"] / shape["shm_counts_per_ty_per_cm2"] if shape["shm_counts_per_ty_per_cm2"] else 0.0
            print(
                f"m={m:5.1f} eps={eps:9.2f}  HI {s_hi:.3e}  SHM {s_shm:.3e}  "
                f"f* u={diag['argmax_speed_kms']:.1f} w0={diag['weight_on_u0']:.4f} "
                f"left={diag['sqp_left_vertex']} dq={diag['q_drop']:.3e} "
                f"tv={diag['shape_tv']} rate={rate:.4g}",
                flush=True,
            )
        # fog index (O'Hare): n = -(d ln sigma / d ln eps)^-1 ; n=1 background free, n=2 Poisson, n->inf saturation
        for i in range(1, len(rows) - 1):
            for key in ("sigma_halo_independent", "sigma_shm"):
                a, b = rows[i - 1][key], rows[i + 1][key]
                if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                    slope = (np.log(b) - np.log(a)) / (np.log(rows[i + 1]["exposure_ty"]) - np.log(rows[i - 1]["exposure_ty"]))
                    rows[i][f"n_{key}"] = float(-1.0 / slope) if slope < -1e-6 else 1e3
        out[f"{m:g}"] = rows
        speeds = sorted({d["argmax_speed_kms"] for d in diags})
        fstar[f"{m:g}"] = dict(shape=shape, all_u0=speeds == [0.0] and not any(d["sqp_left_vertex"] for d in diags), rows=diags)
        print(
            f"  shape u0/SHM={shape['u0_over_shm']:.6g} tv={shape['shape_tv']} "
            f"cos={shape['cosine']} argmax speeds={speeds}",
            flush=True,
        )

    payload = dict(
        note="Asimov 3 sigma discovery, 8B flux nuisance 3.81%, no other backgrounds; XENONnT SR0+SR1 observable",
        v_esc=A.V_ESC, v_earth=A.V_EARTH, v0_shm=V0_SHM, rho=RHO,
        shm_6GeV_rate_check=dict(ours=rate6, release=rel6["rate"]),
        results=out,
        fstar=dict(
            note="Diagnostic only. results[] cross sections are the bisection; fstar is worst_case_q at that cross section (1e-36 if the bisection returned infinity). shape_tv is half the L1 distance between the u=0 spectrum and 8B after matching total counts.",
            by_mass=fstar,
        ))
    (OUT / "discovery.json").write_text(json.dumps(payload, indent=2))
    plot_discovery(payload)


if __name__ == "__main__":
    main()

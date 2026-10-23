#!/usr/bin/env python3
"""Separation exposures with XENONnT background nuisances profiled.

The truth is the nominal 8B plus background expectation.  The alternative is
the isotropic dark-matter cone plus the released AC, ER, and radiogenic
templates.  AC has separate SR0 and SR1 rates, while ER and radiogenic rates
are shared across runs, matching statistical_model_base.yaml.

Radiogenic yield shapes use a simplicial, piecewise-linear interpolation of
the release nodes with |t_Ly|, |t_Qy| <= 1.  Within one triangle, nonnegative
node coefficients make the expected counts affine; their sum is the
radiogenic normalization.  Each triangle is therefore a convex subproblem.
The union of triangles is searched explicitly.  The free result uses the
conic hull of all nine one-node-range templates and relaxes interpolation.

Outputs results/profiled_backgrounds.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize, nnls

import analysis as A
from backgrounds import TEMPLATES, background_vector, poisson_fit_offset
from isotropic import generator_matrix
from window import OUT, V_SHELL

MASSES = (13.0, 15.0, 16.0, 18.0, 20.0, 25.0)
RUNS = ("sr0", "sr1")
PROFILE_YIELD_NODES = (-1.0, 0.0, 1.0)
RELEASE_YIELD_NODES = tuple(float(x) for x in range(-3, 4))
SIGMAS = {
    "ac_sr0": 0.0904,
    "ac_sr1": 0.0576,
    "er": 1.0,
    "rg": 0.58,
}
SIGMA_SOURCE = (
    "data/xenonnt/light_wimp_data_release/data/statistical_model_base.yaml: "
    "ac_sr0_rate_multiplier uncertainty 0.0904, ac_sr1_rate_multiplier 0.0576, "
    "er_rate_multiplier 1.0, nr_rate_multiplier 0.58; relative_uncertainty true. "
    "XENON:2024hup Eq. (1) states Gaussian ancillary constraints and Table I "
    "gives the rounded combined expectations; the background text gives the "
    "rounded per-run rates and uncertainties."
)
YIELD_SOURCE = (
    "data/xenonnt/light_wimp_data_release/data/statistical_model_base.yaml: "
    "t_ly and t_qy are shared shape parameters with uncertainty 1, anchors "
    "-3,...,3, and fit limits [-3,3]; data/xenonnt/light_wimp_data_release/"
    "utils.py:get_yield defines t=0 as the median YBe model and t=+/-1 as the "
    "upper/lower model. This analysis profiles the requested one-node range."
)


@dataclass
class FitResult:
    deviance: float
    rate: np.ndarray
    dm_coefficients: np.ndarray
    nuisance_coefficients: np.ndarray
    nuisance_names: list[str]
    yield_tly: float
    yield_tqy: float
    triangle: list[tuple[float, float]] | None
    iterations: int
    kkt_max: float
    active_stationarity: float
    gradient_fd_error: float | None = None


def _component_vectors():
    """Livetime-weighted release templates with their release correlations."""
    total = sum(A.SR_LIVETIME.values())
    fractions = {sr: A.SR_LIVETIME[sr] / total for sr in RUNS}
    nbin = A.load_template(A.XE_DATA / TEMPLATES["ac"].format(sr="sr0")).size

    ac = {}
    for sr in RUNS:
        run = np.zeros(2 * nbin)
        sl = slice(0, nbin) if sr == "sr0" else slice(nbin, 2 * nbin)
        run[sl] = fractions[sr] * A.load_template(
            A.XE_DATA / TEMPLATES["ac"].format(sr=sr)
        )
        ac[sr] = run

    er = np.concatenate(
        [
            fractions[sr]
            * A.load_template(A.XE_DATA / TEMPLATES["er"].format(sr=sr))
            for sr in RUNS
        ]
    )
    rg = {}
    for tly in RELEASE_YIELD_NODES:
        for tqy in RELEASE_YIELD_NODES:
            rg[(tly, tqy)] = np.concatenate(
                [
                    fractions[sr]
                    * A.load_template(
                        A.XE_DATA
                        / "rg"
                        / (
                            f"template_XENONnT_{sr}_rg_cevns_"
                            f"tly_{tly:.1f}_tqy_{tqy:.1f}.h5"
                        )
                    )
                    for sr in RUNS
                ]
            )

    summed, _ = background_vector()
    nominal = ac["sr0"] + ac["sr1"] + er + rg[(0.0, 0.0)]
    if not np.allclose(summed, nominal, rtol=1e-13, atol=1e-14):
        raise RuntimeError("component templates do not reproduce background_vector()")
    return ac, er, rg, nominal


def _triangles():
    """Eight triangles covering the four cells in [-1,1]^2."""
    result = []
    for x0 in (-1.0, 0.0):
        for y0 in (-1.0, 0.0):
            a, b = (x0, y0), (x0 + 1.0, y0)
            c, d = (x0, y0 + 1.0), (x0 + 1.0, y0 + 1.0)
            result.extend([(a, b, d), (a, d, c)])
    return result


def _poisson_deviance(truth, fitted):
    fitted = np.maximum(fitted, 1e-300)
    delta = (truth - fitted) / fitted
    zero = truth <= 0
    small = np.abs(delta) < 1e-4
    term = np.empty_like(delta)
    d = delta[small]
    term[small] = d**2 / 2.0 - d**3 / 6.0 + d**4 / 12.0 - d**5 / 20.0
    regular = ~small & ~zero
    d = delta[regular]
    term[regular] = (1.0 + d) * np.log1p(d) - d
    term[zero] = 1.0
    return 2.0 * float(np.sum(fitted * term))


class ConvexFit:
    """Active-column solver for one affine background model."""

    def __init__(
        self,
        dm: np.ndarray,
        truth_rate: np.ndarray,
        nuisance_columns: list[np.ndarray],
        nuisance_names: list[str],
        sigma_scale: float | None,
        rg_indices: list[int],
        rg_nodes: list[tuple[float, float]],
    ):
        self.truth_rate = truth_rate
        self.n_dm = dm.shape[1]
        columns = [dm[:, j] for j in range(self.n_dm)] + nuisance_columns
        norms = np.array([np.linalg.norm(v) for v in columns])
        if np.any(norms <= 0):
            raise ValueError("zero design column")
        self.design = np.column_stack(columns) / norms
        self.norms = norms
        self.nuisance_names = nuisance_names
        self.sigma_scale = sigma_scale
        self.rg_indices = np.array([self.n_dm + i for i in rg_indices], dtype=int)
        self.rg_nodes = np.asarray(rg_nodes, float)

        self.norm_groups: dict[str, list[int]] = {}
        for i, name in enumerate(nuisance_names):
            group = "rg" if name.startswith("rg[") else name
            self.norm_groups.setdefault(group, []).append(self.n_dm + i)

        nominal_background = np.zeros_like(truth_rate)
        for name, column in zip(nuisance_names, nuisance_columns):
            if not name.startswith("rg[") or name == "rg[+0,+0]":
                nominal_background += column
        target = np.maximum(truth_rate - nominal_background, 0.0)
        dm0, _ = nnls(self.design[:, : self.n_dm], target)
        self.initial_dm = np.flatnonzero(dm0 > 1e-10 * max(dm0.max(), 1.0))
        if not len(self.initial_dm):
            self.initial_dm = np.array(
                [int(np.argmax(self.design[:, : self.n_dm].T @ truth_rate))]
            )

    def _prior(self, x, gradient):
        if self.sigma_scale is None:
            return 0.0
        penalty = 0.0
        for name, indices in self.norm_groups.items():
            coefficient = sum(x[j] / self.norms[j] for j in indices)
            theta = coefficient - 1.0
            sigma = SIGMAS[name] * self.sigma_scale
            penalty += (theta / sigma) ** 2
            if gradient is not None:
                for j in indices:
                    gradient[j] += 2.0 * theta / (sigma**2 * self.norms[j])
        return penalty

    def objective_gradient(self, x, exposure):
        rate = np.maximum(self.design @ x, 1e-300)
        value = exposure * _poisson_deviance(self.truth_rate, rate)
        gradient = 2.0 * exposure * self.design.T @ (
            1.0 - self.truth_rate / rate
        )
        value += self._prior(x, gradient)
        return float(value), gradient

    def hessian(self, x, exposure):
        rate = np.maximum(self.design @ x, 1e-300)
        hessian = 2.0 * exposure * (
            self.design.T
            @ ((self.truth_rate / rate**2)[:, None] * self.design)
        )
        if self.sigma_scale is not None:
            for name, indices in self.norm_groups.items():
                derivative = np.zeros(len(x))
                derivative[indices] = 1.0 / self.norms[indices]
                sigma = SIGMAS[name] * self.sigma_scale
                hessian += 2.0 * np.outer(derivative, derivative) / sigma**2
        return hessian

    def _newton_polish(self, x, allowed, exposure):
        """Damped Newton solve on the current positive face."""
        scale = max(exposure, 1.0)
        value, gradient = self.objective_gradient(x, exposure)
        for _ in range(100):
            threshold = 1e-11 * max(float(x.max()), 1.0)
            face = allowed[x[allowed] > threshold]
            if not len(face):
                break
            g = gradient[face] / scale
            if np.max(np.abs(g)) < 3e-10:
                break
            hessian = self.hessian(x, exposure)[np.ix_(face, face)] / scale
            step = np.linalg.lstsq(hessian, -g, rcond=1e-12)[0]
            directional = float(g @ step)
            if not np.isfinite(directional) or directional >= 0:
                step = -g
                directional = float(g @ step)
            negative = step < 0
            alpha = 1.0
            if np.any(negative):
                alpha = min(
                    alpha,
                    0.995 * float(np.min(-x[face[negative]] / step[negative])),
                )
            accepted = False
            for _ in range(50):
                candidate = x.copy()
                candidate[face] += alpha * step
                candidate = np.maximum(candidate, 0.0)
                candidate_value, candidate_gradient = self.objective_gradient(
                    candidate, exposure
                )
                if candidate_value <= value + 1e-4 * alpha * directional * scale:
                    x, value, gradient = (
                        candidate,
                        candidate_value,
                        candidate_gradient,
                    )
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted or alpha < 1e-12:
                break
        return x

    def fit(self, exposure, finite_difference=False):
        nuisance = np.arange(self.n_dm, self.design.shape[1])
        active_dm = set(int(i) for i in self.initial_dm)
        total_iterations = 0
        x_full = np.zeros(self.design.shape[1])
        for j in nuisance:
            if j not in self.rg_indices:
                x_full[j] = self.norms[j]
        nominal_rg = np.flatnonzero(np.all(self.rg_nodes == (0.0, 0.0), axis=1))
        if len(nominal_rg):
            j = self.rg_indices[int(nominal_rg[0])]
            x_full[j] = self.norms[j]

        for refinement in range(80):
            active = np.array(sorted(active_dm) + list(nuisance), dtype=int)
            x0 = np.maximum(x_full[active], 0.0)

            def fun(z):
                x = np.zeros_like(x_full)
                x[active] = z
                value, gradient = self.objective_gradient(x, exposure)
                scale = max(exposure, 1.0)
                return value / scale, gradient[active] / scale

            result = minimize(
                fun,
                x0,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0.0, None)] * len(active),
                options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10, "maxls": 60},
            )
            total_iterations += int(result.nit)
            x_full[:] = 0.0
            x_full[active] = np.maximum(result.x, 0.0)
            x_full = self._newton_polish(x_full, active, exposure)
            value, gradient = self.objective_gradient(x_full, exposure)

            positive = x_full > 1e-8 * max(x_full.max(), 1.0)
            stationarity = float(
                np.max(np.abs(gradient[positive])) if np.any(positive) else 0.0
            )
            inactive_dm = np.array(
                [j for j in range(self.n_dm) if j not in active_dm], dtype=int
            )
            worst = (
                float(np.min(gradient[inactive_dm])) if len(inactive_dm) else 0.0
            )
            scale = max(exposure, 1.0)
            tolerance = 2e-8 * scale
            if worst >= -tolerance and stationarity <= 1e-6 * scale:
                break

            if worst < -tolerance:
                order = inactive_dm[np.argsort(gradient[inactive_dm])]
                active_dm.update(int(j) for j in order[: min(4, len(order))])
                continue

            # Keep refining this face after the exact-Hessian Newton polish.
        else:
            raise RuntimeError(
                "active-column refinement did not reach KKT conditions: "
                f"worst={worst / scale:.3e}, stationarity={stationarity / scale:.3e}, "
                f"active_dm={len(active_dm)}"
            )

        value, gradient = self.objective_gradient(x_full, exposure)
        bound_violation = np.maximum(-gradient[x_full <= 1e-10], 0.0)
        scale = max(exposure, 1.0)
        kkt_max = (
            float(bound_violation.max() / scale) if len(bound_violation) else 0.0
        )
        positive = x_full > 1e-8 * max(x_full.max(), 1.0)
        stationarity = float(
            np.max(np.abs(gradient[positive])) / scale
            if np.any(positive)
            else 0.0
        )

        fd_error = None
        if finite_difference:
            checks = np.flatnonzero(positive)
            if len(checks):
                x_check = x_full.copy()
                x_check[checks] *= 1.003
                _, check_gradient = self.objective_gradient(x_check, exposure)
                errors = []
                for j in checks[: min(12, len(checks))]:
                    step = 2e-6 * max(1.0, x_check[j])
                    xp, xm = x_check.copy(), x_check.copy()
                    xp[j] += step
                    xm[j] = max(0.0, xm[j] - step)
                    fp = self.objective_gradient(xp, exposure)[0]
                    fm = self.objective_gradient(xm, exposure)[0]
                    fd = (fp - fm) / (xp[j] - xm[j])
                    errors.append(
                        abs(fd - check_gradient[j]) / max(1.0, abs(fd))
                    )
                fd_error = float(max(errors))

        nuisance_coefficients = x_full[self.n_dm :] / self.norms[self.n_dm :]
        rg_coefficients = x_full[self.rg_indices] / self.norms[self.rg_indices]
        rg_total = float(rg_coefficients.sum())
        if rg_total > 0:
            yield_tly, yield_tqy = (
                rg_coefficients @ self.rg_nodes / rg_total
            ).tolist()
        else:
            yield_tly = yield_tqy = 0.0
        triangle = [tuple(v) for v in self.rg_nodes.tolist()] \
            if len(self.rg_nodes) <= 3 else None
        return FitResult(
            deviance=value,
            rate=self.design @ x_full,
            dm_coefficients=x_full[: self.n_dm] / self.norms[: self.n_dm],
            nuisance_coefficients=nuisance_coefficients,
            nuisance_names=self.nuisance_names,
            yield_tly=float(yield_tly),
            yield_tqy=float(yield_tqy),
            triangle=triangle,
            iterations=total_iterations,
            kkt_max=kkt_max,
            active_stationarity=stationarity,
            gradient_fd_error=fd_error,
        )


def _fitters(dm, truth_rate, ac, er, rg, mode):
    common_columns = [ac["sr0"], ac["sr1"], er]
    common_names = ["ac_sr0", "ac_sr1", "er"]
    if mode == "free":
        nodes = [
            (x, y) for x in PROFILE_YIELD_NODES for y in PROFILE_YIELD_NODES
        ]
        names = common_names + [f"rg[{x:+.0f},{y:+.0f}]" for x, y in nodes]
        return [
            ConvexFit(
                dm,
                truth_rate,
                common_columns + [rg[node] for node in nodes],
                names,
                None,
                list(range(3, 3 + len(nodes))),
                nodes,
            )
        ]
    scale = float(mode)
    fitters = []
    for triangle in _triangles():
        names = common_names + [
            f"rg[{x:+.0f},{y:+.0f}]" for x, y in triangle
        ]
        fitters.append(
            ConvexFit(
                dm,
                truth_rate,
                common_columns + [rg[node] for node in triangle],
                names,
                scale,
                [3, 4, 5],
                list(triangle),
            )
        )
    return fitters


def _best_fit(fitters, exposure, finite_difference=False):
    results = [fitter.fit(exposure, finite_difference) for fitter in fitters]
    return min(results, key=lambda result: result.deviance)


def _root(fitters, fixed_exposure):
    cache = {}

    def dev(log_exposure):
        exposure = float(np.exp(log_exposure))
        key = round(log_exposure, 13)
        if key not in cache:
            cache[key] = _best_fit(fitters, exposure)
        return cache[key].deviance - 9.0

    lo, hi = fixed_exposure / 4.0, fixed_exposure * 4.0
    while dev(np.log(lo)) > 0:
        lo /= 4.0
    while dev(np.log(hi)) < 0:
        hi *= 4.0
        if hi > 1e14:
            raise RuntimeError("could not bracket D(exposure)=9")
    exposure = float(
        np.exp(brentq(dev, np.log(lo), np.log(hi), xtol=2e-11, rtol=2e-11))
    )
    result = cache.get(round(np.log(exposure), 13))
    if result is None:
        result = _best_fit(fitters, exposure)
    validation = _best_fit(fitters, exposure, finite_difference=True)
    result.gradient_fd_error = validation.gradient_fd_error
    return exposure, result, len(cache)


def _fixed_fit(dm, truth_rate, nominal, exposure):
    deviance_per_ty, coefficients = poisson_fit_offset(
        dm, truth_rate, nominal
    )
    return exposure * deviance_per_ty, coefficients


def main():
    started = time.perf_counter()
    energies, ly_axis, qy_axis, g0, g1, _b0, _b1 = A.load_mono_grid()
    yields = A.load_yield()
    energy = np.linspace(A.E_ROI[0], A.E_ROI[1], 241)
    response = A.released_response(
        energies,
        ly_axis,
        qy_axis,
        g0,
        g1,
        energy,
        A.yield_curve(yields, "ly", 0.0),
        A.yield_curve(yields, "qy", 0.0),
    )
    spectrum_energy, spectrum, _ = A.spectrum_from_release(
        A.XE_DATA / "signal" / "cevns_spectrum.json"
    )
    phi = A.fold_spectrum(
        energy, np.interp(energy, spectrum_energy, spectrum), energy, response
    )
    ac, er, rg, nominal = _component_vectors()
    truth_rate = phi + nominal

    fixed_reference = json.loads((OUT / "backgrounds.json").read_text())
    reference_rows = {float(row["mass"]): row for row in fixed_reference["rows"]}
    rows = []
    max_fixed_difference = 0.0
    for mass in MASSES:
        dm = generator_matrix(
            energy,
            response,
            mass,
            A.XE_ISOTOPES,
            V_SHELL,
            "isotropic",
            A.V_EARTH,
        )
        reference = reference_rows[mass]["exposure_ty_with_bkg"]
        fixed_deviance_per_ty, _ = poisson_fit_offset(
            dm, truth_rate, nominal
        )
        fixed = 9.0 / fixed_deviance_per_ty
        fixed_difference = abs(fixed / reference - 1.0)
        max_fixed_difference = max(max_fixed_difference, fixed_difference)
        if fixed_difference >= 0.01:
            raise RuntimeError(
                f"fixed fit at {mass:g} GeV differs from backgrounds.json by "
                f"{100 * fixed_difference:.3f}%"
            )

        sensitivity = {}
        fits = {}
        for label, mode in (("profiled", 1.0), ("sigma_x3", 3.0), ("free", "free")):
            fitters = _fitters(dm, truth_rate, ac, er, rg, mode)
            exposure, fit, root_evaluations = _root(fitters, fixed)
            if exposure < fixed * (1.0 - 2e-7):
                raise RuntimeError(
                    f"profiling improved separation at {mass:g} GeV: "
                    f"{exposure} < {fixed}"
                )
            sensitivity[label] = exposure
            fits[label] = fit
            print(
                f"m={mass:4.0f} {label:9s} {exposure:12.5g} t y  "
                f"D={fit.deviance:.9f} KKT={fit.kkt_max:.2e} "
                f"grad={fit.active_stationarity:.2e} roots={root_evaluations}",
                flush=True,
            )

        fit = fits["profiled"]
        theta = {}
        for name in ("ac_sr0", "ac_sr1", "er"):
            i = fit.nuisance_names.index(name)
            theta[name] = float(fit.nuisance_coefficients[i] - 1.0)
        rg_indices = [
            i for i, name in enumerate(fit.nuisance_names) if name.startswith("rg[")
        ]
        theta["rg"] = float(fit.nuisance_coefficients[rg_indices].sum() - 1.0)
        theta_sigma = {
            name: theta[name] / SIGMAS[name] for name in theta
        }
        row = {
            "mass": mass,
            "exposure_ty_fixed": fixed,
            "exposure_ty_profiled": sensitivity["profiled"],
            "exposure_ty_sigma_x3": sensitivity["sigma_x3"],
            "exposure_ty_free": sensitivity["free"],
            "ratio_profiled_over_fixed": sensitivity["profiled"] / fixed,
            "ratio_sigma_x3_over_fixed": sensitivity["sigma_x3"] / fixed,
            "ratio_free_over_fixed": sensitivity["free"] / fixed,
            "theta": theta,
            "theta_in_sigma": theta_sigma,
            "yield": {"t_ly": fit.yield_tly, "t_qy": fit.yield_tqy},
            "active_triangle": fit.triangle,
            "deviance_at_root": fit.deviance,
            "kkt_max": fit.kkt_max,
            "active_stationarity": fit.active_stationarity,
            "gradient_fd_relative_error": fit.gradient_fd_error,
            "free_deviance_at_root": fits["free"].deviance,
            "free_kkt_max": fits["free"].kkt_max,
            "free_gradient_fd_relative_error": fits["free"].gradient_fd_error,
        }
        rows.append(row)

    ratios = [row["ratio_profiled_over_fixed"] for row in rows]
    free_ratios = [row["ratio_free_over_fixed"] for row in rows]
    runtime = time.perf_counter() - started
    output = {
        "source": "XENONnT light-WIMP data release (XENON:2024hup)",
        "sigma_source": SIGMA_SOURCE,
        "yield_source": YIELD_SOURCE,
        "sigmas": SIGMAS,
        "ac_correlation": (
            "Independent SR0 and SR1 nuisance parameters, as defined in the "
            "release YAML. The count-weighted equivalent shared width is "
            "4.8584%, but it is not used."
        ),
        "model": (
            "truth eps*(8B+nominal backgrounds); hypothesis eps*(isotropic "
            "DM cone+profiled backgrounds); Poisson Asimov deviance plus "
            "Gaussian normalization constraints; D(eps)=9 solved in log eps"
        ),
        "interpolation": (
            "Radiogenic templates use simplicial piecewise-linear interpolation "
            "over t_ly,t_qy in [-1,1]. In each triangle, nonnegative node "
            "coefficients are affine expected-count parameters and their sum "
            "is the shared radiogenic normalization. The finite-prior fit is "
            "the minimum over eight convex subproblems."
        ),
        "free_definition": (
            "Conic hull of the independent SR0/SR1 AC columns, the shared ER "
            "column, and all nine combined-run radiogenic nodes in [-1,1]^2. "
            "It drops Gaussian penalties and interpolation constraints, so it "
            "contains every finite-prior model and gives an exposure upper bound."
        ),
        "fixed_validation_max_relative_difference": max_fixed_difference,
        "profiled_ratio_range": [min(ratios), max(ratios)],
        "free_ratio_range": [min(free_ratios), max(free_ratios)],
        "runtime_seconds": runtime,
        "rows": rows,
    }
    path = OUT / "profiled_backgrounds.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"wrote {path} in {runtime:.1f} s", flush=True)


if __name__ == "__main__":
    main()

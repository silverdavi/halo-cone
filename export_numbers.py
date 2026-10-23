#!/usr/bin/env python3
"""Every number in the manuscript, computed once (Rules/LATEX.md, CODE.md).

Reads results/*.json and verification/*.json, derives the analytic edges, and writes
  results/numbers.json      consolidated metrics (figures read this)
  paper/numbers.tex         one macro per number, provenance in a comment
  paper/tables/ideal.tex    Table I   (ideal-detector windows)
  paper/tables/robust.tex   Table II  (robustness of the released-observable window)
"""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

import numpy as np

import analysis as A
from window import mass_for

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
PAPER = ROOT / "paper"
VER = ROOT / "verification"
(PAPER / "tables").mkdir(exist_ok=True)

U_MEV = 16.36                      # end of the Bahcall interpolant (verification/PROOF.txt)
ISOTOPES = {124: 123.9058920, 126: 125.9042983, 128: 127.9035310, 129: 128.9047808611, 130: 129.903509349,
            131: 130.90508406, 132: 131.9041550856, 134: 133.90539466, 136: 135.907214484}
M_E = 0.00051099895


def nuclear_mass_gev(a):
    return ISOTOPES[a] * 0.93149410242 - 54 * M_E


def e_end_kev(a):
    """Recoil end point 2U^2/m_N for neutrino end point U."""
    return 2 * U_MEV**2 / (nuclear_mass_gev(a) * 1e3) * 1e3


def m_hi(E_thr, a=131, vE=A.V_EARTH):
    return mass_for(E_thr, vE, ISOTOPES[a])


def m_lo(a=131, vmax=A.V_MAX):
    return mass_for(e_end_kev(a), vmax, ISOTOPES[a])


def first_mass_below(rows, limit):
    for r in rows:
        if r["exposure_ty_3sigma"] <= limit:
            return r["mass"]
    return None


def at_mass(rows, m):
    return min(rows, key=lambda r: abs(r["mass"] - m))


def nearest(rows, key, value):
    return min(rows, key=lambda r: abs(math.log(r[key] / value)))


def main():
    N = {}
    # --- kinematics
    N.update(vE=A.V_EARTH, vesc=A.V_ESC, vmax=A.V_MAX, vE_min=217.0, vE_max=247.0, U_MeV=U_MEV)
    # --- end points and analytic edges, 131Xe reference
    N["E_end_131"] = e_end_kev(131)
    N["E_b_124"] = e_end_kev(124)
    N["E_end_136"] = e_end_kev(136)
    N["m_lo_131"] = m_lo(131)
    N["m_lo_min"] = min(m_lo(a) for a in ISOTOPES)
    N["m_lo_max"] = max(m_lo(a) for a in ISOTOPES)
    N["E_c_131"] = N["E_end_131"] * (A.V_EARTH / A.V_MAX) ** 2
    N["m_hi_051_131"] = m_hi(0.51)
    N["m_hi_030_131"] = m_hi(0.30)
    N["m_hi_051_124"] = m_hi(0.51, 124)
    N["m_hi_051_136"] = m_hi(0.51, 136)
    N["m_hi_030_136"] = m_hi(0.30, 136)
    N["m_hi_list"] = {str(t): m_hi(t) for t in (0.2, 0.3, 0.4, 0.51, 0.7, 1.0)}
    N["m_hi_051_vEmin"] = m_hi(0.51, 131, 217.0)
    N["m_hi_051_vE218"] = m_hi(0.51, 131, 218.0)
    N["m_hi_051_vE248"] = m_hi(0.51, 131, 248.0)
    N["m_hi_056_131"] = m_hi(0.56)          # bin-shifted (threshold + one 0.05 keV bin)
    N["m_hi_056_136"] = m_hi(0.56, 136)

    # --- convexity certificate (Bahcall table, Arb/rational intervals)
    cert = json.load(open(VER / "certificate.json"))
    central = cert["results"][0]
    N["q_star_MeV"] = central["q_star_MeV"]
    N["E_star_keV"] = central.get("E_star_keV")
    N["mode_MeV"] = central["mode_MeV"]
    N["E_star_max_all_isotopes_keV"] = cert.get("E_star_max_keV", 0.182)

    # --- natural-xenon certificate
    nat = json.load(open(VER / "natural_xenon" / "results.json"))

    def ball(s):
        return float(re.match(r"\[?(-?[0-9.e+-]+)", s).group(1))

    N["nat_m_hi_136"] = ball(nat["upper_mass_each"][-1])
    N["nat_m_hi_124"] = ball(nat["upper_mass_each"][0])
    N["nat_strict_low"] = ball(nat["strict_low_mass"])
    N["nat_m_hi_030_136"] = ball(nat["upper_mass_at_030"])
    N["nat_support_min_35"] = ball(nat["cutoff_support_mass_each"][0])
    N["nat_support_max_35"] = ball(nat["cutoff_support_mass_each"][-1])
    N["nat_cells"] = nat["mass_partition_cells"]
    N["nat_uniform_bound"] = ball(nat["uniform_upper_bound"])
    N["nat_E_b"] = ball(nat["endpoint_keV"])
    cv = nat["curvature"]
    N["nat_p"] = float(cv["p"].strip("[]").split()[0])
    N["nat_r"] = float(cv["r"].strip("[]").split()[0])
    N["nat_margin_b"] = ball(cv["min_2lp_plus_p"])
    N["nat_margin_c"] = ball(cv["max_c"])
    N["nat_Lnu"] = ball(cv["Lnu"])
    cv3 = nat["curvature_at_030"]
    N["nat_p_030"] = float(cv3["p"].strip("[]").split()[0])
    N["nat_r_030"] = float(cv3["r"].strip("[]").split()[0])
    N["nat_Lnu_030"] = ball(cv3["Lnu"])
    N["nat_B2_mass"] = float(nat["B2_counterexample"]["mass_GeV"].strip("[]").split()[0])
    N["nat_B2_breakpoints"] = nat["B2_counterexample"]["breakpoints_checked"]
    N["nat_low_interval"] = nat["low_mass_uniform_dual"]["interval_GeV"]
    # Form-factor variations of the curvature inequalities (not the nominal results.json).
    ff = json.load(open(VER / "natural_xenon" / "form_factor_variations.json"))
    by = {c["name"]: c for c in ff["cases"]}
    N["ff_engel_c"] = ball(by["engel"]["at_0.51"]["max_c"])
    N["ff_engel_c_030"] = ball(by["engel"]["at_0.30"]["max_c"])
    N["ff_lowq_b"] = ball(by["lowq_same_msq"]["at_0.51"]["min_2lp_plus_p"])
    # Re-chosen (p, r): neighborhood ball in form_factor_variations.json, the enclosure the search certified.
    for name, tag, prefix in (
        ("engel", "0.51", "ff_engel_re"),
        ("engel", "0.30", "ff_engel_re30"),
        ("lowq_same_msq", "0.51", "ff_lowq_re"),
    ):
        rec = by[name]["rechosen"][tag]
        N[prefix + "_p"] = rec["p"]
        N[prefix + "_r"] = rec["r"]
        N[prefix + "_b"] = ball(rec["min_2lp_plus_p"])
        N[prefix + "_c"] = ball(rec["max_c"])
        N[prefix + "_L"] = ball(rec["Lnu"])
        N[prefix + "_point_passed"] = rec["at_point"]["flags"]["passed"]
        N[prefix + "_neigh_passed"] = rec["flags"]["passed"]

    # Structural certificate. Floors are strict lower bounds and ceils are strict
    # upper bounds, so the prose inequalities stay inside the Arb balls.
    st = json.load(open(VER / "structural" / "results.json"))

    def arb_bounds(s):
        text = str(s).strip()
        m = re.match(r"\[?\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", text)
        mid = Decimal(m.group(1))
        rm = re.search(r"\+/-\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", text)
        rad = Decimal(rm.group(1)) if rm else Decimal(0)
        return mid - rad, mid + rad

    def qfloor(x, places):
        return format(x.quantize(Decimal(10) ** -places, rounding=ROUND_FLOOR), "f")

    def qceil(x, places):
        return format(x.quantize(Decimal(10) ** -places, rounding=ROUND_CEILING), "f")

    gaps = {}
    for case in st["A_models"]:
        lo, _hi = arb_bounds(case["sufficient_gap"])
        key = case["threshold_keV"]
        gaps[key] = lo if key not in gaps else min(gaps[key], lo)
    N["struct_G_051"] = qfloor(gaps[".51"], 3)
    N["struct_G_030"] = qfloor(gaps[".30"], 3)
    mom = {c["threshold_keV"]: c["maximum_mean_square_radius_fm2"] for c in st["A_moment_only"]}
    N["mom_cap_051"] = str(mom[".51"])
    N["mom_cap_030"] = str(mom[".30"])
    skin = {c["threshold_keV"]: c for c in st["B_shape_independent_skin"]}
    N["skin_max"] = format(Decimal(skin[".51"]["skin_interval_fm"][1]), "f")
    N["skin_dq"] = format(Decimal(skin[".51"]["local_q_width_MeV"]), "f")
    _lo, loss51 = arb_bounds(skin[".51"]["fractional_curvature_loss_upper"])
    _lo, loss30 = arb_bounds(skin[".30"]["fractional_curvature_loss_upper"])
    N["skin_loss_051"] = qceil(loss51, 3)
    N["skin_loss_030"] = qceil(loss30, 3)
    _lo, r2 = arb_bounds(skin[".51"]["maximum_absolute_weak_second_moment_fm2"])
    N["skin_r2"] = qceil(r2, 2)
    full = st["C_full_endpoint_threshold"]
    N["ec_full_lo"] = format(Decimal(full["critical_threshold_bracket_keV"][0]), "f")
    N["ec_full_hi"] = format(Decimal(full["critical_threshold_bracket_keV"][1]), "f")
    N["ec_witness"] = str(full["witness_mass_GeV"])
    iso_hi = max(arb_bounds(v)[1] for v in full["single_isotope_critical_thresholds_keV"])
    N["ec_iso_hi"] = qceil(iso_hi, 3)
    assert Decimal(N["struct_G_051"]) < gaps[".51"]
    assert Decimal(N["struct_G_030"]) < gaps[".30"]
    assert Decimal(N["skin_loss_051"]) > loss51
    assert Decimal(N["skin_loss_030"]) > loss30
    assert Decimal(N["skin_r2"]) > r2
    assert Decimal(N["ec_iso_hi"]) > iso_hi

    # --- ideal detector
    ideal = json.load(open(RES / "ideal.json"))
    N["ideal_dE"] = ideal["dE_keV"]
    N["ideal_thresholds"] = ideal["thresholds"]
    N["ideal_windows"] = {t: ideal["windows"][f"{t}|isotropic"] for t in ideal["thresholds"]}
    N["ideal_windows_S"] = {t: ideal["windows"][f"{t}|streams"] for t in ideal["thresholds"]}
    N["ideal_analytic_m_hi"] = ideal["analytic_m_hi"]
    N["ideal_n_masses"] = len(ideal["masses"])
    N["ideal_mass_range"] = [min(ideal["masses"]), max(ideal["masses"])]
    w051 = ideal["windows"]["0.51|isotropic"]["1e-12"]
    N["ideal_lo_051"], N["ideal_hi_051"] = w051
    N["ideal_lo_051_tol3"], N["ideal_hi_051_tol3"] = ideal["windows"]["0.51|isotropic"]["0.001"]
    # practical lower edge / critical threshold at 1e-3 (where the 1e-3 window first opens)
    opens = [t for t in ideal["thresholds"] if ideal["windows"][f"{t}|isotropic"]["0.001"]]
    N["ideal_tol3_opens_at"] = min(opens)
    closes = [t for t in ideal["thresholds"] if not ideal["windows"][f"{t}|isotropic"]["1e-12"]]
    N["ideal_exact_empty_up_to"] = max(closes)
    N["ideal_exact_open_from"] = min(t for t in ideal["thresholds"] if ideal["windows"][f"{t}|isotropic"]["1e-12"])
    m = np.array(ideal["masses"])
    r051 = np.array(ideal["scans"]["0.51|isotropic"])
    N["ideal_r_one_GeV_above_mhi"] = float(np.interp(N["m_hi_051_131"] + 1.0, m, r051))

    # --- released observable
    win = json.load(open(RES / "window.json"))
    N["rel_window_S"] = win["released_windows_GeV"]["released|streams"]
    N["rel_window_I"] = win["released_windows_GeV"]["released|isotropic"]
    N["rel_tol"] = win["tolerance"]
    N["b8_support_end_1e-3"] = win["b8_support_end_keV"]
    N["m_lo_practical"] = win["analytic_edges_GeV"]["lower_both"]
    N["E_c_practical"] = win["b8_support_end_keV"] * (A.V_EARTH / A.V_MAX) ** 2
    N["livetime_sr0"], N["livetime_sr1"] = A.SR_LIVETIME["sr0"], A.SR_LIVETIME["sr1"]
    N["livetime_total"] = A.SR_LIVETIME["sr0"] + A.SR_LIVETIME["sr1"]
    N["flux_prior"] = A.SOLAR["8B"][1]
    N["b8_flux"] = A.SOLAR["8B"][0]
    disc = json.load(open(RES / "discovery.json"))
    N["shm6_ratio"] = disc["shm_6GeV_rate_check"]["ours"] / disc["shm_6GeV_rate_check"]["release"]
    # 8B rate in the observable (own spectrum through the livetime-weighted response)
    energies, ly_axis, qy_axis, g0, g1, _b0, _b1 = A.load_mono_grid()
    yields = A.load_yield()
    E = np.linspace(A.E_ROI[0], A.E_ROI[1], 241)
    T = A.released_response(energies, ly_axis, qy_axis, g0, g1, E, A.yield_curve(yields, "ly", 0.0),
                            A.yield_curve(yields, "qy", 0.0))
    Es, dR_rel, _ = A.spectrum_from_release(A.XE_DATA / "signal" / "cevns_spectrum.json")
    N["b8_per_ty_observable"] = float(A.fold_spectrum(E, np.interp(E, Es, dR_rel), E, T).sum())
    dR_own, _ = A.compound_spectrum(Es[(Es >= 0.5) & (Es <= 3.0)], A.XE_ISOTOPES, "8B")
    sel = (Es >= 0.5) & (Es <= 3.0)
    N["own_vs_release_max_dev"] = float(np.max(np.abs(dR_own / dR_rel[sel] - 1)))

    # --- robustness
    rob = json.load(open(RES / "robustness.json"))
    N["rob"] = {k: v for k, v in rob.items()}

    # --- separation exposure
    sep_I = win["exposure_released"]["isotropic"]
    sep_S = win["exposure_released"]["streams"]
    above = [r for r in sep_I if r["mass"] > N["rel_window_I"][0]]
    lm = np.log([r["mass"] for r in above])
    le = np.log([min(r["exposure_ty_3sigma"], 1e12) for r in above])
    N["sep_I_at"] = {str(mm): float(np.exp(np.interp(np.log(mm), lm, le))) for mm in (13, 15, 16, 18, 20)}
    hi = [r["exposure_ty_3sigma"] for r in sep_I if 25 <= r["mass"] <= 40]
    N["sep_I_25_40"] = [min(hi), max(hi)]
    # exposure is a steep function above the window: mass at which it falls below 1e6 and 1e7 t y
    N["sep_I_above_1e6_below"] = float(np.exp(np.interp(np.log(1e6), le[::-1], lm[::-1])))
    N["sep_I_above_1e7_below"] = float(np.exp(np.interp(np.log(1e7), le[::-1], lm[::-1])))
    N["sep_mass_reached"] = {str(x): float(np.exp(np.interp(np.log(x), le[::-1], lm[::-1])))
                             for x in (N["livetime_total"], 20.0, 1000.0)}
    N["sep_S_infinite_above"] = min(r["mass"] for r in sep_S if r["exposure_ty_3sigma"] >= 1e7)

    # --- discovery limit
    res = disc["results"]
    eps_grid = [r["exposure_ty"] for r in res["14"]]
    e3, e1e3, e1e4, e3e4 = (nearest(res["14"], "exposure_ty", x)["exposure_ty"] for x in (3.0, 1e3, 1e4, 3e4))
    N["disc_eps_low"], N["disc_eps_1e3"], N["disc_eps_1e4"], N["disc_eps_max"] = e3, e1e3, e1e4, e3e4

    def ratio(mm, eps):
        r = nearest(res[mm], "exposure_ty", eps)
        return r["sigma_halo_independent"] / r["sigma_shm"]

    N["disc_ratio"] = {mm: {f"{eps:g}": ratio(mm, eps) for eps in (e3, e1e3, e1e4)} for mm in ("8", "10", "12", "14", "16", "20", "30", "50")}
    N["disc_n14_1e3"] = nearest(res["14"], "exposure_ty", e1e3)["n_sigma_halo_independent"]
    N["disc_n14_1e4"] = nearest(res["14"], "exposure_ty", e1e4)["n_sigma_halo_independent"]
    n16 = [(r["exposure_ty"], r.get("n_sigma_halo_independent", 0)) for r in res["16"] if "n_sigma_halo_independent" in r]
    N["disc_n16_peak"] = max(n16, key=lambda t: t[1])
    N["disc_n16_1e4"] = nearest(res["16"], "exposure_ty", e1e4)["n_sigma_halo_independent"]
    N["disc_sat_12"] = res["12"][-1]["sigma_halo_independent"]
    N["disc_sat_14"] = res["14"][-1]["sigma_halo_independent"]
    N["disc_m_infinite"] = "6"
    N["disc_ratio_8_range"] = [min(ratio("8", e) for e in eps_grid[2:]), max(ratio("8", e) for e in eps_grid[2:])]
    N["disc_ratio_20_range"] = [min(ratio("20", e) for e in eps_grid[2:]), max(ratio("20", e) for e in eps_grid[2:])]
    hi_ratio = [ratio(mm, e) for mm in ("30", "50") for e in eps_grid[2:]]
    N["disc_ratio_30_50_range"] = [min(hi_ratio), max(hi_ratio)]
    N["disc_ratio_12_14_low"] = [min(ratio(mm, e3) for mm in ("12", "14")), max(ratio(mm, e3) for mm in ("12", "14"))]
    N["disc_ratio_12_14_1e4"] = [min(ratio(mm, e1e4) for mm in ("12", "14")), max(ratio(mm, e1e4) for mm in ("12", "14"))]
    N["disc_ratio_min_max_low"] = [min(ratio(mm, e3) for mm in ("16", "20", "30", "50")), ratio("8", e3)]
    # f* diagnostic. Rows are parallel to results[mass].
    fst = disc["fstar"]["by_mass"]

    def diag_at(mass, eps):
        rows, diags = res[mass], fst[mass]["rows"]
        i = min(range(len(rows)), key=lambda j: abs(math.log(rows[j]["exposure_ty"] / eps)))
        return diags[i]

    def dominant(d):
        return max(d["support"], key=lambda s: s["weight"])

    d8 = diag_at("8", e3)
    d8_hi = diag_at("8", e1e4)
    d12 = diag_at("12", e1e4)
    d14 = diag_at("14", e1e4)
    d30 = diag_at("30", e3)
    dom8, dom12, dom14 = dominant(d8), dominant(d12), dominant(d14)
    rest12 = [s for s in d12["support"] if s["index"] != dom12["index"]]
    wfast = sum(s["weight"] for s in rest12)
    ufast = sum(s["weight"] * s["speed_kms"] for s in rest12) / wfast
    N["disc_u8"] = dom8["speed_kms"]
    N["disc_w8_min"] = min(dominant(d)["weight"] for d in fst["8"]["rows"])
    N["disc_rate8"] = d8["counts_per_ty_per_cm2"] / fst["8"]["shape"]["shm_counts_per_ty_per_cm2"]
    N["disc_tv8"] = d8["shape_tv"]
    N["disc_u12"] = dom12["speed_kms"]
    N["disc_w12"] = dom12["weight"]
    N["disc_u12_fast"] = ufast
    N["disc_tv12"] = d12["shape_tv"]
    N["disc_u14"] = dom14["speed_kms"]
    N["disc_w14"] = dom14["weight"]
    N["disc_tv14"] = d14["shape_tv"]
    N["disc_tv30"] = d30["shape_tv"]
    N["disc_w8_at_1e4"] = dominant(d8_hi)["weight"]

    # --- second target
    comp = json.load(open(RES / "complement.json"))
    N["cawo4_b8_per_ty"] = comp["cawo4_8b_per_ty"]
    inside = {}
    for k, v in comp["results"].items():
        if not k.endswith("xenon_20ty") or not v["cawo4_untagged_ty_for_deviance_9"]:
            continue
        if N["rel_window_I"][0] <= v["mass"] <= N["rel_window_I"][1]:
            inside.setdefault(v["halo_class"], []).append(v["cawo4_untagged_ty_for_deviance_9"])
    N["comp_S_range"] = [min(inside["streams"]), max(inside["streams"])]
    N["comp_I_range"] = [min(inside["isotropic"]), max(inside["isotropic"])]
    pairs = {}
    for k, v in comp["results"].items():
        if k.endswith("xenon_20ty") and v["cawo4_untagged_ty_for_deviance_9"] and N["rel_window_I"][0] <= v["mass"] <= N["rel_window_I"][1]:
            pairs.setdefault(v["mass"], {})[v["halo_class"]] = v["cawo4_untagged_ty_for_deviance_9"]
    f = [p["streams"] / p["isotropic"] for p in pairs.values() if len(p) == 2]
    N["comp_factor_range"] = [min(f), max(f)]
    N["comp_masses_inside"] = sorted(pairs)

    (RES / "numbers.json").write_text(json.dumps(N, indent=1, default=float))
    write_macros(N)
    write_tables(N)
    print("wrote results/numbers.json, paper/numbers.tex, paper/tables/*.tex")


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def sig(x, n=2):
    return f"{x:.{n}g}"


def sci(x, digits=1):
    m, e = f"{x:.{digits}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def b8_tail_over_peak(e0_kev):
    """nu(E0)/nu(0) for the natural-xenon 8B spectrum, with the inputs of
    verification/structural/certify.py (Bahcall table, Helm s = 0.9 fm, exact masses),
    in Arb. nu decreases on [0, E_b], so this is its maximum on [E0, E_b] over its peak."""
    from flint import arb, ctx
    ctx.prec = 128
    rows = [[arb(v) for v in r.split(",")] for r in (VER / "b8_spectrum.csv").read_text().split()]
    xs, ys = [r[0] for r in rows], [r[1] for r in rows]
    U = arb("16.36")
    keep = [i for i in range(len(xs)) if xs[i] < U]
    xs, ys = [xs[i] for i in keep] + [U], [ys[i] for i in keep] + [arb(0)]

    def psi(q):
        """Integral of phi(u) (1 - q^2/u^2) over [q, U], phi linear between nodes."""
        total = arb(0)
        for a, b, ya, yb in zip(xs, xs[1:], ys, ys[1:]):
            if b <= q:
                continue
            lo = q if a < q else a
            s = (yb - ya) / (b - a)
            c = ya - s * a
            total += c * (b - lo) + s * (b * b - lo * lo) / 2
            if q > 0:
                total -= q * q * (s * (b / lo).log() + c * (1 / lo - 1 / b))
        return total

    norm = psi(arb(0))
    fr = {124: ".00095", 126: ".00089", 128: ".0191", 129: ".264", 130: ".0407",
          131: ".212", 132: ".269", 134: ".104", 136: ".0886"}
    hbarc, skin = arb(".1973269804"), arb(".9")
    e0 = arb(e0_kev)

    def nu(E):
        total = arb(0)
        for a, f in fr.items():
            m = arb(str(ISOTOPES[a])) * arb(".93149410242") - 54 * arb(".00051099895")
            qw = arb(a - 54) - (1 - 4 * arb(".23857")) * 54
            weight = arb(f) * m * qw**2
            if E == 0:
                total += weight * norm
                continue
            R = ((arb("1.23") * arb(a).root(3) - arb(".60"))**2
                 + 7 * arb.pi()**2 * arb(".52")**2 / 3 - 5 * skin**2).sqrt()
            p = (2 * m * E * arb("1e-6")).sqrt() / hbarc
            z = p * R
            helm = (3 * (z.sin() - z * z.cos()) / z**3)**2 * (-(p * skin)**2).exp()
            total += weight * helm * psi((m * E / 2).sqrt())
        return total

    ratio = nu(e0) / nu(0)
    return float(ratio.mid())


def write_macros(N):
    L = ["% numbers.tex: generated by export_numbers.py from results/*.json. Do not edit by hand.",
         r"\newcommand{\PH}[1]{\textcolor{red!70!black}{[#1]}}", ""]

    def mac(name, value, src):
        L.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {src}")

    mac("nVE", f"{N['vE']:.0f}", "analysis.V_EARTH")
    mac("nVesc", f"{N['vesc']:.0f}", "analysis.V_ESC")
    mac("nVmax", f"{N['vmax']:.0f}", "v_esc + v_E")
    mac("nVEmin", f"{N['vE_min']:.0f}", "annual minimum of the Earth speed")
    mac("nVEmax", f"{N['vE_max']:.0f}", "annual maximum of the Earth speed")
    mac("nUMeV", f"{N['U_MeV']:.2f}", "end of the Bahcall interpolant; verification/PROOF.txt")
    mac("nEend", f2(N["E_end_131"]), "2U^2/m_N for 131Xe (keV)")
    mac("nEb", f2(N["E_b_124"]) if False else f"{N['E_b_124']:.3f}", "2U^2/m_N for 124Xe (keV)")
    mac("nEendMin", f2(N["E_end_136"]), "2U^2/m_N for 136Xe (keV)")
    mac("nMlo", f1(N["m_lo_131"]), "mass_for(E_end_131, v_max)")
    mac("nMloMin", f2(N["m_lo_min"]), "min over isotopes")
    mac("nMloMax", f2(N["m_lo_max"]), "max over isotopes")
    mac("nEc", f2(N["E_c_131"]), "E_end (v_E/v_max)^2, 131Xe")
    mac("nEcRound", f1(N["E_c_131"]), "rounded E_c")
    mac("nMhi", f1(N["m_hi_051_131"]), "mass_for(0.51 keV, v_E), 131Xe")
    mac("nMhiTwo", f2(N["m_hi_051_131"]), "same, two decimals")
    mac("nMhiThirty", f1(N["m_hi_030_131"]), "mass_for(0.30 keV, v_E), 131Xe")
    mac("nMhiLight", f2(N["m_hi_051_124"]), "124Xe")
    mac("nMhiHeavy", f2(N["m_hi_051_136"]), "136Xe")
    mac("nMhiHeavyOne", f1(N["m_hi_051_136"]), "136Xe, one decimal")
    mac("nMhiHeavyThirty", f"{N['m_hi_030_136']:.3f}", "136Xe at 0.30 keV")
    mac("nMhiList", ",\\,".join(f1(v) for v in N["m_hi_list"].values()), "m_hi at 0.2,0.3,0.4,0.51,0.7,1.0 keV")
    mac("nMhiVEmin", f1(N["m_hi_051_vEmin"]), "m_hi(0.51) at v_E = 217")
    mac("nMhiVEa", f1(N["m_hi_051_vE218"]), "m_hi(0.51) at v_E = 218")
    mac("nMhiVEb", f1(N["m_hi_051_vE248"]), "m_hi(0.51) at v_E = 248")
    mac("nMhiBinShift", f2(N["m_hi_056_131"]), "m_hi(0.56), 131Xe")
    mac("nMhiBinShiftHeavy", f2(N["m_hi_056_136"]), "m_hi(0.56), 136Xe")
    mac("nQstarLo", N["q_star_MeV"][0], "verification/certificate.json")
    mac("nQstarHi", N["q_star_MeV"][1], "verification/certificate.json")
    mac("nQstarShort", N["q_star_MeV"][0][:6], "verification/certificate.json, truncated")
    mac("nEstarLo", N["E_star_keV"][0], "verification/certificate.json")
    mac("nEstarHi", N["E_star_keV"][1], "verification/certificate.json")
    mac("nEstarShort", N["E_star_keV"][0][:5], "verification/certificate.json, truncated")
    mac("nEstarMax", f"{N['E_star_max_all_isotopes_keV']:.3f}", "verification/PROOF.txt, all isotopes and +-3 sigma columns")
    from fractions import Fraction
    mac("nModeMeV", f"{float(Fraction(N['mode_MeV'])):.2f}", "mode of the Bahcall spectrum")
    mac("nNatMhi", f"{N['nat_m_hi_136']:.3f}", "verification/natural_xenon: m_hi(136Xe)")
    mac("nNatMhiLong", f"{N['nat_m_hi_136']:.4f}", "same, four decimals")
    mac("nNatStrictLow", f"{N['nat_strict_low']:.3f}", "mass for 124Xe to reach E_b")
    mac("nNatMhiThirty", f"{N['nat_m_hi_030_136']:.3f}", "m_hi(136Xe, 0.30 keV)")
    mac("nNatSupportMin", f2(N["nat_support_min_35"]), "support bound at 3.5 keV, 124Xe")
    mac("nNatBtwoMass", f"{N['nat_B2_mass']:.4f}", "B2 counterexample mass")
    mac("nNatCells", str(N["nat_cells"]), "mass cells in the five-term certificate")
    mac("nNatCellLo", "6.65", "partition start in verification/natural_xenon/certify.py (covers all masses reaching E_b)")
    mac("nNatCellHi", "7.80", "partition end, above m_hi(136Xe)")
    mac("nNatP", f"{N['nat_p']:.3f}", "weighted operator p at 0.51 keV")
    mac("nNatR", f"{N['nat_r']:.3f}", "weighted operator r at 0.51 keV")
    mac("nNatPthirty", f"{N['nat_p_030']:.3f}", "p at 0.30 keV")
    mac("nNatRthirty", f"{N['nat_r_030']:.3f}", "r at 0.30 keV")
    mac("nNatMarginB", sci(N["nat_margin_b"]), "min_i (2 l_i' + p)")
    mac("nNatMarginC", sci(-N["nat_margin_c"]), "-max_i (l_i''+l_i'^2+p l_i'+r)")
    mac("nNatLnu", sci(N["nat_Lnu"]), "L applied to the target at 0.51 keV")
    mac("nFFSkin", "0.9", "Helm skin s (fm) in verification/natural_xenon/certify.py; Lewin & Smith")
    mac("nFFSkinStep", "0.1", "difference between certificate skin 0.9 fm and Engel skin 1 fm")
    mac("nFFEngelSkin", "1", "Engel skin (fm) quoted by Lewin & Smith; form_factor_variations.json")
    mac("nFFSkinLow", "0.8", "certificate skin minus 0.1 fm; form_factor_variations.json helm_skin_0.8")
    mac("nFFEngelRms", "0.93", "Engel r_rms/A^{1/3} (fm); form_factor_variations.json engel")
    mac("nFFEngelC", sci(N["ff_engel_c"], 2), "max of l''+l'^2+p l'+r, Engel Helm, 0.51 keV")
    mac("nFFEngelCthirty", sci(N["ff_engel_c_030"], 2), "same maximum at 0.30 keV")
    mac("nFFLowqB", sci(N["ff_lowq_b"], 2), "min of 2 l'+p, F^2=1-q^2<r^2>/3, 0.51 keV")
    mac("nFFDelta", sci(1e-6), "delta in rechoose(), verification/natural_xenon/certify.py")
    mac("nFFEngelP", N["ff_engel_re_p"], "re-chosen p, Engel, 0.51 keV; form_factor_variations.json")
    mac("nFFEngelR", N["ff_engel_re_r"], "re-chosen r, Engel, 0.51 keV")
    mac("nFFEngelMB", sci(N["ff_engel_re_b"], 2), "min(2 l'+p), Engel re-chosen, 0.51 keV neighborhood")
    mac("nFFEngelMC", sci(-N["ff_engel_re_c"], 2), "-max(c), Engel re-chosen, 0.51 keV neighborhood")
    mac("nFFEngelLnu", sci(N["ff_engel_re_L"]), "L nu, Engel re-chosen, 0.51 keV neighborhood")
    mac("nFFEngelPthirty", N["ff_engel_re30_p"], "re-chosen p, Engel, 0.30 keV")
    mac("nFFEngelRthirty", N["ff_engel_re30_r"], "re-chosen r, Engel, 0.30 keV")
    mac("nFFEngelMBthirty", sci(N["ff_engel_re30_b"], 2), "min(2 l'+p), Engel re-chosen, 0.30 keV")
    mac("nFFEngelMCthirty", sci(-N["ff_engel_re30_c"], 2), "-max(c), Engel re-chosen, 0.30 keV")
    mac("nFFEngelLnuThirty", sci(N["ff_engel_re30_L"]), "L nu, Engel re-chosen, 0.30 keV")
    mac("nFFLowqP", N["ff_lowq_re_p"], "re-chosen p, low-q, 0.51 keV")
    mac("nFFLowqR", N["ff_lowq_re_r"], "re-chosen r, low-q, 0.51 keV")
    mac("nFFLowqMB", sci(N["ff_lowq_re_b"], 2), "min(2 l'+p), low-q re-chosen, 0.51 keV")
    mac("nFFLowqMC", sci(-N["ff_lowq_re_c"], 2), "-max(c), low-q re-chosen, 0.51 keV")
    mac("nFFLowqLnu", sci(N["ff_lowq_re_L"]), "L nu, low-q re-chosen, 0.51 keV")
    mac("nStructGlo", N["struct_G_051"], "floor of the smallest closed-form gap G at 0.51 keV; verification/structural")
    mac("nStructGloThirty", N["struct_G_030"], "floor of the smallest closed-form gap G at 0.30 keV; verification/structural")
    mac("nMomCap", N["mom_cap_051"], "sufficient mean-square-radius cap at 0.51 keV (fm^2); verification/structural")
    mac("nMomCapThirty", N["mom_cap_030"], "sufficient mean-square-radius cap at 0.30 keV (fm^2); verification/structural")
    mac("nSkinMax", N["skin_max"], "neutron-skin hypothesis upper end (fm); verification/structural")
    mac("nSkinDq", N["skin_dq"], "local momentum width of the shape-independent skin test (MeV)")
    mac("nSkinLoss", N["skin_loss_051"], "ceil of the fractional curvature loss at 0.51 keV; prose uses below this")
    mac("nSkinLossThirty", N["skin_loss_030"], "ceil of the fractional curvature loss at 0.30 keV")
    mac("nSkinRtwo", N["skin_r2"], "ceil of the largest absolute weak mean-square radius (fm^2)")
    mac("nEcFullLo", N["ec_full_lo"], "full-endpoint critical-threshold bracket, lower end (keV)")
    mac("nEcFullHi", N["ec_full_hi"], "full-endpoint critical-threshold bracket, upper end (keV)")
    mac("nEcWitness", N["ec_witness"], "witness mass for the full-endpoint member at nEcFullHi (GeV)")
    mac("nEcIsoHi", N["ec_iso_hi"], "ceil of the nine single-isotope critical thresholds (keV)")
    mac("nNatLowA", N["nat_low_interval"][0], "low-mass functional interval, lower edge (GeV)")
    mac("nNatLowB", N["nat_low_interval"][1], "low-mass functional interval, upper edge (GeV)")
    mac("nSinW", f"{A.SIN2_THETA_W:.4f}", "sin^2 theta_W in analysis.cevns_spectrum; Q_W = N-(1-4 sin^2 theta_W)Z")
    mac("nCresstThr", f"{A.CRESST_THRESHOLD:.4f}", "CaWO4 threshold (keV), analysis.CRESST_THRESHOLD")
    mac("nCresstEmax", f"{A.CRESST_EMAX:.0f}", "CaWO4 ROI upper edge (keV), analysis.CRESST_EMAX")
    mac("nCresstBins", str(25 - 1), "complement.py geomspace(threshold, emax, 25) has 24 bins")
    mac("nCresstRes", f"{A.CRESST_SIGMA_EV:.1f}", "CRESST-III baseline resolution (eV), analysis.CRESST_SIGMA_EV; not smeared again")
    mac("nNatEb", f"{N['nat_E_b']:.3f}", "124Xe end point (keV)")
    mac("nNatEendMin", f2(N["E_end_136"]), "136Xe end point (keV)")
    mac("nIdealDE", f2(N["ideal_dE"]), "ideal bin width (keV)")
    mac("nIdealNmass", str(N["ideal_n_masses"]), "ideal mass grid")
    mac("nIdealMmin", f"{N['ideal_mass_range'][0]:.0f}", "")
    mac("nIdealMmax", f"{N['ideal_mass_range'][1]:.0f}", "")
    mac("nIdealLo", f2(N["ideal_lo_051"]), "numerical lower edge, 0.51 keV, 1e-12")
    mac("nIdealHi", f2(N["ideal_hi_051"]), "numerical upper edge, 0.51 keV, 1e-12")
    mac("nIdealLoTol", f2(N["ideal_lo_051_tol3"]), "numerical lower edge, 0.51 keV, 1e-3")
    mac("nIdealHiTol", f2(N["ideal_hi_051_tol3"]), "numerical upper edge, 0.51 keV, 1e-3")
    mac("nIdealEmptyUpTo", f2(N["ideal_exact_empty_up_to"]), "largest threshold with empty 1e-12 window")
    mac("nIdealOpenFrom", f2(N["ideal_exact_open_from"]), "smallest threshold with non-empty 1e-12 window")
    mac("nIdealRabove", sci(N["ideal_r_one_GeV_above_mhi"]), "residual 1 GeV above m_hi")
    mac("nRelLoS", f1(N["rel_window_S"][0]), "released window, class S, lower")
    mac("nRelLoI", f2(N["rel_window_I"][0]), "released window, class I, lower")
    mac("nRelLoIone", f1(N["rel_window_I"][0]), "released window, class I, lower, one decimal")
    mac("nRelHiI", f1(N["rel_window_I"][1]), "released window, class I, upper")
    mac("nBsupport", f1(N["b8_support_end_1e-3"]), "energy above which 1e-3 of 8B events lie (keV)")
    mac("nMloPractical", f1(N["m_lo_practical"]), "mass_for(2.8 keV, v_max)")
    mac("nEcPractical", f2(N["E_c_practical"]), "2.8 keV (v_E/v_max)^2")
    mac("nLiveA", f2(N["livetime_sr0"]), "SR0 livetime (t y)")
    mac("nLiveB", f2(N["livetime_sr1"]), "SR1 livetime (t y)")
    mac("nLive", f1(N["livetime_total"]), "SR0+SR1 livetime (t y)")
    mac("nFluxPrior", f"{100 * N['flux_prior']:.2f}", "8B flux prior (percent)")
    mac("nFluxPriorRound", f"{100 * N['flux_prior']:.1f}", "8B flux prior, rounded")
    mac("nFluxPriorDelta", f"{N['flux_prior']:.4f}", "8B flux prior, fraction")
    mac("nBflux", r"5.25\times10^{6}", "SNO 8B flux cm^-2 s^-1, analysis.SOLAR")
    mac("nBrate", f1(N["b8_per_ty_observable"]), "8B per t y in [0.51,5] keV through the response")
    mac("nShmDev", f"{100 * abs(N['shm6_ratio'] - 1):.0f}", "6 GeV SHM rate vs release (percent)")
    mac("nOwnDev", f"{100 * N['own_vs_release_max_dev']:.1f}", "own vs released 8B spectrum, 0.5-3 keV (percent)")
    rob = N["rob"]
    mac("nRobVEa", f1(rob["vE=218 vesc=544 | isotropic"][1]), "robustness.json")
    mac("nRobVEb", f1(rob["vE=248 vesc=544 | isotropic"][1]), "robustness.json")
    mac("nRobMhiVEa", f1(rob["vE=218 vesc=544 | analytic m_hi(0.51 keV)"]), "robustness.json")
    mac("nRobMhiVEb", f1(rob["vE=248 vesc=544 | analytic m_hi(0.51 keV)"]), "robustness.json")
    mac("nSepMillionBelow", f1(N["sep_I_above_1e6_below"]), "isotropic separation exposure exceeds 1e6 t y below this mass")
    mac("nSepTenMillionBelow", f1(N["sep_I_above_1e7_below"]), "exceeds 1e7 t y below this mass")
    for mm, name in zip((13, 15, 16, 18, 20), ("Thirteen", "Fifteen", "Sixteen", "Eighteen", "Twenty")):
        v = N["sep_I_at"][str(mm)]
        mac(f"nSep{name}", sci(v) if v >= 1e4 else (f"{v:.0f}" if v >= 20 else f1(v)), f"separation exposure at {mm} GeV")
    mac("nSepHighLo", f1(N["sep_I_25_40"][0]), "separation exposure, 25-40 GeV, min")
    mac("nSepHighHi", f1(N["sep_I_25_40"][1]), "separation exposure, 25-40 GeV, max")
    rk = list(N["sep_mass_reached"])
    mac("nSepMassNow", f"{N['sep_mass_reached'][rk[0]]:.0f}", "mass reached at the SR0+SR1 exposure")
    mac("nSepMassTwenty", f"{N['sep_mass_reached'][rk[1]]:.0f}", "mass reached at 20 t y")
    mac("nSepMassThousand", f"{N['sep_mass_reached'][rk[2]]:.0f}", "mass reached at 1000 t y")
    mac("nSepSabove", f1(N["sep_S_infinite_above"]), "class S: infinite above this mass")
    mac("nDiscEpsLow", f"{N['disc_eps_low']:.0f}", "lowest quoted exposure (t y)")
    mac("nDiscEpsMax", sci(N["disc_eps_max"], 0).replace(".0", ""), "largest exposure in the scan")
    r = N["disc_ratio"]
    e3, e1e3, e1e4 = (f"{x:g}" for x in (N["disc_eps_low"], N["disc_eps_1e3"], N["disc_eps_1e4"]))
    mac("nDiscRatioMinLow", f1(N["disc_ratio_min_max_low"][0]), "smallest HI/SHM ratio above the window at 3 t y")
    mac("nDiscRatioMaxLow", f"{N['disc_ratio_min_max_low'][1]:.0f}", "HI/SHM ratio at 8 GeV, 3 t y")
    mac("nDiscRatioMidLo", f"{N['disc_ratio_12_14_low'][0]:.0f}", "12-14 GeV at 3 t y")
    mac("nDiscRatioMidHi", f"{N['disc_ratio_12_14_low'][1]:.0f}", "12-14 GeV at 3 t y")
    mac("nDiscRatioMidLoHigh", f"{N['disc_ratio_12_14_1e4'][0]:.0f}", "12-14 GeV at 1e4 t y")
    mac("nDiscRatioMidHiHigh", f"{N['disc_ratio_12_14_1e4'][1]:.0f}", "12-14 GeV at 1e4 t y")
    mac("nDiscNfourteenK", f"{N['disc_n14_1e3']:.0f}", "fog index, 14 GeV, 1e3 t y")
    mac("nDiscNfourteenTenK", f"{N['disc_n14_1e4']:.0f}", "fog index, 14 GeV, 1e4 t y")
    mac("nDiscNsixteenPeak", f"{N['disc_n16_peak'][1]:.0f}", "fog index peak, 16 GeV")
    mac("nDiscNsixteenPeakEps", f"{N['disc_n16_peak'][0]:.0f}", "exposure of that peak (t y)")
    mac("nDiscEpsTenK", f"{N['disc_eps_1e4']:.0f}", "exposure grid point near 1e4 t y")
    mac("nDiscEpsK", f"{N['disc_eps_1e3']:.0f}", "exposure grid point near 1e3 t y")
    mac("nDiscNsixteenTenK", f"{N['disc_n16_1e4']:.0f}", "fog index, 16 GeV, 1e4 t y")
    mac("nDiscSatLo", sci(min(N["disc_sat_12"], N["disc_sat_14"])), "saturation sigma, 12-14 GeV")
    mac("nDiscSatHi", sci(max(N["disc_sat_12"], N["disc_sat_14"])), "saturation sigma, 12-14 GeV")
    mac("nDiscRatioTwentyLo", f1(N["disc_ratio_20_range"][0]), "20 GeV ratio range")
    mac("nDiscRatioTwentyHi", f1(N["disc_ratio_20_range"][1]), "")
    mac("nDiscRatioHighLo", f1(N["disc_ratio_30_50_range"][0]), "30-50 GeV ratio range")
    mac("nDiscRatioHighHi", f1(N["disc_ratio_30_50_range"][1]), "")
    mac("nDiscRatioEightLo", f"{N['disc_ratio_8_range'][0]:.0f}", "8 GeV ratio range over exposures")
    mac("nDiscRatioEightHi", f"{N['disc_ratio_8_range'][1]:.0f}", "")
    mac("nDiscUgrid", "3", "first positive shell speed (km/s), discovery.py geomspace(3, v_esc, 90)")
    mac("nDiscUEight", f"{N['disc_u8']:.2f}", "dominant shell speed at 8 GeV, 3 t y; discovery.json fstar")
    mac("nDiscWEight", f"{100 * N['disc_w8_min']:.0f}", "minimum dominant-shell weight at 8 GeV, percent, over the exposure grid")
    mac("nDiscRateEight", sci(N["disc_rate8"], 2), "8 GeV f* rate over SHM rate at 3 t y")
    mac("nDiscTvEight", f"{100 * N['disc_tv8']:.0f}", "8 GeV shape total variation vs 8B, percent of counts, at 3 t y")
    mac("nDiscUTwelve", f"{N['disc_u12']:.2f}", "dominant shell speed at 12 GeV, 1e4 t y")
    mac("nDiscWTwelve", f"{100 * N['disc_w12']:.0f}", "dominant weight at 12 GeV, 1e4 t y, percent")
    mac("nDiscUTwelveFast", f"{N['disc_u12_fast']:.0f}", "weight-averaged speed of the remaining 12 GeV shells at 1e4 t y")
    mac("nDiscTvTwelve", f"{100 * N['disc_tv12']:.1f}", "12 GeV f* shape total variation vs 8B, percent, at 1e4 t y")
    mac("nDiscUFourteen", f"{N['disc_u14']:.2f}", "dominant shell speed at 14 GeV, 1e4 t y")
    mac("nDiscWFourteen", f"{100 * N['disc_w14']:.1f}", "dominant weight at 14 GeV, 1e4 t y, percent")
    mac("nDiscTvFourteen", f"{100 * N['disc_tv14']:.1f}", "14 GeV f* shape total variation vs 8B, percent, at 1e4 t y")
    mac("nDiscTvThirty", f"{100 * N['disc_tv30']:.0f}", "30 GeV escape-shell shape total variation vs 8B, percent of counts")
    mac("nCawoRate", f"{N['cawo4_b8_per_ty']:.0f}", "8B per t y in CaWO4 (CRESST-III efficiencies)")
    mac("nCompSlo", f1(N["comp_S_range"][0]), "CaWO4 exposure, class S, min over window masses")
    mac("nCompShi", f1(N["comp_S_range"][1]), "")
    mac("nCompIlo", f2(N["comp_I_range"][0]), "CaWO4 exposure, class I")
    mac("nCompIhi", f2(N["comp_I_range"][1]), "")
    mac("nCompFactorLo", f"{N['comp_factor_range'][0]:.0f}", "ratio S/I")
    mac("nCompFactorHi", f"{N['comp_factor_range'][1]:.0f}", "")
    # --- BEGIN backgrounds (backgrounds.py -> results/backgrounds.json; self-contained block)
    bkg = json.load(open(RES / "backgrounds.json"))
    bkg_at = {int(r["mass"]): r for r in bkg["rows"]}
    mac("nBkgRate", f1(bkg["background_per_ty"]),
        "AC+ER+radiogenic NR per combined t y in the ROI; XENONnT release templates (XENON:2024hup), livetime-weighted SR0+SR1")
    mac("nBkgCounts", f1(bkg["expected_counts_total"]), "same backgrounds, expected events in the SR0+SR1 exposure")
    mac("nBkgFactorLo", f1(bkg["ratio_range"][0]), "min over 13-25 GeV of (exposure with bkg)/(8B-only exposure), isotropic class")
    mac("nBkgFactorHi", f1(bkg["ratio_range"][1]), "max of the same ratio")
    mac("nBkgSepTwenty", f1(bkg_at[20]["exposure_ty_with_bkg"]), "separation exposure at 20 GeV with fixed backgrounds (t y)")
    mac("nBkgSepSixteen", f"{bkg_at[16]['exposure_ty_with_bkg']:.0f}", "separation exposure at 16 GeV with fixed backgrounds (t y)")
    # --- END backgrounds
    # --- BEGIN natural-xenon subsection (sec:isotopes): membership proposition; self-contained block
    st_full = json.load(open(VER / "structural" / "results.json"))["C_full_endpoint_threshold"]
    mac("nEthrLow", "0.30", "second threshold of the natural-xenon and structural certificates (keV)")
    mac("nEcFullCells", str(st_full["exclusion_mass_cells"]), "mass cells in the full-endpoint exclusion; verification/structural")
    mac("nEcFullMassLo", st_full["exclusion_mass_interval_GeV"][0], "full-endpoint exclusion, lower mass (GeV); verification/structural")
    mac("nEcFullMassHi", st_full["exclusion_mass_interval_GeV"][1], "full-endpoint exclusion, upper mass (GeV); above it Eq. (closed) at 4.45 keV applies")
    # --- END natural-xenon subsection
    # --- BEGIN annual modulation (modulation.py -> results/modulation.json; self-contained block)
    # Worst case = the exact mimic with the smallest dR/dv_E (linear program), i.e. the largest separation exposure.
    mod = json.load(open(RES / "modulation.json"))
    mod_ideal = {k: v["worst_case"] for k, v in mod["ideal"].items()}
    mod_rel = {k: v["worst_case"] for k, v in mod["released"].items()}
    mac("nModAmpB", f1(100 * mod["ideal"]["7.0"]["b8"]["amplitude"]), "8B annual modulation, percent half-range, Kepler orbit e = 0.0167")
    mac("nModPhaseB", mod["ideal"]["7.0"]["b8"]["date_of_max"].replace("Jan", "January"), "date of the 8B maximum (perihelion)")
    mac("nModAmpDM", f"{100 * mod_ideal['7.0']['dm']['amplitude']:.0f}", "smallest modulation over exact isotropic mimics at 7 GeV, ideal 0.51 keV detector, percent")
    mac("nModAmpDMNNLS", f"{100 * mod['ideal']['7.0']['dm']['amplitude']:.0f}", "same for the NNLS vertex mimic")
    mac("nModAmpDMRel", f"{100 * mod_rel['7.0']['dm']['amplitude']:.0f}", "smallest modulation over mimics at 7 GeV, released XENONnT observable, percent")
    mac("nModPhaseDM", mod_ideal["7.0"]["dm"]["date_of_max"].replace("Jun", "June"), "date of the mimic's maximum (Earth speed maximum)")
    mac("nModRateIdeal", f"{mod['ideal']['7.0']['rate_b8_per_ty']:.0f}", "8B events per t y above 0.51 keV with unit efficiency (ideal detector)")
    mac("nModExposure", f1(mod_ideal["7.0"]["exposure_3sigma_ty"]), "deviance 9 from timing alone at 7 GeV, ideal detector, worst-case mimic (t y)")
    mac("nModExposureLo", f1(mod_ideal["6.7"]["exposure_3sigma_ty"]), "same at 6.7 GeV")
    mac("nModExposureHi", f1(mod_ideal["7.6"]["exposure_3sigma_ty"]), "same at 7.6 GeV")
    mac("nModExposureRel", f"{mod_rel['7.0']['exposure_3sigma_ty']:.0f}", "deviance 9 from timing alone at 7 GeV, released observable, worst-case mimic (t y)")
    mac("nModExposureRelLo", f"{mod_rel['6.7']['exposure_3sigma_ty']:.0f}", "same at 6.7 GeV")
    mac("nModExposureRelHi", f"{mod_rel['7.6']['exposure_3sigma_ty']:.0f}", "same at 7.6 GeV")
    # --- END annual modulation
    # --- BEGIN profiled backgrounds (profiled_backgrounds.py -> results/profiled_backgrounds.json)
    prof = json.load(open(RES / "profiled_backgrounds.json"))
    prof_at = {int(row["mass"]): row for row in prof["rows"]}
    prof_ratios = [row["ratio_profiled_over_fixed"] for row in prof["rows"]]
    free_ratios = [row["ratio_free_over_fixed"] for row in prof["rows"]]
    theta20 = max(abs(value) for value in prof_at[20]["theta_in_sigma"].values())
    mac("nProfSepTwenty", f1(prof_at[20]["exposure_ty_profiled"]),
        "profiled separation exposure at 20 GeV (t y); results/profiled_backgrounds.json")
    mac("nProfSepSixteen", f"{prof_at[16]['exposure_ty_profiled']:.0f}",
        "profiled separation exposure at 16 GeV (t y)")
    mac("nProfFactorLo", f2(min(prof_ratios)), "min profiled/fixed over 13--25 GeV")
    mac("nProfFactorHi", f2(max(prof_ratios)), "max profiled/fixed over 13--25 GeV")
    mac("nFreeSepTwenty", f1(prof_at[20]["exposure_ty_free"]),
        "free-background-cone separation exposure at 20 GeV (t y)")
    mac("nFreeFactorHi", f2(max(free_ratios)), "max free/fixed over 13--25 GeV")
    mac("nProfSigAc", r"9.04/5.76", "AC normalization widths for SR0/SR1 (percent); release YAML")
    mac("nProfSigEr", f"{100 * prof['sigmas']['er']:.0f}", "ER normalization width (percent); release YAML")
    mac("nProfSigRg", f"{100 * prof['sigmas']['rg']:.0f}", "radiogenic normalization width (percent); release YAML")
    mac("nProfThetaTwenty", f2(theta20), "largest absolute fitted normalization shift at 20 GeV, in prior sigma")
    mac("nProfTarget", "9", "target Asimov deviance")
    mac("nProfDevianceFactor", "2", "Poisson deviance prefactor")
    mac("nProfYieldSpan", "1", "radiogenic yield range in release-node units")
    for mass, name in ((13, "Thirteen"), (16, "Sixteen"), (20, "Twenty"), (25, "TwentyFive")):
        row = prof_at[mass]
        values = {
            "Mass": f"{mass}",
            "Fixed": sci(row["exposure_ty_fixed"], 1) if row["exposure_ty_fixed"] >= 1e4 else
                     (f"{row['exposure_ty_fixed']:.0f}" if row["exposure_ty_fixed"] >= 100 else f1(row["exposure_ty_fixed"])),
            "Profiled": sci(row["exposure_ty_profiled"], 1) if row["exposure_ty_profiled"] >= 1e4 else
                        (f"{row['exposure_ty_profiled']:.0f}" if row["exposure_ty_profiled"] >= 100 else f1(row["exposure_ty_profiled"])),
            "Free": sci(row["exposure_ty_free"], 1) if row["exposure_ty_free"] >= 1e4 else
                   (f"{row['exposure_ty_free']:.0f}" if row["exposure_ty_free"] >= 100 else f1(row["exposure_ty_free"])),
        }
        for suffix, value in values.items():
            mac(f"nProfTab{name}{suffix}", value, f"profiled-background table, {mass} GeV {suffix.lower()}")
    # --- END profiled backgrounds
    # --- BEGIN formerly literal numbers (certificates, figure inputs, published rates)
    from fractions import Fraction
    prof_rows = {int(row["mass"]): row for row in json.load(open(RES / "profiled_backgrounds.json"))["rows"]}
    mac("nFreeFactorTwentyFive", f2(prof_rows[25]["ratio_free_over_fixed"]),
        "free/fixed separation exposure at 25 GeV; results/profiled_backgrounds.json")
    mac("nProfPctTwentyFive", f1(100 * (prof_rows[25]["ratio_profiled_over_fixed"] - 1)),
        "profiled/fixed - 1 at 25 GeV (percent); results/profiled_backgrounds.json")
    cert_lit = json.load(open(VER / "certificate.json"))
    mac("nCertMassGeV", f1(float(Fraction(cert_lit["mass_MeV"])) / 1000),
        "nuclear mass used by the convexity certificate (GeV); verification/certificate.json mass_MeV")
    cols = {r["column"]: r for r in cert_lit["results"] if r["endpoint_convention"] == "continuous interpolant"}
    mac("nQstarMinus", f"{float(cols[3]['q_star_MeV'][0]):.5f}", "q_* for the -3 sigma column (MeV); verification/certificate.json")
    mac("nQstarPlus", f"{float(cols[4]['q_star_MeV'][0]):.5f}", "q_* for the +3 sigma column (MeV); verification/certificate.json")
    mac("nUMinus", f"{float(Fraction(cols[3]['U_MeV'])):.2f}", "end point U of the -3 sigma column (MeV); verification/certificate.json")
    mac("nUPlus", f"{float(Fraction(cols[4]['U_MeV'])):.2f}", "end point U of the +3 sigma column (MeV); verification/certificate.json")
    rem = Fraction(cert_lit["per_log_series_remainder_max"])
    mac("nLogRemExp", f"{round(-math.log10(rem)):d}", "log-series remainder bound 10^-N; verification/certificate.json")
    q_lo, q_hi = (Decimal(s) for s in N["q_star_MeV"])
    mac("nQstarWidthExp", f"{round(-math.log10(q_hi - q_lo)):d}",
        "width of the certified q_* enclosure is 10^-N MeV; verification/certificate.json")
    base = json.load(open(VER / "structural" / "results.json"))["baseline"]
    e0_kev = base["E0_keV"].strip("[").split()[0]
    e0_over_end = float(e0_kev) / float(base["endpoint_keV"].strip("[").split()[0])
    mac("nFiveTermRatio", f"{math.sqrt(e0_over_end):.2f}", "sqrt(E0/E_b) of the five-term argument; verification/structural/results.json")
    mac("nFiveTermPct", f"{100 * (1 - math.sqrt(e0_over_end)):.0f}", "1 - sqrt(E0/E_b), percent of the momentum range")
    tail = b8_tail_over_peak(e0_kev)
    mac("nTailPeakExp", f"{math.floor(-math.log10(tail)):d}",
        f"natural-xenon 8B spectrum on [E0, E_b] is below 10^-N of its peak (max ratio {tail:.2e}); b8_tail_over_peak")
    SHELL_U = 400.0  # fixed input: U in viz/render_shell.py and viz/compose_shell.py
    mac("nShellU", f"{SHELL_U:.0f}", "Galactic shell speed drawn in Fig. 1; viz/render_shell.py U")
    mac("nShellKinkLo", f"{abs(SHELL_U - A.V_EARTH):.0f}", "|u - v_E| for the Fig. 1 shell")
    mac("nShellKinkHi", f"{SHELL_U + A.V_EARTH:.0f}", "u + v_E for the Fig. 1 shell")
    mac("nPubBrateSRzero", "3.6", "fixed input: published SR0 8B rate (t^-1 y^-1), XENON:2024hup")
    mac("nPubBrateSRone", "3.3", "fixed input: published SR1 8B rate (t^-1 y^-1), XENON:2024hup")
    # --- END formerly literal numbers
    (PAPER / "numbers.tex").write_text("\n".join(L) + "\n")


def write_tables(N):
    def win(w):
        return f"[{w[0]:.2f}, {w[1]:.2f}]" if w else "--"

    rows = []
    for t in N["ideal_thresholds"]:
        if t not in (0.2, 0.3, 0.35, 0.4, 0.51, 0.7, 1.0, 1.3):
            continue
        w = N["ideal_windows"][t]
        rows.append(f"{t:.2f} & {N['ideal_analytic_m_hi'][str(t)]:.2f} & {win(w['1e-12'])} & {win(w['0.001'])} \\\\")
    (PAPER / "tables" / "ideal.tex").write_text("\n".join(rows) + "\n")

    rob = N["rob"]
    iso = lambda k: rob[k + " | isotropic"]  # noqa: E731
    yields = [iso(k.replace(" | isotropic", "")) for k in rob if k.startswith("yield") and k.endswith("isotropic")]
    ylo = [w[0] for w in yields]
    yhi = [w[1] for w in yields]
    nominal = iso("yield tly=+0 tqy=+0")

    def rng(a, b, d=2):
        return f"{a:.{d}f}" if abs(a - b) < 10 ** (-d) / 2 else f"{a:.{d}f}--{b:.{d}f}"

    lines = [
        rf"nominal ($\vE={N['vE']:.0f}$, $\vesc={N['vesc']:.0f}$) & {nominal[0]:.2f} & {nominal[1]:.1f} \\",
        rf"yield morphs $t_{{\rm Ly}},t_{{\rm Qy}}=\pm1$ & {rng(min(ylo), max(ylo))} & {rng(min(yhi), max(yhi), 1)} \\",
        rf"$\vE=218\kms$ & {iso('vE=218 vesc=544')[0]:.2f} & {iso('vE=218 vesc=544')[1]:.1f} \\",
        rf"$\vE=248\kms$ & {iso('vE=248 vesc=544')[0]:.2f} & {iso('vE=248 vesc=544')[1]:.1f} \\",
        rf"$\vesc=500\kms$ & {iso('vE=232 vesc=500')[0]:.2f} & {iso('vE=232 vesc=500')[1]:.1f} \\",
        rf"$\vesc=600\kms$ & {iso('vE=232 vesc=600')[0]:.2f} & {iso('vE=232 vesc=600')[1]:.1f} \\",
        rf"own \B\ spectrum & {iso('own 8B spectrum')[0]:.2f} & {iso('own 8B spectrum')[1]:.1f} \\",
        rf"tolerance $10^{{-4}}$ ($10^{{-2}}$) & {iso('tolerance 0.0001')[0]:.2f} ({iso('tolerance 0.01')[0]:.2f}) & "
        rf"{iso('tolerance 0.0001')[1]:.1f} ({iso('tolerance 0.01')[1]:.1f}) \\",
    ]
    (PAPER / "tables" / "robust.tex").write_text("\n".join(lines) + "\n")
    profiled_rows = [
        rf"$\nProfTab{name}Mass$ & $\nProfTab{name}Fixed$ & $\nProfTab{name}Profiled$ & $\nProfTab{name}Free$ \\"
        for name in ("Thirteen", "Sixteen", "Twenty", "TwentyFive")
    ]
    profiled_rows.append(r"\bottomrule")
    (PAPER / "tables" / "profiled.tex").write_text("\n".join(profiled_rows) + "\n")
    # --- BEGIN natural-xenon subsection (sec:isotopes): form-factor gap table
    write_formfactor_table()
    # --- END natural-xenon subsection


# --- BEGIN natural-xenon subsection (sec:isotopes) helpers; self-contained
def arb_lo_hi(s):
    """Lower and upper end of an Arb ball string '[mid +/- rad]' (or a plain decimal) as Decimals."""
    text = str(s).strip()
    m = re.match(r"\[?\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", text)
    mid = Decimal(m.group(1))
    rm = re.search(r"\+/-\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", text)
    rad = Decimal(rm.group(1)) if rm else Decimal(0)
    return mid - rad, mid + rad


def write_formfactor_table():
    """paper/tables/formfactor.tex: closed-form gap G and nominal-(p,r) status per form factor and threshold.

    G floors come from verification/structural/results.json (A_models); the pass/fail flags of the fixed nominal
    pair from verification/natural_xenon/form_factor_variations.json, and for the certificate's own Helm s=0.9 fm
    from verification/natural_xenon/results.json.
    """
    st = json.load(open(VER / "structural" / "results.json"))
    nat = json.load(open(VER / "natural_xenon" / "results.json"))
    ff = {c["name"]: c for c in json.load(open(VER / "natural_xenon" / "form_factor_variations.json"))["cases"]}
    gaps = {(c["model"], c["threshold_keV"]): c for c in st["A_models"]}

    def gap(model, thr):
        c = gaps[(model, thr)]
        assert c["feasible"]
        lo, _ = arb_lo_hi(c["sufficient_gap"])
        return format(lo.quantize(Decimal("0.0001"), rounding=ROUND_FLOOR), "f")

    def status(flags):
        if flags["passed"]:
            return "yes"
        if not flags["ineq_2lp_plus_p"]:
            return "fails 1st"
        assert not flags["ineq_c"]
        return "fails 2nd"

    def nominal_flags(cv):
        lo_b, _ = arb_lo_hi(cv["min_2lp_plus_p"])
        _, hi_c = arb_lo_hi(cv["max_c"])
        return {"passed": lo_b > 0 and hi_c < 0, "ineq_2lp_plus_p": lo_b > 0, "ineq_c": hi_c < 0}

    # (structural model, form_factor_variations case or None for the certificate's own Helm, row label)
    rows_spec = [
        ("helm_.8", "helm_skin_0.8", r"Helm, $s=\nFFSkinLow\,\mathrm{fm}$"),
        ("helm_.9", None, r"Helm, $s=\nFFSkin\,\mathrm{fm}$"),
        ("helm_1", "helm_skin_1.0", r"Helm, $s=\nFFEngelSkin\,\mathrm{fm}$"),
        ("engel", "engel", "Engel"),
        ("lowq", "lowq_same_msq", r"low-$q$"),
    ]
    rows = []
    for model, case, label in rows_spec:
        if case is None:
            s51 = status(nominal_flags(nat["curvature"]))
            s30 = status(nominal_flags(nat["curvature_at_030"]))
        else:
            s51 = status(ff[case]["at_0.51"]["flags"])
            assert ff[case]["neighborhood_0.51"]["flags"]["passed"] == ff[case]["at_0.51"]["flags"]["passed"]
            s30 = status(ff[case]["at_0.30"]["flags"])
        rows.append(f"{label} & {gap(model, '.51')} & {s51} & {gap(model, '.30')} & {s30} \\\\")
    # the table minimum must not undercut the prose floors nStructGlo / nStructGloThirty
    for thr, key in ((".51", "struct_G_051"), (".30", "struct_G_030")):
        floors = [Decimal(gap(m, thr)) for m, _c, _l in rows_spec]
        assert min(floors) >= Decimal(json.loads((RES / "numbers.json").read_text())[key])
    (PAPER / "tables" / "formfactor.tex").write_text("\n".join(rows) + "\n")
# --- END natural-xenon subsection helpers


if __name__ == "__main__":
    main()

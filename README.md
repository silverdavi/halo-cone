# Where ⁸B neutrinos are exactly dark matter in xenon

For fixed mass and detector response the dark-matter spectrum is linear in the
velocity distribution, so with the cross section free the set of possible
spectra is a convex cone. The ⁸B CEνNS spectrum is *exactly* degenerate with
dark matter of mass m if and only if it lies in that cone. This project computes
the cone on the released XENONnT response and asks which astrophysical
assumption controls the answer.

## Status

The tables below are an L2 scan of one observable. They are not the exact-membership theorem.

Exact membership for one isotope, at a 0.51 keV threshold, is the lemma 6.67–7.66 GeV for ¹³¹Xe, with E_crit = 0.392 keV. Natural xenon does not shift that window. In the Bahcall-spectrum, matched-Helm model the reconstructed halo at the lightest isotope is negative from 6.65 to 7.80 GeV, the ¹²⁴Xe endpoint excludes everything lighter, and above 7.80 GeV every isotope threshold lies below v_E. The high-mass step needs the weighted operator (B.8): concavity of η does not survive multiplication by unequal isotope form factors. An independent 60- and 100-digit evaluation of the five-term sum, using only the standard library, reproduces

  η(w_124(E_0)) = −1.92831016050503×10⁻¹⁰ at 7 GeV,

and the uniform bound −1.63935228133417×10⁻¹⁰. The two precisions agree past 10⁻⁶⁰. That checks this flux table and this response model. It is not an interval certificate and not a second flux evaluation. The script and the table are in `verification/`.

The interval certificate is `verification/natural_xenon/certify.py` (python-flint 0.9.0, Arb balls at 192 bits, exact segment integrals, whole mass cells; runs in ~6 s). It proves, for the Bahcall table above and the nine-isotope Helm model:

- the five-term endpoint recursion is negative on 39 cells covering [6.65, 7.80] GeV (uniform bound < −2.15×10⁻¹⁰ in the unnormalised rate), so strict pointwise degeneracy up to the ¹²⁴Xe endpoint E_b = 4.639 keV is empty for natural xenon at 0.51 keV;
- upper edge: with L = ∂²_x + 0.146 ∂_x + 0.092 (x = √E) every isotope weight satisfies 2ℓ' + p ≥ 0 and ℓ'' + ℓ'² + pℓ' + r ≤ 0 at 0.51 keV while L applied to the ⁸B target is > 1.0×10⁶, so any isotropic mimic on [0.51, E_cut] needs m ≤ m_hi(¹³⁶Xe) = 7.7932 GeV, for every E_cut;
- the truncated-window sufficiency conjecture is false: m = 6.0233 GeV satisfies v_i(3.5 keV) ≤ 776 km/s for every isotope but is excluded on [0.51, 3.5] keV by a nine-point separating functional (83 shell breakpoints checked);
- for every threshold ≤ 0.30 keV the window on [E_thr, 3.5] keV is empty (support excludes m < 5.764, a three-point functional excludes [5.764, 5.893], the curvature operator with p = 0.113, r = 0.089 excludes m > 5.893 = m_hi(¹³⁶Xe; 0.30)).

Not proved: the exact truncated membership set, its sharp upper edge, or the sharp mixture critical threshold (numerically between 0.30 and 0.35 keV).

Exact time-resolved membership fails separately. The ⁸B flux peaks at perihelion and the Earth's Galactic speed peaks in June, so the neutrino rate is not a function of Earth speed, while every isotropic halo is. `modulation.py` (-> `results/modulation.json`) quantifies this inside the window: the ⁸B rate varies by ±3.3% with its maximum on 3 January; every exact isotropic mimic at 7 GeV varies by at least ±11% with its maximum on 2 June (the minimum over the cone is a linear program, attained by the shell construction of Theorem 1). With the spectrum integrated over the ROI, the Asimov deviance between the two time series reaches 9 at 3.0 t·y for an ideal detector recording all 294 ⁸B events per tonne-year above 0.51 keV (2.5–3.3 t·y across the window) and at 100 t·y in the released XENONnT observable. The earlier phase-floor bound ε* ≤ log(400)/D_phase ≈ 514 t·y (D_phase = 0.0116 per t·y, `verification/README.txt`) is superseded by this computation.

## Result

The answer depends on isotropy, not on the Maxwellian form.

| Halo class | Generators | Exact degeneracy with ⁸B (XENONnT observable) |
|---|---|---|
| Any lab-frame speed distribution on [0, v_esc + v_E] | lab streams | every m ≥ 4.6 GeV |
| Isotropic in the Galactic frame, speeds ≤ v_esc | boosted shells | **4.9 ≤ m ≤ 12.6 GeV** only |

(v_esc = 544, v_E = 232 km/s; residual tolerance 10⁻³ in L2; self-consistent
⁸B target folded through the same response as the generators.)

Why the window closes on top. For an isotropic Galactic-frame halo the lab
halo function η(v_min) is **concave for v_min < v_E**: boosting a shell of
speed v gives η_v(w) = (v + v_E − max(w, |v − v_E|))⁺ / (2 v v_E): flat, then a
ramp down from w = |v − v_E|, then zero from w = v + v_E ≥ v_E. Its only upward
kink sits at or above v_E, so below v_E every generator is concave, and
positive combinations preserve that. (The same holds for the SHM: η'' < 0 for
w < v_E.)
The ⁸B target (dR/dE)/(A²F²) as a function of v_min is strictly convex above
0.16 keV. A concave function cannot equal a strictly convex one on an interval,
so exact degeneracy needs v_min(E_thr) ≥ v_E, i.e.

  m ≤ m_hi(E_thr):  7.7 GeV at E_thr = 0.51 keV,  5.8 GeV at 0.3 keV,  4.7 GeV at 0.2 keV.

Why it closes at the bottom (both classes). η vanishes above v_esc + v_E, so
exact equality fails wherever the ⁸B spectrum is still positive above the
dark-matter kinematic endpoint. For ¹³¹Xe (m_N = 122.3 GeV) and an ⁸B endpoint
of 16.36 MeV that recoil endpoint is E_* = 4.38 keV, so

  m ≥ m_lo = 6.66 GeV.

At 5.2 GeV the dark-matter endpoint is only 2.73 keV. The neutrino rate above
that is 0.16 events per tonne-year (1.9×10⁻⁴ of the spectrum): invisible to an
L2 tolerance of 10⁻³, fatal for exact equality. The earlier 5.2 GeV edge was
that tolerance applied to a spectrum cut at 10⁻³ of its peak.

Concavity below v_E is necessary and not sufficient. The cone membership test
is the inversion: with a = v_E and q = −2a η',

  R(v) = Σ_{j≥0} q(v + (2j+1)a),    q(w) = R(a−w) − R(a+w)  for w < a,

and η is in the cone iff R is nonnegative and nonincreasing, supported in
[0, v_esc], the low-w identity holds, and the jump of η at a is nonnegative.
The speed measure is dF(v) = v(−dR). For a target that is strictly convex on
its whole nonzero range above the certified inflection of the Bahcall spectrum
(q_* = 3.1815348583 MeV, E_R = 0.16553 keV at m_N = 122.3 GeV; `verification/certify.py`), membership collapses to the two edges

  w(E_thr) ≥ v_E  and  w(E_*) ≤ v_esc + v_E,

i.e. 6.66 ≤ m ≤ 7.67 GeV at a 0.51 keV threshold. The window exists only for
E_thr ≥ E_* (v_E / (v_esc+v_E))² = 0.391 keV. Below that threshold the exact
degeneracy is empty for every mass. A 0.3 keV ideal threshold does this; the
0.1 keV resolution quoted before is not what removes it.

Finite resolution opens the window beyond the analytic edges; it converges as
bins shrink. Isotropic class, ideal energy-resolving xenon detector:

| threshold → / bin width ↓ | 0.3 keV | 0.4 keV | 0.51 keV | 0.7 keV | 1.0 keV |
|---|---|---|---|---|---|
| 0.5 keV  | 5.1–10.5 | 5.2–11.1 | 5.2–11.7 | 5.3–12.6 | 5.2–14.1 |
| 0.25 keV | 5.2–8.7  | 5.2–9.2  | 5.2–9.9  | 5.3–11.1 | 5.5–12.6 |
| 0.1 keV  | none     | 8.1–8.2  | 5.2–8.7  | 5.3–9.9  | 5.5–11.7 |
| 0.05 keV | none     | none     | 5.2–8.2  | 5.3–9.5  | 5.5–11.3 |
| exact edges | empty | 6.66–6.74 | 6.66–7.67 | 6.66–9.1 | 6.66–11.0 |

The table above is an L2 < 10⁻³ scan, so it still reports a window at masses
where only the extreme tail disagrees. The exact window at 0.51 keV is
6.66–7.67 GeV, and it is empty for every threshold below 0.391 keV. Separating
a 5.2 GeV isotropic halo from ⁸B by the tail alone takes ~60 tonne-years,
because that tail is 0.16 events per tonne-year; the exact edge and the
practical edge are not the same.

Outside the window the mismatch is finite and xenon alone separates the two
hypotheses with the halo profiled. Asimov Poisson deviance 9 (≈3σ), XENONnT
observable, cross section and halo refit:

| m [GeV] | 5.3–12.6 | 13 | 14 | 15 | 16 | 18 | 20 | 25 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| streams [t·y] | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ |
| isotropic [t·y] | ∞ | 5×10⁵ | 3×10⁴ | 1100 | 140 | 17 | 6.9 | 2.7 | 2.3 |

These exposures include the ⁸B Poisson fluctuations only. Adding the released
SR0+SR1 accidental-coincidence, electronic-recoil and radiogenic-neutron
templates with fixed normalisation (`backgrounds.py` -> `results/backgrounds.json`)
lengthens the isotropic exposure by a factor 2.3–2.9 over 13–25 GeV: 18.0 t·y
at 20 GeV and 360 t·y at 16 GeV.

Exposures are tonne-years of *combined* SR0+SR1-like exposure: the release
templates are normalised per tonne-year of each science run, so the two runs are
weighted by their livetimes (1.174 : 2.343 t·y) in `analysis.released_response`.
(An earlier version of these tables counted a tonne-year in each run as one
tonne-year and was a factor 2 too optimistic.) Above the window the curve
flattens at ~2.3 t·y (a heavy isotropic halo gives a flat spectrum; ⁸B is
steep). XENONnT's 3.5 t·y separates isotropic-halo dark matter from ⁸B for
m ≥ 23 GeV, 20 t·y for m ≥ 18 GeV, 1000 t·y for m ≥ 15 GeV; nothing in xenon
reaches 5–12.6 GeV.

Halo-independent discovery limit (worst case over isotropic halos, Asimov 3σ,
⁸B flux nuisance 3.81%, XENONnT observable; `discovery.py`):

| m [GeV] | 6 | 8 | 10 | 12 | 14 | 16 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| σ_HI / σ_SHM at 3.5 t·y | ∞ | 670 | 63 | 20 | 8.7 | 4.5 | 1.6 | 1.4 |
| σ_HI / σ_SHM at 10⁴ t·y | ∞ | 660 | 112 | 63 | 42 | 18 | 2.2 | 1.4 |
| fog index n of σ_HI at 10³ t·y | – | 2.8 | 3.4 | 5.1 | 7.4 | 6.8 | 2.5 | 2.2 |

Below m_hi the halo with dark matter at rest in the Galaxy gives no recoils
above threshold, so no cross section guarantees discovery. In every case
examined the worst case is that slowest shell.

Second target inside the window (forecast; CRESST-III Detector A cut survival
and band acceptance, CaWO₄, 24 log bins 0.03–16 keV, all solar fluxes; xenon
fixed at the stated exposure; joint refit of halo and cross section):

| m [GeV] | 5.5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|
| streams, untagged CaWO₄ for deviance 9 [t·y] | 3.6 | 3.0 | 2.6 | 2.7 | 4.5 |
| isotropic, same [t·y] | 0.29 | 0.24 | 0.17 | 0.13 | 0.07 |

(xenon fixed at 20 t·y; with 3.5 t·y the numbers change by <10%.)

Isotropy lowers the required CaWO₄ exposure by a factor 10–80. The single most
informative CaWO₄ bin inside the published acceptance is oxygen recoils at
7.3–12.3 keV, but alone it needs ≳ 13 t·y; the whole untagged spectrum is what
breaks the degeneracy. An ideal oxygen bin at 25–30 keV is a one-functional
separator — dark matter in the exact window cannot recoil above 23 keV on
oxygen, while ⁸B reaches 36 keV — but that bin holds 2×10⁻³ events per
tonne-year of oxygen, so the exposure to use it is thousands of tonne-years.
Minimality of the functional is not minimality of the exposure.
CRESST-III Detector A is 1.6×10⁻⁵ t·y.

Checks. The mono-energy fold reproduces the official ⁸B and 3/6/12 GeV WIMP
templates at cosine ≥ 0.9997 (`results/summary.json`); a random conic
combination projects back at 10⁻¹⁶; the stream cone reproduces the released
WIMP templates below 10⁻⁶. With the *official* ⁸B template as target instead of
the self-consistent fold, every residual sits on a 1.8% floor from template
construction (the yield-morph 1σ shape radius is 3.9%); at that tolerance the
windows agree with the self-consistent ones to within 0.5 GeV
(`results/isotropic.json`).

Caveats. Elastic isospin-conserving SI only. v_E fixed at 232 km/s; using the
annual range shifts the isotropic upper edge by about ±1 GeV, and because a
time average of concave functions is concave the lemma holds with v_E replaced
by its annual minimum. The stream class includes dark matter nearly at rest in
the lab frame; a co-rotating dark disk is the physical case where heavy dark
matter can mimic ⁸B. The deviance-9 exposures are expected values, not
look-elsewhere-corrected significances.

## Layout

```
halo_cone/
  run_all.sh         rebuild everything in order (results -> numbers -> figures -> Fig. 1 -> PDF)
  analysis.py        released-observable response, 8B spectrum, CRESST forecast (original run)
  isotropic.py       generators for both halo classes, NNLS residual
  window.py          windows vs halo class, resolution, threshold; analytic edges; exposures -> results/window.json
  ideal.py           ideal energy-resolving detector, 1e-12 and 1e-3 windows vs threshold     -> results/ideal.json
  robustness.py      window edges under yield morphs, v_E, v_esc, spectrum source, tolerance -> results/robustness.json
  discovery.py       halo-independent 3-sigma discovery limit and fog index                   -> results/discovery.json
  complement.py      CaWO4 exposure that lifts the degeneracy, by class                       -> results/complement.json
  modulation.py      annual modulation of 8B vs the worst-case isotropic mimic inside the window -> results/modulation.json
  backgrounds.py     released AC/ER/radiogenic templates; background-corrected separation exposures -> results/backgrounds.json
  profiled_backgrounds.py  separation exposures with release normalization and radiogenic-yield nuisances -> results/profiled_backgrounds.json
  export_numbers.py  results/*.json + certificates -> paper/numbers.tex (one macro per number), paper/tables/*.tex
  figures.py         Figs. 2-6 (matplotlib)                                                   -> paper/figs/*.pdf,*.png
  config/figstyle.py shared palette, roles, fonts, panel labels, legends-below, save_fig
  viz/compose_shell.py  Fig. 1: exact vector geometry of the boosted shell (flat fills, hidden lines dashed) + eta_u(w) panel -> paper/figs/shell.pdf
  hero/              cover art (Cycles render + drawsvg composition), not part of the paper build
  verification/      certify.py: rational-interval certificate of the 8B convexity lemma (q_*, E_*)
  verification/natural_xenon/  certify.py: Arb certificate for natural xenon (upper edge 7.793 GeV, 0.30 keV emptiness, strict set empty)
  paper/main.tex     RevTeX 4.2 (PRD); thin driver that inputs macros.tex, numbers.tex and sections/*.tex
  paper/sections/    00_abstract ... 07_appendix, 99_references
                     (main.tex, macros.tex and sections/ are added to the public repository after acceptance;
                     run_all.sh skips the PDF build without them)
  paper/tables/      generated table bodies (ideal.tex, robust.tex, formfactor.tex)
  fetch_data.sh      downloads data/ from the pinned public sources and checks data_SHA256SUMS
  data/              XENONnT light-WIMP release v0.0.1 (Zenodo 10.5281/zenodo.14522707), CRESST-III Detector A
                     (arXiv:1905.07335 ancillary files), solar fluxes (not versioned)
  results/           *.json outputs of the scripts above, numbers.json
```

```sh
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
./fetch_data.sh           # public inputs into data/, checked against data_SHA256SUMS
./run_all.sh              # ~8 min; SKIP_SLOW=1 ./run_all.sh rebuilds numbers, figures and the PDF only
```

Fig. 1 needs the Homebrew cairo library for `cairosvg`; `viz/fonts.conf` points fontconfig at the TeX Live copy of
TeX Gyre Heros. The cover art in `hero/` needs Blender as a Python module (`bpy`, the eilmer project's `.venv-bpy`).

## Sources

- XENONnT light-WIMP data release (SR0+SR1 templates, mono-energetic basis, yield models).
- CRESST-III Detector A release: `C3P1_DetA_cuteff.dat`, `C3P1_DetA_eff_AR_{O,Ca,W}.dat`.
- Solar fluxes: Vinyoles et al. 2017 (GS98); ⁸B uncertainty from the XENONnT release.
- Halo-independent convex formalism: Gelmini, Gondolo, Huh, Witte, JCAP 12 (2017) 039;
  Ferrer, Ibarra, Wild (2015); Ibarra, Rappelt (2017).
- Astrophysical uncertainties on the floor: Davis, JCAP 03 (2015) 012; O'Hare, PRL 127, 251802 (2021).

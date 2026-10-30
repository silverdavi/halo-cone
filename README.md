# halo-cone: where solar ⁸B neutrinos can exactly mimic dark matter in xenon

Code and interval-arithmetic certificates for D. H. Silver, *Isotropy, not the Maxwellian, decides
where solar ⁸B neutrinos can exactly mimic dark matter in xenon* (submitted to JCAP).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23824845.svg)](https://doi.org/10.5281/zenodo.23824845)

Archive: Zenodo, concept DOI [10.5281/zenodo.23824845](https://doi.org/10.5281/zenodo.23824845) (all versions);
v1.0.1 is [10.5281/zenodo.23824846](https://doi.org/10.5281/zenodo.23824846).

Solar ⁸B neutrinos scattering coherently on xenon give a recoil spectrum close to that of a light
dark-matter particle. If any dark-matter velocity distribution is allowed, some distribution
reproduces the neutrino spectrum exactly at every mass above a kinematic minimum, so no exposure
can separate the two by spectral shape. If the halo is only required to be isotropic in the
Galactic frame, exact mimicry is confined to a closed mass window, 6.7–7.7 GeV for a single xenon
isotope at the 0.51 keV XENONnT threshold, and it disappears for thresholds below 0.39 keV,
whatever the speed distribution. This repository computes the windows on the public XENONnT
light-dark-matter release and on an ideal detector, the separation exposures, discovery limits,
annual modulation and CaWO₄ complementarity that follow, and proves the analytic steps with
standalone interval-arithmetic certificates.

## Main results

| Halo class | Exact degeneracy with ⁸B |
|---|---|
| any velocity distribution | every m ≥ 6.7 GeV (ideal ¹³¹Xe detector); every m ≥ 4.6 GeV in the released XENONnT observable |
| isotropic in the Galactic frame | 6.7 ≤ m ≤ 7.7 GeV (¹³¹Xe, 0.51 keV threshold, exact); 5.0–12.6 GeV in the released observable at residual 10⁻³ |

Why: the halo function of every isotropic halo is concave in v_min below the Earth's speed v_E,
while the ⁸B spectrum written as a halo function is strictly convex. Exact equality therefore
needs v_min(E_thr) ≥ v_E (upper edge) and v_min(E_end) ≤ v_esc + v_E (lower edge). For natural
xenon the upper edge is 7.8 GeV (¹³⁶Xe), proved by a weighted curvature argument.
The numbers in the paper are written by `export_numbers.py` into `paper/numbers.tex` and
`results/numbers.json`; those files are the reference, not this README.

## Reproduce

Python 3.11; about 8 minutes on a laptop.

```sh
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
./fetch_data.sh            # public inputs into data/, checked against data_SHA256SUMS
./run_all.sh               # results/*.json -> paper/numbers.tex, paper/tables/ -> paper/figs/
SKIP_SLOW=1 ./run_all.sh   # reuse results/*.json; rebuild numbers, tables and figures only
```

`fetch_data.sh` downloads:

- XENONnT light-WIMP data release v0.0.1 (commit 64d2e13), Zenodo
  [doi:10.5281/zenodo.14522707](https://doi.org/10.5281/zenodo.14522707);
- CRESST-III Detector A efficiencies (ancillary files of arXiv:1905.07335v3);
- solar neutrino fluxes (Vinyoles et al. 2017, GS98) as normalised in
  [cajohare/NeutrinoFog](https://github.com/cajohare/NeutrinoFog) at commit 0df3d0c.

Figure 1 (`viz/compose_shell.py`) needs the Homebrew cairo library for `cairosvg`;
`viz/fonts.conf` points fontconfig at the TeX Live copy of TeX Gyre Heros.
`HALO_FIG_W2` and `HALO_SHELL_FONT_SCALE` re-render the full-width figures for a single-column
page (the JCAP build uses `HALO_FIG_W2=6.1111 HALO_SHELL_FONT_SCALE=1.07`).

## Certificates

Each certificate is a standalone script that checks the SHA-256 of its input table and needs only
the standard library and, for the Arb ones, `python-flint==0.9.0`. Each runs in seconds.

| Script | Proves | Last line on success |
|---|---|---|
| `verification/certify.py` | convexity of the ⁸B target: inflection q_* and E_*, in exact rational interval arithmetic | `"status": "ALL RATIONAL INTERVAL ASSERTIONS PASSED"` |
| `verification/natural_xenon/certify.py` | natural xenon (Arb, 192 bits): upper edge 7.793 GeV, emptiness at thresholds ≤ 0.30 keV, the strict five-term obstruction on 6.65–7.80 GeV | `"status": "ALL INTERVAL ASSERTIONS PASSED"` |
| `verification/structural/certify.py` | the upper-edge separator for five form-factor models at two thresholds, and the neutron-skin and moment bounds | `All asserted interval tests passed; wrote results.json.` |
| `verification/verify_five_terms.py` | an independent 60- and 100-digit evaluation of the five-term sum (standard library only; not an interval enclosure) | |

```sh
.venv/bin/python verification/certify.py
(cd verification/natural_xenon && ../../.venv/bin/python certify.py)
(cd verification/structural && ../../.venv/bin/python certify.py)
```

`PROOF.txt` in each folder states what is proved, what is a numerical indication, and what is open.

## Layout

```
run_all.sh               rebuild everything in order
analysis.py              released-observable response, 8B spectrum, CRESST forecast
isotropic.py             generators for both halo classes, NNLS residual
window.py                windows vs halo class, resolution and threshold; analytic edges   -> results/window.json
ideal.py                 ideal energy-resolving detector                                  -> results/ideal.json
robustness.py            edges under yield morphs, v_E, v_esc, spectrum source, tolerance -> results/robustness.json
discovery.py             halo-independent 3-sigma discovery limit and fog index           -> results/discovery.json
complement.py            CaWO4 exposure that lifts the degeneracy                         -> results/complement.json
modulation.py            annual modulation of 8B vs the worst-case isotropic mimic        -> results/modulation.json
backgrounds.py           released AC/ER/radiogenic templates, separation exposures        -> results/backgrounds.json
profiled_backgrounds.py  separation exposures with normalization and yield nuisances      -> results/profiled_backgrounds.json
export_numbers.py        results/*.json + certificates -> paper/numbers.tex, paper/tables/*.tex
figures.py               Figs. 2-6 (matplotlib)                                           -> paper/figs/
viz/compose_shell.py     Fig. 1, boosted-shell geometry and its halo function             -> paper/figs/shell.pdf
config/figstyle.py       palette, fonts, figure widths
verification/            interval-arithmetic certificates (above)
hero/                    cover art, not part of the paper build
paper/                   numbers.tex, tables/ and figs/ (generated); the manuscript source is
                         added when the arXiv version is posted
```

## Scope

Elastic, isospin-conserving spin-independent scattering. v_E = 232 km/s and v_esc = 544 km/s
(robustness table in the paper). Halos with anisotropic components (Gaia–Enceladus debris, streams,
a dark disk) are outside the theorem.

## Citation

```bibtex
@misc{Silver:halocone,
  author    = {Silver, David H.},
  title     = {Code and certificates for ``Isotropy, not the Maxwellian, decides where solar
               $^8$B neutrinos can exactly mimic dark matter in xenon''},
  version   = {v1.0.1},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23824845},
  url       = {https://github.com/silverdavi/halo-cone}
}
```

Please also cite the XENONnT data release (doi:10.5281/zenodo.14522707) when you use its templates.

## License

MIT (`LICENSE`). The downloaded inputs keep their own licences.

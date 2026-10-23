#!/usr/bin/env bash
# Rebuild every result, number, table and figure from the released data, in order.
# Usage: ./run_all.sh            (uses the project venv at .venv)
#        SKIP_SLOW=1 ./run_all.sh (reuse results/*.json; only numbers, figures, paper)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

if [ -z "${SKIP_SLOW:-}" ]; then
  echo "== Cone tests on the released XENONnT observable and on an ideal detector =="
  [ -f results/mono_grid.npz ] || $PY analysis.py   # mono-energetic response grid from the release (slow once)
  $PY window.py        # windows vs class, resolution, threshold; separation exposures   -> results/window.json
  $PY ideal.py         # ideal energy-resolving detector, exact and 1e-3 windows        -> results/ideal.json
  $PY robustness.py    # yields, v_E, v_esc, spectrum source, tolerance                 -> results/robustness.json
  $PY discovery.py     # halo-independent discovery limit and fog index                 -> results/discovery.json
  $PY complement.py    # CaWO4 exposure that lifts the degeneracy                        -> results/complement.json

  echo "== Interval-arithmetic certificates =="
  $PY verification/certify.py                   # convexity of the 8B target (q_*, E_*)
  (cd verification/natural_xenon && ../../$PY certify.py)   # natural xenon: upper edge, 0.30 keV emptiness, strict set
fi

echo "== Annual modulation inside the window (about 25 s, so it runs in both paths) =="
$PY modulation.py    # 8B versus the worst-case isotropic mimic, timing alone                 -> results/modulation.json
$PY backgrounds.py   # released AC/ER/radiogenic templates: background-corrected separation exposures (under 1 s) -> results/backgrounds.json
$PY profiled_backgrounds.py  # profile release normalization and radiogenic-yield nuisances    -> results/profiled_backgrounds.json

echo "== Numbers: results/*.json + certificates -> paper/numbers.tex, paper/tables/*.tex =="
$PY export_numbers.py

echo "== Figures 2-6 (matplotlib, config/figstyle.py) -> paper/figs/ =="
$PY figures.py

echo "== Figure 1: exact vector geometry of the boosted shell (drawsvg + cairosvg) =="
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib $PY viz/compose_shell.py   # needs Homebrew cairo

echo "== Manuscript (RevTeX 4.2) =="
if [ ! -f paper/main.tex ]; then
  echo "paper/main.tex not in this tree (manuscript text is released after acceptance); skipping paper build"
elif command -v latexmk >/dev/null 2>&1; then
  (cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
else
  echo "latexmk not found; skipping paper build"
fi

echo "== done: paper/main.pdf =="

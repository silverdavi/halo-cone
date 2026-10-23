Independent evaluation of the natural-xenon five-term obstruction
================================================================

Run: python verify_five_terms.py
Requires only Python 3's standard library.

Inputs: the b8_spectrum.csv from the previously saved isotropic_halo_checks.zip,
the nine NIST atomic-mass decimal strings, and the abundance/Helm/weak-response
constants explicitly defined in verify_five_terms.py. This is an independent
implementation using the SAME flux input, not an independent flux measurement
or a different published evaluation. No new flux file was supplied in this turn.

Implementation: decimal arithmetic at 60 and 100 digits; Machin's formula for
pi; sine/cosine Taylor series; Newton cube roots; direct analytic integration
over every nonzero segment of the piecewise-linear neutrino flux. The script
imports neither the original code nor NumPy/SciPy. The original tail-integral
arrays and their floating-point cancellations are not reused.

At m=7 GeV and E0=4.4553366028102366591495697210 keV, the five contributions are
 +2.140669509438794829533204e-10
 -3.769820106093594005459389e-11
 +5.298021434634559017803947e-12
 -1.255717014462972885950031e-10
 -2.489260859217838485752665e-10
Sum:
 -1.928310160505030352537391e-10

Uniform upper bound on 6.65 <= m <= 7.80 GeV:
 -1.639352281334165991328297e-10

Uniformity uses the previously proved monotonicities of T(E), c_i(E)/c_0(E),
and alpha_i(m). The bound is the sum at m=6.65 with its positive third term
replaced by that term at m=7.80. The code also checks the four strict ordering
margins that make the recursion have exactly five terms at the mass endpoints
and at 7 GeV; monotonicity in m extends these endpoint checks to the interval.

The corresponding halo value is negative on a recoil interval by continuity,
so this is an obstruction to equality almost everywhere, not only at one point.

Precision check: the 60- and 100-digit sums differ by less than 1e-60 absolute.
The last digits differ from the original double-precision output; the sign and
quoted two-significant-digit conclusions agree. results.json retains the input
hash, nuclear masses, normalization, ordering margins, and full precision output.

LIMITATION: agreement at two precisions is not a rigorous interval enclosure.
No interval-arithmetic certificate or physical spectrum uncertainty analysis
is claimed. The zero-above-endpoint and linear-interpolation conventions are
part of the mathematical target being checked.

Paper scope
-----------
Single-isotope lemma: 6.67--7.66 GeV and Ecrit=0.392 keV for 131Xe.
Natural xenon: exact membership is empty at 0.51 keV in the specified flux and
matched-Helm model, combining this recursion obstruction with the endpoint and
weighted curvature tests. Retain the weighted differential operator in B.8:
concavity of eta alone does not imply concavity after multiplying by different
energy-dependent isotope form factors.
Annual signal: a common isotropic halo depends on time only through a(t), which
cannot reproduce the out-of-phase solar flux in the specified annual model.
The exposure grid is exploratory, not a continuum exposure certificate.

Exposure-bound direction
------------------------
For 5% type-I and worst-case type-II errors, the idealized pure-neutrino versus
pure-DM model has epsilon* <= log(400)/D_H. Thus the phase-only LOWER bound
D_H >= D_phase also gives epsilon* <= log(400)/D_phase, about 514 tonne-years
for D_phase=0.0116487688. It gives no 140-tonne-year lower bound on epsilon*.
This statement concerns perfect recoil/time measurement, the specified known
neutrino intensity, and no added detector backgrounds or flux nuisances.
SUPERSEDED: the paper now uses modulation.py (results/modulation.json), which
minimises the annual modulation over the exact isotropic mimics (a linear
program) and gives deviance 9 from timing alone at 3.0 t y for an ideal
0.51 keV detector and 100 t y in the released observable. The 514 t y bound
above is a loose upper bound and is no longer quoted.

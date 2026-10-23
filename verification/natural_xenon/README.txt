Run:
  python -m pip install -r requirements.txt
  python certify.py

Read PROOF.txt for statements, hypotheses and proofs. results.json contains
the certified balls and exact separating-functional coefficients.

Proved:
  * Strict natural-xenon membership is empty at threshold 0.51 keV.
  * Continuous-spectrum membership at that threshold requires m<=7.793213... GeV.
  * The proposed truncated-window sufficiency is false: m=6.0233 GeV is excluded
    at threshold 0.51 keV and cutoff 3.5 keV despite satisfying every stated
    individual endpoint-support condition.
  * At cutoff 3.5 keV, every threshold <=0.30 keV has an empty membership set.

Not proved: a nonnegative realizing halo for the truncated problem, its exact
membership set, a sharp upper edge, or its exact critical threshold.

All input decimal masses and abundances are fixed model parameters; nuclear
mass uncertainty and flux-model uncertainty are not included. The central
continuous table ends at 16.36 MeV. No floating-point exploration or LP solver
is needed to reproduce the certificate.

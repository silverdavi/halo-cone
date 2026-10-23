XENON STRUCTURAL CERTIFICATE

Read the status statement at the top of PROOF.txt first. It distinguishes
proved results, false proposed statements, numerical indications, and open
questions. In particular, nonemptiness and the lower edge on [.51,3.5] keV
remain open; an exclusion enclosure is not an existence bracket.

Run in this directory:
  python -m pip install -r requirements.txt
  python certify.py

Expected final message:
  All asserted interval tests passed; wrote results.json.

The run verifies the input table's SHA-256, uses python-flint/Arb at 192-bit
precision, and regenerates results.json from exact segment integrals. It
does not read the old results or require scipy or an optimization solver.
The original certificate's main tests are rerun under the baseline key.

Files:
  PROOF.txt       Full arguments, assumptions, corrections, and open items.
  certify.py      Standalone verifier (only python-flint beyond the stdlib).
  results.json    Generated rigorous enclosures and status metadata.
  b8_spectrum.csv Unchanged input table.
  requirements.txt Pinned python-flint dependency.

Principal results:
  All ten requested form-factor cases pass a closed-form separator.
  Moment-only and neutron-skin upper-bound robustness are proved.
  Correct endpoint and cutoff atom formulas are given.
  All m<=6.0233 GeV are excluded on [.51,3.5] keV.
  The set on [.51,5] keV is empty.
  The exact full-endpoint critical threshold is in [4.45,4.58] keV.
  The 3.5-keV-cut critical threshold has a non-sharp [.3128,3.499] bracket,
  with a continuous-spectrum realizing halo on [3.499,3.5] at m=7 GeV.

All fixed decimal physical inputs are exact model hypotheses. Only explicitly
ranged inputs (such as the 0--.3 fm skin) represent covered uncertainty.

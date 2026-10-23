#!/usr/bin/env bash
# Download the public inputs into data/ and check them against data_SHA256SUMS.
#   data/xenonnt  XENONnT light-WIMP data release, tag v0.0.1 (commit 64d2e13),
#                 Zenodo doi:10.5281/zenodo.14522707 (the release README is kept as RELEASE.md)
#   data/cresst   CRESST-III Detector A efficiencies, ancillary files of arXiv:1905.07335v3
#   data/fluxes   solar neutrino fluxes (Vinyoles et al. 2017, GS98) as normalised by
#                 C. A. J. O'Hare, github.com/cajohare/NeutrinoFog, commit 0df3d0c
set -euo pipefail
cd "$(dirname "$0")"
SUMS="$PWD/data_SHA256SUMS"
mkdir -p data/cresst data/fluxes
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

zip=light_wimp_data_release-v0.0.1.zip
curl -fsSL -o "$tmp/$zip" "https://zenodo.org/api/records/14522707/files/XENONnT/$zip/content"
(cd "$tmp" && grep " $zip\$" "$SUMS" | shasum -a 256 -c -)
unzip -q "$tmp/$zip" -d "$tmp"
rm -rf data/xenonnt
mv "$tmp/XENONnT-light_wimp_data_release-64d2e13" data/xenonnt
mv data/xenonnt/README.md data/xenonnt/RELEASE.md

for f in C3P1_DetA_AR.dat C3P1_DetA_DataRelease_SD.xy C3P1_DetA_DataRelease_SI.xy C3P1_DetA_cuteff.dat \
         C3P1_DetA_eff_AR_Ca.dat C3P1_DetA_eff_AR_O.dat C3P1_DetA_eff_AR_W.dat C3P1_DetA_full.dat; do
  curl -fsSL -o "data/cresst/$f" "https://arxiv.org/src/1905.07335v3/anc/$f"
done

nf=https://raw.githubusercontent.com/cajohare/NeutrinoFog/0df3d0cd2322602dc147a66d94055f7f64879d80/data/neutrinos/normalised
for f in 13N 15O 17F 7Be1 7Be2 8B hep pep pp; do
  curl -fsSL -o "data/fluxes/$f.txt" "$nf/$f.txt"
done

(cd data && grep -v " $zip\$" "$SUMS" | shasum -a 256 -c -)
echo "data/ ready"

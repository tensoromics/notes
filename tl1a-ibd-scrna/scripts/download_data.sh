#!/usr/bin/env bash
# Download single-cell data for the TL1A / IBD analysis.
# All files are public, no auth. Downloads are RESUMABLE (curl -C -): if a big
# pull is interrupted, just re-run this script and it continues where it stopped.
#
# Usage:
#   bash scripts/download_data.sh primary     # ~3 GB  : UC+CD Gut Cell Atlas compartments (recommended start)
#   bash scripts/download_data.sh xcheck      # ~0.3 GB: CD terminal-ileum stromal cross-check
#   bash scripts/download_data.sh ibdverse    # ~25 GB : full IBDverse TI atlas (premium CD cross-check + eQTL)
#   bash scripts/download_data.sh eqtl        # ~0.3 GB: IBDverse DE + cell-type eQTL summary stats
#   bash scripts/download_data.sh all-light   # primary + xcheck + eqtl (skips the 25 GB pull)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
get () { # url  outfile
  echo ">> $2"
  curl -L -C - -A "$UA" --fail --retry 5 --retry-delay 5 -o "$DATA/$2" "$1"
}

CXG="https://datasets.cellxgene.cziscience.com"

primary () {  # UC primary + CD, one harmonized annotation space (Gut Cell Atlas "metaplasia" integration, Nature)
  get "$CXG/e43f0ac0-a426-4471-8c4f-a9f99d612357.h5ad" "gca_mesenchymal_77k.h5ad"   # 616 MB  fibrosis arm + IAF program
  get "$CXG/48ecc234-7073-4de9-8be6-1c2f08d4c146.h5ad" "gca_myeloid_52k.h5ad"       # 373 MB  TL1A (TNFSF15) source
  get "$CXG/01db6084-3cff-4537-b52f-139dc9df50a5.h5ad" "gca_endothelial_60k.h5ad"   # 562 MB  secondary TL1A source
  get "$CXG/a7d8cf9b-2825-42ea-99f3-e96c9aee23c7.h5ad" "gca_t_nk_263k.h5ad"         # 1.47 GB DR3 inflammation arm
}

xcheck () {   # CD terminal-ileum stromal ("immune dysregulation in Crohn's disease")
  get "$CXG/34217af0-2b64-464b-9a5c-59b292938b37.h5ad" "cd_ti_stromal_75k.h5ad"     # 264 MB
}

ibdverse () { # full IBDverse terminal-ileum atlas (ArrayExpress E-MTAB-16999) — 25.5 GB, CD vs healthy, no blood
  get "https://www.ebi.ac.uk/biostudies/files/E-MTAB-16999/Full_cohort.h5ad" "ibdverse_TI_full_25gb.h5ad"
}

eqtl () {     # IBDverse DE results + cell-type eQTL summary stats (BioStudies S-BSST2944)
  get "https://www.ebi.ac.uk/biostudies/files/S-BSST2944/DGE" "ibdverse_DGE.zip"
}

case "${1:-primary}" in
  primary)    primary ;;
  xcheck)     xcheck ;;
  ibdverse)   ibdverse ;;
  eqtl)       eqtl ;;
  all-light)  primary; xcheck; eqtl ;;
  *) echo "unknown target: $1"; exit 1 ;;
esac
echo "done -> $DATA"

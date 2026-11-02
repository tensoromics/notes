#!/usr/bin/env bash
# Rebuild the seven figures for the TenK10K x ETS2 note into one folder (default: figures/).
#
#     ./make_figures.sh [out_dir]
set -euo pipefail
cd "$(dirname "$0")"
out="${1:-figures}"
mkdir -p "$out"

python fig1_caqtl_example.py   "$out/fig1-caqtl-example.png"
python fig2_ets2_locus.py      "$out/fig2-ets2-locus.png"
python fig3_smr.py             "$out/fig3-smr.png"
python fig4_search_windows.py  "$out/fig4-search-windows.png"
python fig5_caqtl_peak.py      "$out/fig5-caqtl-peak.png"
python fig6_direction_chain.py "$out/fig6-direction-chain.png"
python fig7_eqtl_distance.py   "$out/fig7-eqtl-distance.png"

"""Measure the mito% (and ribo%) distribution on the raw Smillie counts.

Answers empirically: would standard mitochondrial QC remove meaningful cells,
or is the atlas already clean? Reports median/MAD adaptive thresholds and how
many cells each would drop.
"""
import numpy as np
import scanpy as sc
from scipy.stats import median_abs_deviation

from load_smillie import load_smillie
from config import CONFIG as cfg

adata = load_smillie(cfg.data_path)

# same basic QC prepare.py does first
sc.pp.filter_cells(adata, min_genes=cfg.min_genes_per_cell)   # min_genes=200
sc.pp.filter_genes(adata, min_cells=cfg.min_cells_per_gene)   # min_cells=10
print(f"\nafter basic QC (min_genes=200, min_cells=10): {adata.n_obs} cells x {adata.n_vars} genes")

adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
print(f"MT- genes in gene set: {int(adata.var['mt'].sum())} -> {list(adata.var_names[adata.var['mt']])}")
print(f"RPS/RPL genes in gene set: {int(adata.var['ribo'].sum())}")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True,
                           percent_top=None, log1p=False)

mt = adata.obs["pct_counts_mt"].values
med, mad = float(np.median(mt)), float(median_abs_deviation(mt))
n = len(mt)
print(f"\n=== pct_counts_mt (per-cell mitochondrial fraction) ===")
print(f"median = {med:.2f}%   MAD = {mad:.2f}   mean = {mt.mean():.2f}%   max = {mt.max():.2f}%")
for p in [50, 75, 90, 95, 99, 99.9]:
    print(f"  {p:>5}th pct: {np.percentile(mt, p):.2f}%")

print("\ncells removed by threshold:")
for nmads in [3, 5]:
    thr = med + nmads * mad
    rm = int((mt > thr).sum())
    print(f"  adaptive {nmads} MADs (> {thr:.2f}%) : {rm:>6} cells  ({100*rm/n:.2f}%)")
for fixed in [8, 10, 20, 50]:
    rm = int((mt > fixed).sum())
    print(f"  fixed  > {fixed:>2}%            : {rm:>6} cells  ({100*rm/n:.2f}%)")

rb = adata.obs["pct_counts_ribo"].values
print(f"\npct_counts_ribo: median = {np.median(rb):.2f}%  (context only; RPS/RPL are excluded by HVG anyway)")

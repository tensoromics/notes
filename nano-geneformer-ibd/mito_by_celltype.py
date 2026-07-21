"""Is the high-mito tail dead cells (technical) or epithelium (biological)?

Colonic epithelium is metabolically active and genuinely high-mito, so a naive
mito filter can delete real cells. Break mito% down by cell type, and check
whether high-mito cells are low-complexity (dead) or normal (real epithelium).
"""
import numpy as np
import pandas as pd
import scanpy as sc

from load_smillie import load_smillie
from config import CONFIG as cfg

adata = load_smillie(cfg.data_path)
sc.pp.filter_cells(adata, min_genes=cfg.min_genes_per_cell)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)

obs = adata.obs.copy()
obs["pct_mt"] = adata.obs["pct_counts_mt"].values
obs["nGene"] = pd.to_numeric(obs["nGene"], errors="coerce")

# epithelial cluster keywords in Smillie's annotation
EPI = ("Enterocyte", "Goblet", "TA ", "TA1", "TA2", "Stem", "Cycling", "Tuft",
       "Enteroendocrine", "Best4", "M cell", "Immature Enterocyte", "Secretory",
       "Immature Goblet", "Enterocyte Progenitor")
obs["epi"] = obs["Cluster"].astype(str).str.startswith(EPI)

print("=== median mito% by cell-type cluster (top 20 highest) ===")
g = obs.groupby("Cluster")["pct_mt"].agg(["median", "count"]).sort_values("median", ascending=False)
print(g.head(20).to_string())
print("\n=== lowest 8 ===")
print(g.tail(8).to_string())

print("\n=== high-mito cells: dead or real? ===")
for thr in [20, 50]:
    hi = obs["pct_mt"] > thr
    epi_frac = obs.loc[hi, "epi"].mean()
    print(f"cells with mito% > {thr}% (n={int(hi.sum())}):")
    print(f"   median nGene = {obs.loc[hi,'nGene'].median():.0f}  vs  {obs.loc[~hi,'nGene'].median():.0f} for the rest")
    print(f"   fraction that are epithelial clusters = {epi_frac:.0%}")

print("\n=== mito% by compartment (epithelial vs non) ===")
print(obs.groupby("epi")["pct_mt"].agg(["median", "mean", "count"]).to_string())
print("\n=== mito% by health status ===")
print(obs.groupby("Health")["pct_mt"].median().to_string())

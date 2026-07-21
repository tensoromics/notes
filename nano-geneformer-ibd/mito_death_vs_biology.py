"""H1 (intrinsic epithelial biology) vs H2 (prep-induced epithelial death)?

Cell-type association can't tell them apart. WITHIN epithelium, test whether
mito% tracks a death/stress signature: dissociation-stress genes UP, complexity
DOWN, ribosomal DOWN => technical death (H2). Flat => intrinsic biology (H1).
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

from load_smillie import load_smillie
from config import CONFIG as cfg

ad = load_smillie(cfg.data_path)
sc.pp.filter_cells(ad, min_genes=cfg.min_genes_per_cell)
ad.var["mt"] = ad.var_names.str.startswith("MT-")
ad.var["ribo"] = ad.var_names.str.startswith(("RPS", "RPL"))
sc.pp.calculate_qc_metrics(ad, qc_vars=["mt", "ribo"], inplace=True,
                           percent_top=None, log1p=False)

EPI = ("Enterocyte", "Goblet", "TA ", "TA1", "TA2", "Stem", "Cycling TA", "Tuft",
       "Enteroendocrine", "Best4", "M cell", "Immature", "Secretory")
ad.obs["epi"] = ad.obs["Cluster"].astype(str).str.startswith(EPI)

# dissociation / stress signature (van den Brink et al. 2017 core)
stress = ["FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "ATF3", "HSPA1A", "HSPA1B",
          "HSPB1", "DNAJB1", "DNAJA1", "HSPH1", "HSP90AA1", "HSP90AB1", "NR4A1",
          "IER2", "IER3", "DUSP1", "ZFP36", "SOCS3", "BTG2"]
present = [g for g in stress if g in ad.var_names]
print(f"stress genes present: {len(present)}/{len(stress)} -> {present}")

sc.pp.normalize_total(ad, target_sum=1e4)
sc.pp.log1p(ad)
sc.tl.score_genes(ad, present, score_name="stress")

e = ad.obs[ad.obs["epi"]].copy()   # epithelial cells only
print(f"\n=== WITHIN EPITHELIUM (n={len(e)}): does mito% track death/stress? ===")
y = e["pct_counts_mt"].values
for col, label in [("n_genes_by_counts", "nGene"), ("total_counts", "total_counts"),
                   ("pct_counts_ribo", "ribo%"), ("stress", "stress-score")]:
    x = pd.to_numeric(e[col], errors="coerce").values
    m = np.isfinite(x) & np.isfinite(y)
    r, _ = spearmanr(x[m], y[m])
    print(f"  spearman(mito%, {label:13s}) = {r:+.3f}")

e["mtbin"] = pd.qcut(e["pct_counts_mt"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
print("\nepithelial cells by mito% quartile (median per bin):")
tab = e.groupby("mtbin").agg(
    mito_pct=("pct_counts_mt", "median"),
    nGene=("n_genes_by_counts", "median"),
    total_counts=("total_counts", "median"),
    ribo_pct=("pct_counts_ribo", "median"),
    stress=("stress", "median"),
    n=("pct_counts_mt", "size"))
print(tab.to_string())

print("\nread: if mito% rises with stress and falls with nGene/counts/ribo -> prep-induced death (H2);")
print("      if nGene/counts/stress are flat across mito quartiles -> intrinsic biology (H1).")

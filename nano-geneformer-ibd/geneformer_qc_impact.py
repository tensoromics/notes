"""Apply Geneformer's QC rule to the Smillie atlas and report the impact.

Geneformer (Genecorpus-30M): keep cells within 3 SD of the mean, computed
per dataset, for (a) total counts and (b) mitochondrial %. We use per-SAMPLE
as the 'dataset' unit. This is tissue-aware by construction.
"""
import numpy as np
import pandas as pd
import scanpy as sc

from load_smillie import load_smillie
from config import CONFIG as cfg

ad = load_smillie(cfg.data_path)
sc.pp.filter_cells(ad, min_genes=cfg.min_genes_per_cell)
ad.var["mt"] = ad.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(ad, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)

o = ad.obs.copy()
o["pct_mt"] = ad.obs["pct_counts_mt"].values
o["total"] = ad.obs["total_counts"].values
o["logtotal"] = np.log1p(o["total"].values)
o["sample"] = o["Sample"].astype(str)
EPI = ("Enterocyte", "Goblet", "TA ", "TA1", "TA2", "Stem", "Cycling TA", "Tuft",
       "Enteroendocrine", "Best4", "M cell", "Immature", "Secretory")
o["epi"] = o["Cluster"].astype(str).str.startswith(EPI)
n = len(o)
print(f"cells: {n}  samples: {o['sample'].nunique()}  epithelial: {int(o['epi'].sum())}")


def out3sd(x):
    m, s = x.mean(), x.std()
    return (x < m - 3 * s) | (x > m + 3 * s)


for unit, key in [("per-SAMPLE", "sample"), ("whole-DATASET", None)]:
    if key:
        mt_out = o.groupby(key)["pct_mt"].transform(out3sd)
        ct_out = o.groupby(key)["logtotal"].transform(out3sd)
    else:
        mt_out = out3sd(o["pct_mt"])
        ct_out = out3sd(o["logtotal"])
    remove = mt_out | ct_out
    print(f"\n=== Geneformer ±3 SD, {unit} ===")
    print(f"  removed total : {int(remove.sum()):>6} ({100*remove.mean():.1f}%)   -> {int((~remove).sum())} kept")
    print(f"    by mito%    : {int(mt_out.sum()):>6}   by counts: {int(ct_out.sum())}")
    print(f"  epithelial removed : {100*remove[o['epi']].mean():.1f}%   non-epi removed: {100*remove[~o['epi']].mean():.1f}%")
    kept = ~remove
    print(f"  mito% among KEPT epithelial: median {o.loc[kept & o['epi'],'pct_mt'].median():.1f}%  "
          f"max {o.loc[kept & o['epi'],'pct_mt'].max():.1f}%")
    print(f"  mito% among REMOVED cells  : median {o.loc[remove,'pct_mt'].median():.1f}%")

# what per-sample mito thresholds does 3 SD imply (sample of samples)?
print("\nper-sample implied upper mito% threshold (mean+3SD), first 8 samples:")
g = o.groupby("sample")["pct_mt"].agg(lambda x: x.mean() + 3 * x.std()).sort_values()
print(g.head(4).round(1).to_string()); print(g.tail(4).round(1).to_string())

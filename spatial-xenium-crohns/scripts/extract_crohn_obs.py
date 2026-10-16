#!/usr/bin/env python3
"""Extract only the per-cell .obs columns we need from the (9 GB) Crohn's Xenium h5ad,
using h5py so the expression matrix is never loaded. Writes a compact gzip CSV.

Crohn's ileum, 10x Xenium Prime 5k (Koplev et al., Nat Immunol 2026, SCP3259 — login-gated;
not redistributable, so this step is provenance only, not publicly reproducible).

Usage: python extract_crohn_obs.py <in.h5ad> <out.csv.gz>
"""
import sys, h5py, numpy as np, pandas as pd

IN, OUT = sys.argv[1], sys.argv[2]
NUM = ["x_centroid", "y_centroid", "cell_area", "nucleus_area",
       "transcript_counts", "control_probe_counts"]
CAT = ["cell_id", "biosample_id", "donor_id", "inflammation_status", "segmentation_method",
       "megagut_lvl1_majority_voting", "megagut_lvl3_majority_voting", "annot_new"]

f = h5py.File(IN, "r"); obs = f["obs"]
out = {}
for c in NUM:
    out[c] = obs[c][:]
for c in CAT:
    g = obs[c]
    cats = g["categories"][:]
    cats = np.array([x.decode() if isinstance(x, bytes) else x for x in cats], dtype=object)
    codes = g["codes"][:]
    vals = np.where(codes >= 0, cats[np.clip(codes, 0, len(cats) - 1)], None)
    out[c] = vals
f.close()

df = pd.DataFrame(out)
df.to_csv(OUT, index=False, compression="gzip")
print("n_cells:", len(df))
print("sections (biosample_id):", df["biosample_id"].nunique())
print("inflammation:", dict(df["inflammation_status"].value_counts()))
print("wrote", OUT)

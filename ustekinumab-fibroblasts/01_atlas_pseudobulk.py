"""Step 1a: pseudobulk the Koplev et al. Xenium atlas by fine cell type and tissue section.

Reads raw counts in row chunks (the file is about 9 GB) and writes, for every
(cell type, section) pair: summed counts per gene, number of cells with at least
one count per gene, and the number of cells.

Usage: .venv/bin/python 01_atlas_pseudobulk.py /path/to/xenium5k_scVI_CellCharter_batch2_reannot_SCP.h5ad
"""
import sys
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

H5AD = sys.argv[1]
CHUNK = 200_000

f = h5py.File(H5AD, "r")


def categorical(key):
    g = f["obs"][key]
    return pd.Categorical.from_codes(g["codes"][:], g["categories"][:].astype(str))


cell_type = categorical("annot_new")
section = categorical("biosample_id")
genes = f["var"]["_index"][:].astype(str)

group = pd.Categorical(
    [f"{t}||{s}" for t, s in zip(cell_type.astype(str), section.astype(str))]
)
n_groups, n_genes = len(group.categories), len(genes)
codes = group.codes

counts = f["layers"]["counts"]
indptr = counts["indptr"][:]
n_cells = len(indptr) - 1

sums = np.zeros((n_groups, n_genes))
detected = np.zeros((n_groups, n_genes))

for start in range(0, n_cells, CHUNK):
    stop = min(start + CHUNK, n_cells)
    lo, hi = indptr[start], indptr[stop]
    X = sp.csr_matrix(
        (counts["data"][lo:hi], counts["indices"][lo:hi], indptr[start : stop + 1] - lo),
        shape=(stop - start, n_genes),
    )
    # one row per group, one column per cell in this chunk
    G = sp.csr_matrix(
        (np.ones(stop - start), (codes[start:stop], np.arange(stop - start))),
        shape=(n_groups, stop - start),
    )
    sums += (G @ X).toarray()
    X.data = np.ones_like(X.data)
    detected += (G @ X).toarray()
    print(f"{stop:,} / {n_cells:,} cells", flush=True)

labels = pd.Series(group.categories).str.split(r"\|\|", expand=True)
index = pd.MultiIndex.from_arrays([labels[0], labels[1]], names=["cell_type", "section"])
pd.DataFrame(sums, index=index, columns=genes).to_parquet("results/atlas_pseudobulk_sums.parquet")
pd.DataFrame(detected, index=index, columns=genes).to_parquet("results/atlas_pseudobulk_detected.parquet")
pd.Series(np.bincount(codes, minlength=n_groups), index=index, name="n_cells").to_frame().to_parquet(
    "results/atlas_pseudobulk_ncells.parquet"
)
print("done")

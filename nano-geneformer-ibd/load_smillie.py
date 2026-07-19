"""Assemble the Smillie et al. 2019 UC atlas (Broad SCP259) into one AnnData.

The Single Cell Portal ships three compartments (Epi/Fib/Imm), each a
`genes x cells` MatrixMarket matrix with its own genes/barcodes lists, plus
metadata tables whose first data row is a "TYPE" descriptor to skip and whose
first column (NAME) is the cell barcode.

- `Smillie_meta2.txt` : base per-cell metadata (Subject, Health, Cluster, ...).
- `all.meta2.txt` / `all.meta.txt` : richer per-cell metadata (batch, sex, ...);
  merged automatically when a VALID copy is present. A copy that is actually an
  XML error response (expired download link) is detected and skipped.

Cells are disjoint across compartments; we concatenate on the cell axis and keep
the gene intersection (inner join) as the shared vocabulary.
"""
import os

import anndata as ad
import pandas as pd
import scipy.io
import scipy.sparse as sp

COMPARTMENTS = ["Epi", "Fib", "Imm"]
EXTRA_META = ["all.meta2.txt", "all.meta.txt"]   # merged if valid (batch/sex/...)


def _load_compartment(folder, c):
    genes = pd.read_csv(os.path.join(folder, f"{c}.genes.tsv"), header=None)[0].values
    barcodes = pd.read_csv(os.path.join(folder, f"{c}.barcodes2.tsv"), header=None)[0].values
    print(f"  loading {c}: reading matrix (slow for large mtx) ...")
    X = scipy.io.mmread(os.path.join(folder, f"gene_sorted-{c}.matrix.mtx"))  # genes x cells
    X = sp.csr_matrix(X).T.tocsr()                                            # cells x genes
    if X.shape != (len(barcodes), len(genes)):
        raise SystemExit(
            f"{c}: matrix {X.shape} != {len(barcodes)} barcodes x {len(genes)} genes. "
            f"File may be truncated (check the download)."
        )
    a = ad.AnnData(X=X)
    a.obs_names = barcodes
    a.var_names = genes
    a.var_names_make_unique()
    return a


def _read_scp_meta(path):
    """Read an SCP metadata TSV (barcode in col 0, optional 'TYPE' row skipped).
    Returns a DataFrame indexed by barcode, or None if the file is an error page."""
    with open(path, "r", errors="replace") as f:
        head = f.read(256)
    if head.lstrip()[:5] == "<?xml" or "<Error" in head:
        print(f"  [warn] {os.path.basename(path)} is an XML error response "
              f"(expired download link), not metadata -- skipping it.")
        return None
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    if len(df) and str(df.iloc[0, 0]).strip().upper() == "TYPE":
        df = df.iloc[1:]                      # drop SCP group/numeric descriptor row
    return df.set_index(df.columns[0])


def load_smillie(folder, meta_file="Smillie_meta2.txt"):
    parts = []
    for c in COMPARTMENTS:
        if os.path.exists(os.path.join(folder, f"gene_sorted-{c}.matrix.mtx")):
            parts.append(_load_compartment(folder, c))
        else:
            print(f"  [skip] {c}: matrix not found (still downloading / failed?)")
    if not parts:
        raise SystemExit(f"No compartment matrices found in {folder}")

    adata = parts[0] if len(parts) == 1 else ad.concat(parts, join="inner", axis=0)
    print(f"  combined: {adata.n_obs} cells x {adata.n_vars} genes "
          f"({len(parts)}/{len(COMPARTMENTS)} compartments, gene intersection)")

    base = _read_scp_meta(os.path.join(folder, meta_file))
    if base is None:
        raise SystemExit(f"{meta_file} is unreadable/expired -- re-download it.")
    common = adata.obs_names.intersection(base.index)
    if len(common) == 0:
        raise SystemExit("No barcode overlap between matrices and metadata.")
    adata = adata[common].copy()
    adata.obs = base.reindex(adata.obs_names).copy()

    # merge richer covariates (batch, sex, ...) from all.meta*.txt if a valid copy exists
    for name in EXTRA_META:
        p = os.path.join(folder, name)
        if not os.path.exists(p):
            continue
        extra = _read_scp_meta(p)
        if extra is None:
            print(f"  note: {name} present but unusable -- re-download it to add "
                  f"batch/sex covariates.")
            continue
        extra = extra.reindex(adata.obs_names)
        new_cols = [c for c in extra.columns if c not in adata.obs.columns]
        for c in new_cols:
            adata.obs[c] = extra[c].values
        if new_cols:
            print(f"  merged extra metadata from {name}: {new_cols}")
        break

    print(f"  final: {adata.n_obs} cells x {adata.n_vars} genes; "
          f"obs columns = {list(adata.obs.columns)}")
    return adata


if __name__ == "__main__":
    import sys
    a = load_smillie(sys.argv[1] if len(sys.argv) > 1 else "smillie")
    print(a)

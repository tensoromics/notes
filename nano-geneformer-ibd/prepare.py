"""Turn an scRNA-seq AnnData into rank-encoded token sequences (Geneformer-style).

Pipeline:
  load -> QC filter -> CP10k normalize -> pick top-N HVGs -> per-gene nonzero
  median scaling -> rank genes within each cell -> token sequence (top max_len).
Also saves a 50-dim PCA of the log-normalized data as the linear-probe baseline,
and a deterministic patient-level train/val split.

Usage:
  python prepare.py --data /path/to/file.h5ad
  python prepare.py --data /path/to/10x_mtx_dir --patient-col Subject --label-col Health
"""
import argparse
import json
import os

import numpy as np
import scanpy as sc

from config import CONFIG


def load_adata(path):
    if os.path.isdir(path):
        # Smillie SCP259 layout: three compartment matrices + shared metadata
        if os.path.exists(os.path.join(path, "gene_sorted-Epi.matrix.mtx")):
            from load_smillie import load_smillie
            return load_smillie(path)
        return sc.read_10x_mtx(path, var_names="gene_symbols")
    if path.endswith(".h5ad"):
        return sc.read_h5ad(path)
    if path.endswith(".h5"):
        return sc.read_10x_h5(path)
    if path.endswith(".loom"):
        return sc.read_loom(path)
    if path.endswith((".csv", ".tsv", ".txt")):
        # assumes genes x cells; transpose to cells x genes. Adjust if yours differs.
        return sc.read_csv(path, delimiter="\t" if not path.endswith(".csv") else ",").T
    raise ValueError(f"Unrecognized data format: {path}")


def require_col(adata, col, kind):
    if col not in adata.obs.columns:
        raise SystemExit(
            f"\n[config error] {kind} column '{col}' not in your file.\n"
            f"Your obs columns are:\n  {list(adata.obs.columns)}\n"
            f"Edit config.py (or pass --{kind.replace('_', '-')}) to match.\n"
        )


def main():
    cfg = CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cfg.data_path)
    ap.add_argument("--out", default=cfg.out_dir)
    ap.add_argument("--patient-col", default=cfg.patient_col)
    ap.add_argument("--label-col", default=cfg.label_col)
    ap.add_argument("--celltype-col", default=cfg.celltype_col)
    ap.add_argument("--n-hvg", type=int, default=cfg.n_hvg)
    ap.add_argument("--max-len", type=int, default=cfg.max_len)
    ap.add_argument("--subsample", type=int, default=0,
                    help="randomly keep N cells (0 = all) for a fast test run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.data} ...")
    adata = load_adata(args.data)
    adata.var_names_make_unique()
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    if args.subsample and adata.n_obs > args.subsample:
        sc.pp.subsample(adata, n_obs=args.subsample, random_state=cfg.seed)
        print(f"  subsampled to {adata.n_obs} cells (fast test run)")

    require_col(adata, args.patient_col, "patient_col")
    require_col(adata, args.label_col, "label_col")

    # ---- QC ------------------------------------------------------------------
    sc.pp.filter_cells(adata, min_genes=cfg.min_genes_per_cell)
    sc.pp.filter_genes(adata, min_cells=cfg.min_cells_per_gene)
    print(f"  after QC: {adata.n_obs} cells x {adata.n_vars} genes")

    # keep raw counts, normalize to CP10k (non-log) for rank encoding
    sc.pp.normalize_total(adata, target_sum=1e4)

    # ---- PCA baseline (log-normalized) for the probe -------------------------
    adata_log = adata.copy()
    sc.pp.log1p(adata_log)
    sc.pp.highly_variable_genes(adata_log, n_top_genes=args.n_hvg)
    sc.pp.pca(adata_log, n_comps=50, use_highly_variable=True)
    X_pca = np.asarray(adata_log.obsm["X_pca"], dtype=np.float32)

    # subset both to the same HVGs (vocab)
    hvg_mask = adata_log.var["highly_variable"].values
    adata = adata[:, hvg_mask].copy()
    gene_names = list(adata.var_names)
    n_genes = len(gene_names)
    print(f"  vocab: {n_genes} HVGs (+{cfg.N_SPECIAL} special tokens)")

    # ---- per-gene nonzero median (Geneformer normalization) ------------------
    Xcsc = adata.X.tocsc()
    gene_median = np.ones(n_genes, dtype=np.float32)
    for j in range(n_genes):
        col = Xcsc.data[Xcsc.indptr[j]:Xcsc.indptr[j + 1]]
        if col.size:
            gene_median[j] = np.median(col)
    gene_median[gene_median == 0] = 1.0

    # ---- rank-encode each cell -> token sequence -----------------------------
    Xcsr = adata.X.tocsr()
    n_cells = adata.n_obs
    tokens = np.zeros((n_cells, args.max_len), dtype=np.int64)  # PAD=0
    for i in range(n_cells):
        lo, hi = Xcsr.indptr[i], Xcsr.indptr[i + 1]
        idx = Xcsr.indices[lo:hi]
        vals = Xcsr.data[lo:hi] / gene_median[idx]
        order = np.argsort(-vals)[:args.max_len]
        toks = idx[order] + cfg.N_SPECIAL  # gene col index -> token id
        tokens[i, :toks.size] = toks
        if (i + 1) % 20000 == 0:
            print(f"    encoded {i + 1}/{n_cells}")

    # ---- deterministic patient-level split -----------------------------------
    patients = adata.obs[args.patient_col].astype(str).values
    uniq = sorted(set(patients))
    val_patients = set(uniq[:: cfg.val_every])
    split = np.array(["val" if p in val_patients else "train" for p in patients])
    print(f"  patients: {len(uniq)} total, {len(val_patients)} held out for val")
    print(f"  cells: {int((split == 'train').sum())} train / {int((split == 'val').sum())} val")

    labels = adata.obs[args.label_col].astype(str).values
    celltypes = (adata.obs[args.celltype_col].astype(str).values
                 if args.celltype_col in adata.obs.columns
                 else np.array(["NA"] * n_cells))

    # ---- save ----------------------------------------------------------------
    np.save(os.path.join(args.out, "tokens.npy"), tokens)
    np.save(os.path.join(args.out, "pca.npy"), X_pca)
    np.save(os.path.join(args.out, "gene_median.npy"), gene_median)
    np.savez(os.path.join(args.out, "meta.npz"),
             patient=patients, label=labels, celltype=celltypes, split=split)
    vocab = {
        "pad": cfg.PAD, "mask": cfg.MASK, "n_special": cfg.N_SPECIAL,
        "gene_to_id": {g: i + cfg.N_SPECIAL for i, g in enumerate(gene_names)},
        "id_to_gene": {i + cfg.N_SPECIAL: g for i, g in enumerate(gene_names)},
        "vocab_size": n_genes + cfg.N_SPECIAL,
        "max_len": args.max_len,
    }
    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    print(f"\nDone. Wrote tokens/pca/meta/vocab to '{args.out}/'.")
    print(f"Label classes: {sorted(set(labels))}")


if __name__ == "__main__":
    main()

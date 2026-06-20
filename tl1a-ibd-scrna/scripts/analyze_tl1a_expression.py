#!/usr/bin/env python3
"""
TL1A axis single-cell expression — starter analysis.

For a cellxgene compartment .h5ad (downloaded via scripts/download_data.sh), this maps
the TL1A--DR3 axis across cell types and disease state:
  - TL1A  = TNFSF15  (the ligand; source = myeloid)
  - DR3   = TNFRSF25 (the receptor; effector = Th17/ILC3; fibrosis = fibroblasts)
  - IAF signature (inflammation-associated fibroblast; anti-TNF-resistance program)

Outputs a per-(cell_type x disease) table (mean expression + % cells) and a dotplot.

Usage:
    python scripts/analyze_tl1a_expression.py --input data/gca_mesenchymal_77k.h5ad
    python scripts/analyze_tl1a_expression.py --input data/gca_myeloid_52k.h5ad --outdir figures

Requires the scanpy stack (requirements.txt) in a Python 3.12 venv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc

TL1A_AXIS = ["TNFSF15", "TNFRSF25"]                       # ligand, receptor
IAF = ["IL13RA2", "IL11", "OSMR", "TNFRSF11B", "CHI3L1", "IL24"]  # inflammatory-fibroblast / anti-TNF-resistance
GENES = TL1A_AXIS + IAF


def resolve_symbols(adata):
    """cellxgene var is indexed by Ensembl id with a feature_name (symbol) column.
    Return {symbol: var_index} for genes present."""
    if "feature_name" in adata.var.columns:
        sym = adata.var["feature_name"].astype(str)
    else:
        sym = pd.Series(adata.var_names, index=adata.var_names)  # already symbols?
    lookup = {}
    for g in GENES:
        hits = sym.index[sym.values == g]
        if len(hits):
            lookup[g] = hits[0]
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to a cellxgene compartment .h5ad")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--celltype-col", default="cell_type")
    ap.add_argument("--disease-col", default="disease")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"input not found: {args.input}\nrun: bash scripts/download_data.sh primary")
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"loading {args.input} ...")
    adata = sc.read_h5ad(args.input)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    print(f"  obs columns: {list(adata.obs.columns)[:12]}...")

    # use raw counts -> log-normalize for comparable expression, if raw is present
    if adata.raw is not None and adata.raw.n_vars >= adata.n_vars:
        adata = adata.raw.to_adata()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    lookup = resolve_symbols(adata)
    present = list(lookup.keys())
    missing = [g for g in GENES if g not in lookup]
    print(f"  genes present: {present}")
    if missing:
        print(f"  genes absent (not expressed/measured here): {missing}")
    if not present:
        sys.exit("none of the TL1A-axis genes are in this object")

    ct = args.celltype_col if args.celltype_col in adata.obs else None
    dis = args.disease_col if args.disease_col in adata.obs else None
    if ct is None:
        sys.exit(f"no '{args.celltype_col}' column in obs; pass --celltype-col")
    group = [ct] + ([dis] if dis else [])

    # per (cell_type[, disease]): mean log-norm expression + % cells expressing
    rows = []
    X = adata[:, [lookup[g] for g in present]].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    expr = pd.DataFrame(X, columns=present, index=adata.obs_names)
    meta = adata.obs[group].copy()
    for keys, idx in expr.groupby([meta[c].values for c in group]).groups.items():
        sub = expr.loc[idx]
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group, keys))
        rec["n_cells"] = len(sub)
        for g in present:
            rec[f"{g}_mean"] = round(float(sub[g].mean()), 4)
            rec[f"{g}_pct"] = round(float((sub[g] > 0).mean() * 100), 1)
        rows.append(rec)
    table = pd.DataFrame(rows).sort_values(group)
    out_csv = os.path.join("results", "tl1a_axis_by_celltype.csv")
    table.to_csv(out_csv, index=False)
    print(f"\nsaved table -> {out_csv}  ({len(table)} groups)")
    cols = group + ["n_cells"] + [f"{g}_pct" for g in present]
    print(table[cols].head(25).to_string(index=False))

    # IAF signature score (if enough IAF genes present)
    iaf_present = [g for g in IAF if g in lookup]
    if len(iaf_present) >= 3:
        sc.tl.score_genes(adata, [lookup[g] for g in iaf_present], score_name="IAF_score")
        print("\nIAF signature score by cell type (top):")
        print(adata.obs.groupby(ct, observed=True)["IAF_score"].mean()
              .sort_values(ascending=False).head(10).round(3).to_string())

    # dotplot
    try:
        adata.var["symbol"] = adata.var.get("feature_name", pd.Series(adata.var_names, index=adata.var_names))
        ad = adata[:, [lookup[g] for g in present]].copy()
        ad.var_names = present
        fig_path = os.path.join(args.outdir, "tl1a_axis_dotplot.png")
        sc.pl.dotplot(ad, present, groupby=ct, show=False, save=None)
        import matplotlib.pyplot as plt
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nsaved dotplot -> {fig_path}")
    except Exception as e:
        print(f"(dotplot skipped: {e})")


if __name__ == "__main__":
    main()

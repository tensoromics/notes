"""Where published inflammatory-fibroblast markers sit in the Koplev Xenium atlas.

For each gene: is it on the Xenium panel and on the UNITI array, and how does it fare
against each part of the Amendment 1 rule used to pick the 8-gene signature.
Usage: python 09_published_markers.py GENE [GENE ...]
"""
import sys
import pandas as pd

from common import probe_to_symbol

IF = "M COL7A1hi inflammatory fibroblast"
sums = pd.read_parquet("results/atlas_pseudobulk_sums.parquet")
detected = pd.read_parquet("results/atlas_pseudobulk_detected.parquet")
ncells = pd.read_parquet("results/atlas_pseudobulk_ncells.parquet")["n_cells"]

# the same steps as 02_atlas_signatures.py, without restricting to genes on the array
keep = ncells >= 20
sums, detected, ncells = sums[keep], detected[keep], ncells[keep]
cpm = sums.div(sums.sum(axis=1), axis=0) * 1e6
mean_cpm = cpm.groupby(level="cell_type").mean()
cells_per_type = ncells.groupby(level="cell_type").sum()
det_frac = detected.groupby(level="cell_type").sum().div(cells_per_type, axis=0)
usable = [t for t in mean_cpm.index if cells_per_type[t] >= 1000 and t != "nan"]
fib = [t for t in usable if "fibroblast" in t and t != IF]
nonfib = [t for t in usable if "fibroblast" not in t]
on_array = set(probe_to_symbol().values)

rows = []
for g in sys.argv[1:]:
    if g not in mean_cpm.columns:
        rows.append({"gene": g, "panel": "no"})
        continue
    x = mean_cpm.loc[IF, g]
    top_fib, top_non = mean_cpm.loc[fib, g].idxmax(), mean_cpm.loc[nonfib, g].idxmax()
    r_fib = x / max(mean_cpm.loc[top_fib, g], 1e-3)
    r_non = x / max(mean_cpm.loc[top_non, g], 1e-3)
    det = det_frac.loc[IF, g]
    fails = [name for name, ok in [("cpm<100", x >= 100), ("detected<5%", det >= 0.05),
                                   ("vs fibroblasts<1.5", r_fib >= 1.5), ("vs non-fibroblasts<2", r_non >= 2)] if not ok]
    rank_fib = mean_cpm.loc[usable, g].rank(ascending=False)[IF]
    rows.append({"gene": g, "panel": "yes", "array": "yes" if g in on_array else "no",
                 "IF_cpm": round(x), "detected": round(det, 2), "rank_all_types": int(rank_fib),
                 "vs_fib": round(r_fib, 2), "top_other_fib": top_fib.replace(" fibroblast", ""),
                 "vs_nonfib": round(r_non, 2), "top_nonfib": top_non,
                 "result": "in signature" if not fails else "fails: " + ", ".join(fails)})
pd.set_option("display.width", 250, "display.max_columns", None, "display.max_colwidth", 40)
print(pd.DataFrame(rows).set_index("gene").fillna("").to_string())

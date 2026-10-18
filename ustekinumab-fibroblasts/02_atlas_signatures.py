"""Step 1b: pick signature genes from the atlas pseudobulk.

Amended rule (see "Amendment 1" in PLAN.md; fixed before any trial outcome was used).
A gene enters the inflammatory fibroblast (IF) signature if, in the atlas:
  - IF mean expression is at least 100 CPM and the gene is detected in at least 5% of IFs
  - IF expression is at least 1.5x every other fibroblast type
  - IF expression is at least 2x every non-fibroblast cell type
Cell types with fewer than 1,000 cells, and unlabeled cells, are left out of the
comparison because their averages are noisy.

Comparison scores use the same thresholds:
  pan-fibroblast: median over fibroblast types (IF included) vs every non-fibroblast type
  inflammatory myeloid: the lower of Monocyte and CD14hi macrophage vs every non-myeloid
    type (the atlas has no neutrophil label)
"""
import json
import pandas as pd

from common import probe_to_symbol

MIN_CELLS_PER_SECTION = 20
MIN_CELLS_PER_TYPE = 1000
IF = "M COL7A1hi inflammatory fibroblast"

sums = pd.read_parquet("results/atlas_pseudobulk_sums.parquet")
detected = pd.read_parquet("results/atlas_pseudobulk_detected.parquet")
ncells = pd.read_parquet("results/atlas_pseudobulk_ncells.parquet")["n_cells"]

keep = ncells >= MIN_CELLS_PER_SECTION
sums, detected, ncells = sums[keep], detected[keep], ncells[keep]
cpm = sums.div(sums.sum(axis=1), axis=0) * 1e6

# each section counts once per cell type
mean_cpm = cpm.groupby(level="cell_type").mean()
cells_per_type = ncells.groupby(level="cell_type").sum()
det_frac = detected.groupby(level="cell_type").sum().div(cells_per_type, axis=0)

# genes on the array, by current HGNC symbol (the array annotation uses some old names)
array_genes = set(probe_to_symbol().values)
mean_cpm = mean_cpm.loc[:, mean_cpm.columns.isin(list(array_genes))]

usable = [t for t in mean_cpm.index if cells_per_type[t] >= MIN_CELLS_PER_TYPE and t != "nan"]
fib = [t for t in usable if "fibroblast" in t]
nonfib = [t for t in usable if t not in fib]
print(f"{len(usable)} cell types used ({len(fib)} fibroblast); {mean_cpm.shape[1]} genes on the array")


def select(target_expr, target_det, ref_sets, label):
    """Genes whose target expression beats every type in each reference set by its fold."""
    table = pd.DataFrame({"target_cpm": target_expr, "detected": target_det})
    ok = (table["target_cpm"] >= 100) & (table["detected"] >= 0.05)
    for name, (types, fold) in ref_sets.items():
        ref = mean_cpm.loc[types]
        table[f"vs_{name}"] = target_expr / ref.max().clip(lower=1e-3)
        table[f"next_{name}"] = ref.idxmax()
        ok &= table[f"vs_{name}"] >= fold
    first = f"vs_{next(iter(ref_sets))}"
    chosen = table[ok].sort_values(first, ascending=False)
    print(f"\n{label}: {len(chosen)} genes")
    print(chosen.round(2).to_string())
    return chosen, table


if_sel, if_table = select(
    mean_cpm.loc[IF], det_frac.loc[IF],
    {"other_fibroblasts": ([t for t in fib if t != IF], 1.5), "non_fibroblasts": (nonfib, 2.0)},
    "Inflammatory fibroblast",
)
fib_sel, _ = select(
    mean_cpm.loc[fib].median(), detected.loc[fib].sum().div(cells_per_type[fib].sum()),
    {"non_fibroblasts": (nonfib, 2.0)},
    "Pan-fibroblast",
)
# monocytes and macrophages share most genes, so compare against non-myeloid cells only
my = ["Monocyte", "CD14hi macrophage"]
myeloid = [t for t in usable if "macrophage" in t.lower() or t.endswith(" DC")
           or t.startswith("DC_") or t in ("Monocyte", "Mast")]
print("myeloid types:", myeloid)
my_sel, _ = select(
    mean_cpm.loc[my].min(), detected.loc[my].sum().div(cells_per_type[my].sum()),
    {"non_myeloid": ([t for t in usable if t not in myeloid], 2.0)},
    "Inflammatory myeloid",
)

# sensitivity versions of the IF signature
strict = if_table[(if_table.target_cpm >= 100) & (if_table.detected >= 0.05)
                  & (if_table.vs_other_fibroblasts >= 2) & (if_table.vs_non_fibroblasts >= 2)]
loose = if_table[(if_table.target_cpm >= 100) & (if_table.detected >= 0.05)
                 & (if_table.vs_other_fibroblasts >= 1.25) & (if_table.vs_non_fibroblasts >= 1.5)]
loose = loose.sort_values("vs_other_fibroblasts", ascending=False)
print(f"\nstrict IF ({len(strict)}): {list(strict.index)}")
print(f"loose IF ({len(loose)}): {list(loose.index)}")

signatures = {
    "frozen": "2026-10-17",
    "rule": "amendment 1 in PLAN.md",
    "IF": list(if_sel.index),
    "IF_strict": list(strict.sort_values("vs_other_fibroblasts", ascending=False).index),
    "IF_loose": list(loose.index),
    "pan_fibroblast": list(fib_sel.index[:25]),
    "inflammatory_myeloid": list(my_sel.index[:25]),
}
json.dump(signatures, open("results/signatures.json", "w"), indent=1)
if_table.sort_values("vs_other_fibroblasts", ascending=False).to_csv("results/if_gene_table.csv")
print("\n" + json.dumps(signatures, indent=1))

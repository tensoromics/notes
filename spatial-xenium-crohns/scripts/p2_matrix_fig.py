#!/usr/bin/env python3
"""Part 2: what the authors shared, for the same cells: the cell table and the count matrix.

Left: the cell table (center, area, cell type), a few of the rows. Right: the count matrix.

Rows: 30 epithelial cells and 30 T cells, picked at random from one inflamed section.
Columns: EPCAM and CD3E, then 98 other genes picked at random and sorted by how often they are
detected. Each square is one gene in one cell; white means zero dots.
Prints the size of the full matrix and its share of zeros.

Usage: python p2_matrix_fig.py [h5ad] [out_dir]      -> p2_matrix.png
"""
import sys, os
import numpy as np
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

H5AD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "xenium5k_scVI_CellCharter_batch2_reannot_SCP.h5ad")
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else
                      os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(OUT, exist_ok=True)
INK, MUTED, EPI, TCELL = "#1b2024", "#6b727a", "#1C8577", "#7A5DA9"
rng = np.random.default_rng(1)

f = h5py.File(H5AD, "r")
obs, X = f["obs"], f["layers"]["counts"]
def column(name):
    g = obs[name]
    return g[:] if isinstance(g, h5py.Dataset) else g["categories"].asstr()[:][g["codes"][:]]
section, types = column("biosample_id"), column("megagut_lvl1_majority_voting")
xc, yc, area = column("x_centroid"), column("y_centroid"), column("cell_area")
genes = f["var"]["_index"].asstr()[:]
indptr = X["indptr"][:]
n_cells, n_genes, nnz = len(indptr) - 1, len(genes), int(indptr[-1])

# one inflamed section, 30 cells of each type
inflamed = sorted({s for s in set(section) if "inf" in s.lower()})[0]
pick = lambda kind: rng.choice(np.flatnonzero((section == inflamed) & (types == kind)), 30, replace=False)
rows = np.r_[np.sort(pick("Epithelial")), np.sort(pick("T and NK cells"))]

# counts of every gene in the chosen cells
M = np.zeros((len(rows), n_genes), dtype=np.int32)
for i, c in enumerate(rows):
    a, b = indptr[c], indptr[c + 1]
    M[i, X["indices"][a:b]] = X["data"][a:b]

# columns: EPCAM, CD3E, then 98 random other genes, most often detected first
first = [int(np.flatnonzero(genes == g)[0]) for g in ("EPCAM", "CD3E")]
others = rng.choice(np.setdiff1d(np.arange(n_genes), first), 98, replace=False)
others = others[np.argsort(-(M[:, others] > 0).sum(axis=0), kind="stable")]
cols = np.r_[first, others]
S = M[:, cols]

print(f"full matrix: {n_cells:,} cells x {n_genes:,} genes = {n_cells * n_genes:,} entries")
print(f"nonzero: {nnz:,} ({nnz / (n_cells * n_genes):.1%}); zeros: {1 - nnz / (n_cells * n_genes):.1%}")
print(f"this corner: {S.shape}, nonzero {(S > 0).mean():.1%}, max {S.max()}")
print("EPCAM in the 30 T cells:", S[30:, 0].tolist())

# draw: 0 is white, 1 / 2 / 3+ dots get darker; EPCAM and CD3E get wider columns on the left
cmap = ListedColormap(["white", "#c9ced3", "#8a929a", INK])
fig = plt.figure(figsize=(14, 5.8))
gs = fig.add_gridspec(1, 4, width_ratios=[100, 7, 16, 98], wspace=0.015)
at, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3])
for ax, part in ((a1, S[:, :2]), (a2, S[:, 2:])):
    ax.imshow(np.minimum(part, 3), cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
    ax.axhline(29.5, color=INK, lw=1.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_color("#c9ced3")
a1.axvline(0.5, color="#c9ced3", lw=0.8)
a1.set_yticks([0, 30, 59], ["row 1", "row 31", "row 60"], fontsize=10, color=MUTED)
a1.set_xticks([0, 1], ["EPCAM", "CD3E"], rotation=90, fontsize=11.5)
for t, c in zip(a1.get_xticklabels(), (EPI, TCELL)):
    t.set_color(c); t.set_fontweight("bold")
a2.set_yticks([])
a2.set_xticks([])
a2.set_xlabel("98 other genes, picked at random,\nmost often detected on the left", fontsize=11.5,
              color=MUTED, labelpad=8)
# how much more there is
a2.text(1.012, 0.5, f"+ {n_genes - 100:,}\nmore genes\n→", transform=a2.transAxes, ha="left",
        va="center", fontsize=11.5, color=INK, linespacing=1.4)
fig.text(0.53, -0.04, f"↓  + {n_cells - 60:,} more cells", ha="center", va="top", fontsize=11.5, color=INK)
# the cell table: the same 60 cells, in the same order; we print a few rows of each type
at.set_ylim(len(rows) - 0.5, -0.5); at.set_xlim(0, 1); at.axis("off")
heads, xs = ["row", "x (µm)", "y (µm)", "area (µm²)", "cell type"], [0.0, 0.11, 0.31, 0.51, 0.74]
for h, x in zip(heads, xs):
    at.text(x, -1.6, h, ha="left", va="bottom", fontsize=12, color=INK, fontweight="bold")
def table_row(i, y):
    c = rows[i]
    vals = [i + 1, f"{xc[c]:,.1f}", f"{yc[c]:,.1f}", f"{area[c]:.1f}", "epithelial" if i < 30 else "T cell"]
    for v, x in zip(vals, xs):
        at.text(x, y, str(v), ha="left", va="center", fontsize=12, color=INK)
for k, i in enumerate(range(6)):
    table_row(i, 1.0 + 2.4 * k)
at.text(0.4, 16.8, "⋮", ha="center", va="center", fontsize=14, color=MUTED)
for k, i in enumerate(range(30, 36)):
    table_row(i, 31.0 + 2.4 * k)
at.text(0.4, 46.8, "⋮", ha="center", va="center", fontsize=14, color=MUTED)
at.axhline(29.5, xmin=0, xmax=1, color=INK, lw=1.2)
at.set_title("Cell table", fontsize=14, color=INK, loc="left", pad=30)
at.text(0.0, len(rows) + 3.2, "Not shared: the dots (Figure 1A)\nand the outlines (Figure 1B)",
        ha="left", va="top", fontsize=11.5, color=MUTED)

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#c9ced3") for c in cmap.colors]
a2.legend(handles, ["0 dots", "1", "2", "3 or more"], ncol=4, frameon=False, fontsize=11,
          loc="lower right", bbox_to_anchor=(1.0, 1.0))
a1.set_title("Count matrix: one column per gene", fontsize=14, color=INK, loc="left", pad=30)
fig.savefig(os.path.join(OUT, "p2_matrix.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", os.path.join(OUT, "p2_matrix.png"))

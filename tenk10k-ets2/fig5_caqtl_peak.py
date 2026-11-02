"""Figure 5 for the note "TenK10K meets ETS2": the ETS2 enhancer peak across the cell types TenK10K tested.

Data: TenK10K posted caQTL lead table (Hugging Face anglixue/TenK10K_multiome,
caQTL_common_TensorQTL/TenK10K.caQTL.1Mb.final_summary_qval005_feb02_2026_freeze.csv). It lists 26 tested cell types;
peak chr21:39094096-39094839 is a significant caQTL (q < 0.05) in three of them:
    CD14_Mono  lead 21:39092834:CCTGTAATCCCA:C  slope -0.9527 per alt allele (r2 = 1.0 with rs2836882)
    CD16_Mono  lead 21:39092834:CCTGTAATCCCA:C  slope -0.6472
    cDC2       lead 21:39096912:G:A             slope -0.3468 (r2 = 0.87 with rs2836882, Xue Supp Note 2)
rs2836882's own effect (SMR files, per A allele): CD14 -0.952709, CD16 -0.647235. In cDC2 the linked SNP's A allele is
the protective one (b_GWAS -0.1371) and closes the peak, so its risk-side allele opens it. Values are plotted per copy
of the risk-side allele (signs flipped). "No caQTL detected" = not significant or not tested in that cell type.

    python fig5_caqtl_peak.py <out.png>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
VIOLET = "#7c3aed"
LABEL_BOX = dict(facecolor="white", edgecolor="none", pad=1.5)

# label, effect per copy of the risk-side allele (None = no caQTL detected), note
ROWS = [
    ("CD14 monocytes", 0.952709, "rs2836882"),
    ("CD16 monocytes", 0.647235, "rs2836882"),
    ("cDC2 dendritic cells", 0.346825, "linked SNP, r² = 0.87"),
    ("Other myeloid: cDC1, pDC, ASDC", None, ""),
    ("T cells (12 types)", None, ""),
    ("B cells and plasmablasts (4 types)", None, ""),
    ("NK cells (3 types)", None, ""),
    ("Progenitors (HSPC)", None, ""),
]

out = sys.argv[1]
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)

n = len(ROWS)
ys = list(range(n, 0, -1))
for (label, val, note), y in zip(ROWS, ys):
    if val is None:
        ax.text(0.02, y, "no caQTL detected", ha="left", va="center", fontsize=10, color=MUTED)
        continue
    ax.plot([0, val], [y, y], color=VIOLET, lw=2, solid_capstyle="butt", zorder=2)
    ax.plot([val], [y], "o", markersize=10, color=VIOLET, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    ax.text(val + 0.025, y, f"+{val:.2f}  ({note})", ha="left", va="center", fontsize=10.5, color=INK,
            zorder=4, bbox=LABEL_BOX)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=10.5, color=INK)
ax.tick_params(axis="y", length=0)

ax.set_xlim(0, 1.3)
ax.set_ylim(0.4, n + 0.6)
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="x", colors=MUTED, labelsize=10, labelcolor=INK)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax.set_xlabel("change in openness per copy of the risk-side allele (normalized units)", fontsize=10.5, color=INK)

fig.suptitle("The ETS2 enhancer peak is a caQTL in 3 of the 26 cell types TenK10K tested", fontsize=13, color=INK,
             x=0.01, ha="left", y=0.985)
fig.text(0.01, 0.9, "Peak chr21:39,094,096–39,094,839 (743 bp), with rs2836882 inside it. It counts as a caQTL only in "
         "cell types where its openness depends on genotype.", fontsize=10, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)

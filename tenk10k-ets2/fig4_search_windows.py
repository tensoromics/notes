"""Figure 4 for the note "TenK10K meets ETS2": the two eQTL search windows around ETS2, and where rs2836882 falls.

Coordinates are GRCh38 from the Ensembl REST API (lookup/symbol, queried 2026-10-31). The TenK10K expression atlas
(Cuomo et al., Methods) tests variants within 100 kb of each gene body; Xue et al. also colocalize at +/-1 Mb, where the
ETS2 eQTL colocalizes with IBD at PP.H4 = 0.996 (Supplementary Note 2, Extended Data Fig. 8a).

    python fig4_search_windows.py <out.png>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#0f172a"
MUTED = "#64748b"
GENE_GRAY = "#94a3b8"
TEAL = "#0d9488"
VIOLET = "#7c3aed"

ETS2_START, ETS2_END = 38_805_115, 38_824_956  # + strand
SNP = 39_094_644  # rs2836882

XMIN, XMAX = 38.70, 39.36
# White box behind window labels so the dotted SNP guide passes under the text
LABEL_BOX = dict(facecolor="white", edgecolor="none", pad=1.5)
mb = lambda bp: bp / 1e6

out = sys.argv[1]
fig, ax = plt.subplots(figsize=(11, 3.4), dpi=200)

# Track at y = 0: ETS2 and the SNP only
ax.axhline(0, color=GENE_GRAY, lw=0.8, zorder=1)
ax.add_patch(plt.Rectangle((mb(ETS2_START), -0.09), mb(ETS2_END) - mb(ETS2_START), 0.18,
                           facecolor=TEAL, edgecolor="none", zorder=3))
ax.text((mb(ETS2_START) + mb(ETS2_END)) / 2, 0.16, "ETS2 →", ha="center", va="bottom", fontsize=10.5, color=INK)
ax.plot([mb(SNP)], [0], marker="D", markersize=9, color=VIOLET, markeredgecolor="white",
        markeredgewidth=1.2, zorder=5)
ax.text(mb(SNP), 0.16, "rs2836882", ha="center", va="bottom", fontsize=10.5, color=INK)

# Faint guide from the SNP down through both windows
ax.plot([mb(SNP)] * 2, [-0.08, -1.05], color=VIOLET, lw=1, ls=":", zorder=1)

# Window 1: +/-100 kb around the ETS2 gene body
w1 = (mb(ETS2_START - 100_000), mb(ETS2_END + 100_000))
y1 = -0.45
ax.plot(w1, [y1, y1], color=MUTED, lw=2, ls=(0, (4, 2)), solid_capstyle="butt")
for x in w1:
    ax.plot([x, x], [y1 - 0.07, y1 + 0.07], color=MUTED, lw=2)
gap_kb = round((SNP - (ETS2_END + 100_000)) / 1000)
ax.text(w1[1] + 0.008, y1, f"±100 kb around ETS2 (TenK10K expression map)\nends {gap_kb} kb short of the SNP: never tested",
        ha="left", va="center", fontsize=10, color=INK, zorder=4, bbox=LABEL_BOX)

# Window 2: +/-1 Mb around the ETS2 gene body, which runs past both edges of the plot
y2 = -0.92
ax.annotate("", xy=(XMIN, y2), xytext=(XMAX, y2),
            arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=2, mutation_scale=14))
ax.plot([mb(SNP)], [y2], marker="o", markersize=7, color=VIOLET, markeredgecolor="white", zorder=5)
ax.text(XMIN + 0.012, y2 + 0.09, "±1 Mb around ETS2 (37.8 to 39.8 Mb, past both edges): includes the SNP, "
        "and the ETS2 eQTL colocalizes with IBD (PP.H4 = 0.996)",
        ha="left", va="bottom", fontsize=10, color=INK, zorder=4, bbox=LABEL_BOX)

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(-1.12, 0.45)
ax.set_yticks([])
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="x", colors=MUTED, labelsize=10, labelcolor=INK)
ax.set_xticks(np.arange(38.7, 39.36, 0.1))
ax.set_xlabel("chromosome 21 position (Mb, GRCh38)", fontsize=10.5, color=INK)
ax.set_title(f"A ±100 kb search around ETS2 stops {gap_kb} kb short of the IBD variant", fontsize=13, color=INK, pad=10)

fig.tight_layout()
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)

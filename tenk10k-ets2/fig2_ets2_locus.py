"""Figure 2 for the note "TenK10K meets ETS2": the chr21q22 locus, with the enhancer's Hi-C contact to ETS2.

Coordinates are GRCh38 from the Ensembl REST API (lookup/symbol and overlap/region, queried 2026-10-31).
All protein-coding genes between 38.70 and 39.36 Mb are drawn, plus LINC02940; other non-coding RNAs are left out.
The gap with no protein-coding genes runs from the end of ETS2 (38,824,956) to the start of PSMG1 (39,172,420).

    python fig2_ets2_locus.py <out.png>
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

# name, start (bp), end (bp), strand
ETS2 = ("ETS2", 38_805_115, 38_824_956, +1)
OTHER_GENES = [
    ("PSMG1", 39_172_420, 39_183_517, -1),
    ("BRWD1", 39_184_176, 39_321_559, -1),
    ("HMGN1", 39_342_279, 39_349_653, -1),
]
LINC = ("LINC02940", 38_988_567, 39_062_894, -1)
SNP = 39_094_644  # rs2836882

XMIN, XMAX = 38.70, 39.36
mb = lambda bp: bp / 1e6

out = sys.argv[1]
fig, ax = plt.subplots(figsize=(11, 3.7), dpi=200)


def draw_gene(name, start, end, strand, color, height, label_color=INK):
    ax.add_patch(plt.Rectangle((mb(start), -height / 2), mb(end) - mb(start), height,
                               facecolor=color, edgecolor="none", zorder=3))
    arrow = "→" if strand > 0 else "←"
    label = f"{name} {arrow}" if strand > 0 else f"{arrow} {name}"
    ax.text((mb(start) + mb(end)) / 2, -0.24, label, ha="center", va="top",
            fontsize=10.5, color=label_color)


# Gene track at y = 0
ax.axhline(0, color=GENE_GRAY, lw=0.8, zorder=1)
draw_gene(*ETS2, color=TEAL, height=0.18)
for g in OTHER_GENES:
    draw_gene(*g, color=GENE_GRAY, height=0.18)
draw_gene(*LINC, color="#cbd5e1", height=0.10, label_color=MUTED)

# Enhancer SNP
ax.plot([mb(SNP)], [0], marker="D", markersize=9, color=VIOLET, markeredgecolor="white",
        markeredgewidth=1.2, zorder=5)
ax.text(mb(SNP) + 0.008, 0.16, "rs2836882\nenhancer", ha="left", va="bottom", fontsize=10.5, color=INK)

# Hi-C arc from the enhancer to the ETS2 promoter (gene start, + strand)
x0, x1 = mb(ETS2[1]), mb(SNP)
t = np.linspace(0, 1, 200)
ax.plot(x0 + (x1 - x0) * t, 0.12 + 0.95 * np.sin(np.pi * t), color=TEAL, lw=2, zorder=2)
ax.text((x0 + x1) / 2, 1.13, "Hi-C contact, enhancer to ETS2 promoter: 290 kb (Stankey et al. 2024)",
        ha="center", va="bottom", fontsize=10.5, color=INK)

# The stretch with no protein-coding genes, from the end of ETS2 to the start of PSMG1
d0, d1 = mb(ETS2[2]), mb(OTHER_GENES[0][1])
yd = -0.6
ax.plot([d0, d1], [yd, yd], color=MUTED, lw=1.2)
for x in (d0, d1):
    ax.plot([x, x], [yd - 0.05, yd + 0.05], color=MUTED, lw=1.2)
desert_kb = round((OTHER_GENES[0][1] - ETS2[2]) / 1000)
ax.text((d0 + d1) / 2, yd - 0.09, f"gene desert: no protein-coding genes for {desert_kb} kb",
        ha="center", va="top", fontsize=10, color=MUTED)

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(-0.9, 1.42)
ax.set_yticks([])
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="x", colors=MUTED, labelsize=10, labelcolor=INK)
ax.set_xticks(np.arange(38.7, 39.36, 0.1))
ax.set_xlabel("chromosome 21 position (Mb, GRCh38)", fontsize=10.5, color=INK)
ax.set_title("chr21q22: the IBD enhancer is 290 kb from ETS2", fontsize=13, color=INK, pad=10)

fig.tight_layout()
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)

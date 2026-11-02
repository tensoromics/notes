"""Figure 6 for the note "TenK10K meets ETS2": the mechanism chain from the rs2836882 risk allele to IBD risk,
with the direction evidence for each step.

Every line is a direction reported in a source, not a new estimate:
- Stankey et al., Nature 2024: PU.1 allelic binding (>4-fold to the risk allele), allele-specific ATAC in monocytes
  from 16 heterozygous donors (published data, ref. 70) and 2 macrophage samples, H3K27ac in major vs minor allele
  homozygotes (~2.5-fold across the locus, n = 4 donors of both genotypes),
  MPRA (risk allele G increases transcription), ETS2 eQTL in resting monocytes (Fairfax et al. 2014 genotypes:
  39 AA, 142 AG, 233 GG = 414 people), CRISPR deletion / knockout / overexpression in macrophages.
- TenK10K posted SMR file (IBD, CD14_Mono, peak chr21:39094096-39094839): -0.952709 accessibility and -0.1428 IBD
  log-odds per A allele, i.e. +0.95 and odds x1.15 per G allele. G frequency = 1 - 0.266 = 0.73.
  The IBD GWAS values match Liu et al., Nat Genet 2023 (GWAS Catalog GCST90292538).

    python fig6_direction_chain.py <out.png>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK = "#0f172a"
MUTED = "#64748b"
VIOLET, VIOLET_LIGHT = "#7c3aed", "#ede9fe"
TEAL, TEAL_LIGHT = "#0d9488", "#ccfbf1"
SLATE, SLATE_LIGHT = "#94a3b8", "#f1f5f9"

UP = "▲"

# step label, box style, evidence lines as (marker, text)
STEPS = [
    ("rs2836882 G\nthe risk allele", "snp", [
        ("", "On 73% of chromosomes in TenK10K's donors. The other allele, A, is protective"),
    ]),
    ("PU.1 binds more", "enh", [
        (UP, "Stankey: more than 4× more PU.1 bound to G (macrophages, heterozygous donors)"),
    ]),
    ("Enhancer more open", "enh", [
        (UP, "Stankey: G is the more open allele (published monocyte data, 16 heterozygous donors)"),
        (UP, "TenK10K data: +0.95 per G copy in CD14 monocytes, 922 donors"),
    ]),
    ("Enhancer more active", "enh", [
        (UP, "Stankey: H3K27ac about 2.5× higher across the locus, G vs A homozygotes (4 donors)"),
        (UP, "Stankey: in the reporter assay, G drives more transcription"),
    ]),
    ("More ETS2", "ets2", [
        (UP, "Monocyte eQTL: higher ETS2 with G (414 people, Fairfax et al. data used by Stankey)"),
    ]),
    ("More inflammatory\nmacrophages", "out", [
        (UP, "Stankey: removing the enhancer or ETS2 damps inflammation, adding ETS2 drives it"),
    ]),
    ("Higher IBD risk", "out", [
        (UP, "IBD GWAS (Liu et al. 2023): odds ×1.15 per G copy"),
    ]),
]
STYLE = {
    "snp": dict(fc=VIOLET, ec=VIOLET, tc="white"),
    "enh": dict(fc=VIOLET_LIGHT, ec=VIOLET, tc=INK),
    "ets2": dict(fc=TEAL_LIGHT, ec=TEAL, tc=INK),
    "out": dict(fc=SLATE_LIGHT, ec=SLATE, tc=INK),
}

LINE_H = 3.4      # vertical space per evidence line
BOX_H = 7.0
GAP = 3.6         # space between rows, holds the arrow
BOX_X, BOX_W = 1.0, 24.0
TEXT_X = BOX_X + BOX_W + 4.0
UNIT_IN = 0.083   # inches per vertical data unit, so the figure height follows the number of evidence lines
HEADER_IN, FOOTER_IN = 0.75, 0.1

# Every row gets the height of the tallest, so the arrows between boxes are all the same length
ROW_H = max(max(BOX_H, len(ev) * LINE_H + 1.2) for _, _, ev in STEPS)
heights = [ROW_H] * len(STEPS)
total = sum(heights) + GAP * (len(STEPS) - 1)

out = sys.argv[1]
axes_in = (total + 1.6) * UNIT_IN
fig_h = axes_in + HEADER_IN + FOOTER_IN
fig = plt.figure(figsize=(11, fig_h), dpi=200)
ax = fig.add_axes([0.01, FOOTER_IN / fig_h, 0.98, axes_in / fig_h])
ax.set_xlim(0, 100)
ax.set_ylim(-0.8, total + 0.8)
ax.axis("off")

y_top = total
for i, ((label, kind, evidence), h) in enumerate(zip(STEPS, heights)):
    yc = y_top - h / 2
    st = STYLE[kind]
    ax.add_patch(FancyBboxPatch((BOX_X, yc - BOX_H / 2), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=st["fc"], ec=st["ec"], lw=1.6, zorder=3))
    ax.text(BOX_X + BOX_W / 2, yc, label, ha="center", va="center", fontsize=12, color=st["tc"],
            fontweight="bold" if kind == "snp" else "normal", zorder=4, linespacing=1.25)

    n = len(evidence)
    first = yc + (n - 1) * LINE_H / 2
    for j, (mark, text) in enumerate(evidence):
        y = first - j * LINE_H
        if mark:
            ax.text(TEXT_X, y, mark, ha="left", va="center", fontsize=11, color=INK)
        ax.text(TEXT_X + 2.6, y, text, ha="left", va="center", fontsize=11,
                color=INK if mark != "" else MUTED)

    if i < len(STEPS) - 1:
        y_from = yc - BOX_H / 2 - 0.3
        y_to = (y_top - h - GAP) - (heights[i + 1] / 2 - BOX_H / 2) + 0.3
        ax.annotate("", xy=(BOX_X + BOX_W / 2, y_to), xytext=(BOX_X + BOX_W / 2, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6, mutation_scale=13))
    y_top -= h + GAP

fig.suptitle("From the risk allele to IBD risk, every measured step goes up with G", fontsize=13.5, color=INK,
             x=0.01, ha="left", y=1 - 0.14 / fig_h)
fig.text(0.01, 1 - 0.49 / fig_h, f"{UP} higher with G than with A, as reported by each source",
         fontsize=10, color=MUTED, ha="left")
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)

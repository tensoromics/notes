"""Figure 3 for the note "TenK10K meets ETS2": how SMR turns two per-variant effects into one estimate.

Values are from TenK10K's posted SMR file (Hugging Face anglixue/TenK10K_multiome,
SMR/SMR_results_caQTL2GWAS.zip -> caQTL2GWAS/disease_traits/ibd/CD14_Mono/Chr21_results.smr,
peak chr21:39094096-39094839, top SNP 21:39094644:G:A = rs2836882, effect allele A):
    b_eQTL (accessibility) = -0.952709 per A allele  ->  +0.952709 per G allele
    b_GWAS (IBD log-odds)  = -0.1428   per A allele  ->  +0.1428 per G allele (odds ratio 1.15)
    b_SMR = b_GWAS / b_eQTL = 0.149888 (odds ratio 1.16 per normalized unit of openness), p_SMR = 1.276363e-35
The IBD GWAS values match Liu et al., Nat Genet 2023 (GWAS Catalog GCST90292538).

    python fig3_smr.py <out.png>
"""
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK = "#0f172a"
MUTED = "#64748b"
VIOLET, VIOLET_LIGHT = "#7c3aed", "#ede9fe"
SLATE, SLATE_LIGHT = "#94a3b8", "#f1f5f9"

B_QTL, B_GWAS, B_SMR, P_SMR = 0.952709, 0.1428, 0.149888, 1.276363e-35

out = sys.argv[1]
fig = plt.figure(figsize=(11, 5.2), dpi=200)
ax = fig.add_axes([0.01, 0.02, 0.98, 0.80])
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis("off")

W, H = 24, 9
NODES = {
    "snp": (14, 22, "rs2836882\neach copy of G", dict(fc=VIOLET, ec=VIOLET, tc="white", bold=True)),
    "open": (50, 41, "Enhancer openness\nin CD14 monocytes", dict(fc=VIOLET_LIGHT, ec=VIOLET, tc=INK, bold=False)),
    "ibd": (86, 22, "IBD risk", dict(fc=SLATE_LIGHT, ec=SLATE, tc=INK, bold=False)),
}
for x, y, label, st in NODES.values():
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=st["fc"], ec=st["ec"], lw=1.6, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=12, color=st["tc"], zorder=4, linespacing=1.25,
            fontweight="bold" if st["bold"] else "normal")


def arrow(a, b, color, style="-", shrink=(0, 0)):
    (xa, ya), (xb, yb) = a, b
    ax.annotate("", xy=(xb, yb), xytext=(xa, ya), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, ls=style, mutation_scale=16,
                                shrinkA=shrink[0], shrinkB=shrink[1]))


# Measured: variant -> openness (TenK10K caQTL)
arrow((NODES["snp"][0] + 4, NODES["snp"][1] + H / 2), (NODES["open"][0] - W / 2, NODES["open"][1] - 1), VIOLET)
ax.text(22, 36, f"+{B_QTL:.2f} openness units\nper copy of G\n(TenK10K caQTL)", ha="right", va="center",
        fontsize=10.5, color=INK, linespacing=1.3)

# Measured: variant -> IBD (GWAS)
arrow((NODES["snp"][0] + W / 2, NODES["snp"][1]), (NODES["ibd"][0] - W / 2, NODES["ibd"][1]), INK)
ax.text(50, 19, f"+{B_GWAS:.3f} log-odds of IBD per copy of G, odds ratio {math.exp(B_GWAS):.2f}\n"
        "(IBD GWAS, Liu et al. 2023)", ha="center", va="top", fontsize=10.5, color=INK, linespacing=1.3)

# Inferred: openness -> IBD (SMR)
arrow((NODES["open"][0] + W / 2, NODES["open"][1] - 1), (NODES["ibd"][0] - 4, NODES["ibd"][1] + H / 2), MUTED,
      style=(0, (4, 2)))
ax.text(78, 36, f"SMR estimate: +{B_SMR:.2f} log-odds\nper unit of openness\n(odds ratio {math.exp(B_SMR):.2f})",
        ha="left", va="center", fontsize=10.5, color=INK, linespacing=1.3)

# The calculation
ax.text(50, 6, f"SMR = effect on IBD ÷ effect on openness = {B_GWAS:.3f} ÷ {B_QTL:.3f} = +{B_SMR:.2f}"
        "   (P = 1.3 × 10⁻³⁵)", ha="center", va="center", fontsize=11.5, color=INK,
        bbox=dict(facecolor="white", edgecolor=SLATE, boxstyle="round,pad=0.5"))

fig.suptitle("SMR divides the variant's effect on IBD by its effect on openness", fontsize=13.5, color=INK,
             x=0.01, ha="left", y=0.975)
fig.text(0.01, 0.925, "Solid arrows: measured, per copy of the risk allele G. Dashed arrow: inferred.\n"
         "Both measured effects are positive, so more openness goes with more IBD risk.",
         fontsize=10, color=MUTED, ha="left", va="top", linespacing=1.4)
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)

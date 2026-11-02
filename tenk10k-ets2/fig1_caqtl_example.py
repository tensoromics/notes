"""Figure 1 for the note "TenK10K meets ETS2": what a caQTL looks like. Left, a regional plot of every variant TenK10K
tested against the ETS2 enhancer peak in CD14 monocytes. Right, a model-based boxplot of openness by genotype.

Left panel data: data/ets2_peak_cd14_cis_pairs.csv, the 7,356 rows for peak chr21:39094096-39094839 in
caQTL_common_TensorQTL/CD14_Mono/TenK10K.cis_qtl_pairs.chr21.parquet (made by extract_ets2_peak.py). P is tensorQTL's
nominal P. Seven variants give identical results (same slope, SE, carrier counts), so they tie at the top; the posted
files carry no LD, so the other dots are not colored by it.

Right panel: the per-donor accessibility values are under controlled access (EGA), so this shows the spread the fitted
model implies. It does not show individual donors.
    slope per A allele b = -0.952709, se = 0.0256352 (rs2836882, 21:39094644:G:A)
    genotype counts AA 68, AG 355, GG 499, from ma_samples 423, ma_count 491, af 0.266269:
    N = ma_count / (2 af); minor homozygotes = ma_count - ma_samples; heterozygotes = ma_samples - minor homozygotes.
Model: openness = k * beta_G + e, with k = copies of G, beta_G = 0.952709, e ~ Normal(0, sigma^2).
    sigma comes from the slope's standard error: se = sigma / sqrt(sum (k - mean k)^2), so
    sigma = se * sqrt(N * var(k)). This ignores the covariate adjustment of genotype in tensorQTL, which makes sum
    (k - mean k)^2 slightly smaller, so sigma is approximate (an upper bound).
Boxes: middle 50% (median +/- 0.6745 sigma). Whiskers: middle 90% (median +/- 1.6449 sigma).

    python fig1_caqtl_example.py <out.png>
"""
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
DOT_GRAY = "#94a3b8"
VIOLET, VIOLET_LIGHT = "#7c3aed", "#ede9fe"
TEAL = "#0d9488"

PEAK = (39_094_096, 39_094_839)
ETS2 = (38_805_115, 38_824_956)
RS2836882 = "21:39094644:G:A"
BETA_G, SE = 0.952709, 0.0256352
COUNTS = {"AA": 68, "AG": 355, "GG": 499}

data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ets2_peak_cd14_cis_pairs.csv")
rows = []
with open(data) as f:
    for r in csv.DictReader(f):
        rows.append((r["variant_id"], int(r["variant_id"].split(":")[1]), float(r["pval_nominal"]),
                     float(r["slope"]), float(r["slope_se"]), int(r["ma_count"]), int(r["ma_samples"])))
ref = next(r for r in rows if r[0] == RS2836882)
tied = [r for r in rows if r[3:] == ref[3:]]
others = [r for r in rows if r[3:] != ref[3:]]

out = sys.argv[1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=200, gridspec_kw={"width_ratios": [1.35, 1]})

# Left: regional plot
mb = lambda bp: bp / 1e6
ax1.scatter([mb(r[1]) for r in others], [-math.log10(r[2]) for r in others], s=9, color=DOT_GRAY, alpha=0.7,
            linewidths=0, zorder=2)
ax1.scatter([mb(r[1]) for r in tied], [-math.log10(r[2]) for r in tied], s=40, color=VIOLET, edgecolors="white",
            linewidths=0.8, zorder=4)
top = -math.log10(ref[2])
ax1.annotate(f"{len(tied)} variants tied at the top, including\nrs2836882 and an 11-base deletion",
             xy=(mb(ref[1]), top), xytext=(mb(ref[1]) - 0.95, top - 12), ha="left", va="top", fontsize=10,
             color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
# Gene and peak markers just below zero
ax1.add_patch(plt.Rectangle((mb(ETS2[0]), -9), mb(ETS2[1]) - mb(ETS2[0]), 5, facecolor=TEAL, edgecolor="none",
                            zorder=3))
ax1.text(mb(ETS2[0]) - 0.01, -6.5, "ETS2", ha="right", va="center", fontsize=9.5, color=INK)
ax1.text(mb(PEAK[0]) + 0.02, -6.5, "peak", ha="left", va="center", fontsize=9.5, color=INK)
ax1.plot([mb(PEAK[0])], [-6.5], marker="D", markersize=6, color=VIOLET, zorder=3)
ax1.set_xlim(mb(PEAK[0]) - 1.02, mb(PEAK[1]) + 1.02)
ax1.set_ylim(-12, top + 12)
ax1.set_xlabel("chromosome 21 position (Mb, GRCh38)", fontsize=10.5, color=INK)
ax1.set_ylabel("−log₁₀ P, association with the peak's openness", fontsize=10.5, color=INK)
ax1.set_title(f"All {len(rows):,} variants within 1 Mb, tested against the peak", fontsize=11.5, color=INK,
              loc="left")

# Right: model-based boxplot
n = sum(COUNTS.values())
ks = [0, 1, 2]
mean_k = sum(k * c for k, c in zip(ks, COUNTS.values())) / n
var_k = sum(c * (k - mean_k) ** 2 for k, c in zip(ks, COUNTS.values())) / n
sigma = SE * math.sqrt(n * var_k)
Q50, Q90 = 0.6745, 1.6449
for k, (g, c) in zip(ks, COUNTS.items()):
    med = k * BETA_G
    lo50, hi50 = med - Q50 * sigma, med + Q50 * sigma
    lo90, hi90 = med - Q90 * sigma, med + Q90 * sigma
    ax2.plot([k, k], [lo90, lo50], color=VIOLET, lw=1.5, zorder=2)
    ax2.plot([k, k], [hi50, hi90], color=VIOLET, lw=1.5, zorder=2)
    for y in (lo90, hi90):
        ax2.plot([k - 0.08, k + 0.08], [y, y], color=VIOLET, lw=1.5, zorder=2)
    ax2.add_patch(plt.Rectangle((k - 0.22, lo50), 0.44, hi50 - lo50, facecolor=VIOLET_LIGHT, edgecolor=VIOLET,
                                lw=1.5, zorder=3))
    ax2.plot([k - 0.22, k + 0.22], [med, med], color=INK, lw=2.2, zorder=4)
    ax2.text(k + 0.27, med, "0" if k == 0 else f"+{med:.2f}", va="center", ha="left", fontsize=10.5, color=INK)
ax2.set_xticks(ks)
ax2.set_xticklabels([f"{g}\nn = {c}" for g, c in COUNTS.items()], fontsize=10.5, color=INK)
ax2.set_xlim(-0.55, 2.75)
ax2.set_xlabel("genotype at rs2836882", fontsize=10.5, color=INK)
ax2.set_ylabel("openness relative to AA (normalized units)", fontsize=10.5, color=INK)
ax2.set_title("Openness by genotype at rs2836882", fontsize=11.5, color=INK, loc="left")

for ax in (ax1, ax2):
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10, labelcolor=INK)
ax2.tick_params(axis="x", length=0)

fig.suptitle("A caQTL: the ETS2 enhancer peak in CD14 monocytes is associated with genotype", fontsize=13,
             color=INK, x=0.01, ha="left", y=0.985)
fig.text(0.01, 0.92, "Left: TenK10K's posted results. Right: expected spread under the fitted model after covariate "
         f"adjustment (residual SD ≈ {sigma:.2f}),\nnot individual donors. Boxes: middle 50%. Whiskers: middle 90%. "
         "Line: median.", fontsize=9.5, color=MUTED, ha="left", va="top", linespacing=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.85), w_pad=3)
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out, f"variants={len(rows)} tied={len(tied)} sigma={sigma:.4f}")

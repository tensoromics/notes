"""Figure 7 for the note "TenK10K meets ETS2": how far eQTLs sit from their gene, against the two search windows.

Medians are from IBDverse (Alegbe et al., Nature 2026, Results, "eQTLs detected only at the cell-type level..."):
52 kb (all cells pooled), 114 kb (major population), 200 kb (cell type only), measured to the gene's TSS.
rs2836882 to the ETS2 TSS: 39,094,644 - 38,805,115 = 289,529 bp (Ensembl GRCh38, ETS2 on the + strand).
The TenK10K expression window (Cuomo et al., Methods) is +/-100 kb around the gene body, not the TSS.

    python fig7_eqtl_distance.py <out.png>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0f172a"
MUTED = "#64748b"
BAR = "#94a3b8"
VIOLET = "#7c3aed"
LABEL_BOX = dict(facecolor="white", edgecolor="none", pad=1.5)

ROWS = [
    ("All cells pooled", 52),
    ("Major cell population", 114),
    ("Single cell type only", 200),
]
ETS2_KB = 289.529
XMAX = 350

out = sys.argv[1]
fig, ax = plt.subplots(figsize=(11, 3.9), dpi=200)

# Median bars, top to bottom; the ETS2 enhancer sits on its own row below them
ys = [3.6, 2.6, 1.6]
for (label, kb), y in zip(ROWS, ys):
    ax.barh(y, kb, height=0.5, color=BAR, zorder=3)
    ax.text(kb + 4, y, f"median {kb} kb", va="center", ha="left", fontsize=10.5, color=INK, zorder=4, bbox=LABEL_BOX)
y_ets2 = 0.5
ax.plot([ETS2_KB], [y_ets2], marker="D", markersize=10, color=VIOLET, markeredgecolor="white",
        markeredgewidth=1.2, zorder=5)
ax.plot([0, ETS2_KB], [y_ets2, y_ets2], color=VIOLET, lw=1, ls=":", zorder=2)
ax.text(ETS2_KB - 6, y_ets2 + 0.32, "290 kb", va="bottom", ha="center", fontsize=10.5, color=INK, zorder=4)

ax.set_yticks(ys + [y_ets2])
ax.set_yticklabels([r[0] for r in ROWS] + ["ETS2 enhancer (rs2836882)"], fontsize=10.5, color=INK)
ax.text(-0.01, 4.25, "IBDverse eQTLs, by the resolution\nat which they were detected", transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=9.5, color=MUTED)

# TenK10K expression window: dashed, as in figure 1
ax.axvline(100, color=MUTED, lw=2, ls=(0, (4, 2)), zorder=1)
ax.text(100 + 4, 4.35, "±100 kb: TenK10K expression map", va="center", ha="left", fontsize=10, color=INK,
        zorder=4, bbox=LABEL_BOX)

# IBDverse window: off the right edge of this scale
ax.annotate("±1 Mb: IBDverse (off scale)", xy=(XMAX, 4.35), xytext=(XMAX - 32, 4.35),
            ha="right", va="center", fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=2, mutation_scale=14))

ax.set_xlim(0, XMAX)
ax.set_ylim(0, 4.75)
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=MUTED, labelsize=10, labelcolor=INK)
ax.set_xlabel("distance from the gene's transcription start site (kb)", fontsize=10.5, color=INK)
ax.set_title("eQTLs found only in single cell types are a median 200 kb from their gene", fontsize=13, color=INK, pad=10)

fig.tight_layout()
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)

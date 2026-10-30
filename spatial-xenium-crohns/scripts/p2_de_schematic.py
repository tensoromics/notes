#!/usr/bin/env python3
"""Part 2, last section schematic: how we compare T cells near the epithelium with T cells far from it.

  A  one section: T cells within 10 µm of an epithelial cell (near) and more than 50 µm away (far);
     T cells in between are left out; EPCAM dots counted for the near T cells are ringed
  B  in that section: each gene per 1,000 RNA molecules, near against far, and how many times higher
  C  the same comparison in every section, one dot per section, and the median over the sections

Synthetic and deterministic (fixed seed; made-up numbers); the colors match p2_test_schematic.py.
Usage: python p2_de_schematic.py [out_dir]      -> de_schematic.png
"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                      os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(OUT, exist_ok=True)
INK, MUTED, WRONG, OTHER = "#1b2024", "#6b727a", "#d62728", "#9aa1a8"
EPI, TCELL = "#1C8577", "#7A5DA9"
rng = np.random.default_rng(7)

def dots_in(c, a, b, n, side=None):
    # n uniform dots inside an ellipse; side="left" keeps them in the left third (near the epithelium)
    out = []
    while len(out) < n:
        p = rng.uniform(-1, 1, 2)
        if p @ p <= 0.8 and (side != "left" or p[0] < -0.45):
            out.append(c + p * [a, b])
    return np.array(out)

fig, (A, B, C) = plt.subplots(1, 3, figsize=(17, 5.4),
                              gridspec_kw=dict(width_ratios=[1.5, 0.8, 0.9], wspace=0.28))

# A · one section
XE = -2.6                                  # centers of the epithelial cells (a column)
TOP, BOTTOM = 11.0, -11.0
def band(x0, x1, color, label, sub):
    A.fill_between([x0, x1], BOTTOM, TOP, color=color, zorder=0, lw=0)
    A.text((x0 + x1) / 2, TOP + 0.5, label, ha="center", va="bottom", fontsize=12, color=INK)
    A.text((x0 + x1) / 2, BOTTOM - 0.6, sub, ha="center", va="top", fontsize=12, color=INK,
           weight="bold" if sub != "left out" else "normal")
band(XE, XE + 10, "#efeaf6", "0–10 µm", "near")
MID0, MID1 = XE + 10, XE + 21              # 10-50 µm, drawn shorter than it is
band(MID0, MID1, "#f8f9fa", "10–50 µm", "left out")
A.text((MID0 + MID1) / 2, 6.5, "…", ha="center", va="center", fontsize=22, color=MUTED)
FAR0, FAR1 = MID1, MID1 + 13               # over 50 µm
band(FAR0, FAR1, "#efeaf6", "over 50 µm", "far")
A.text(XE - 3.4, TOP + 2.6, "distance from each T cell's center to the nearest epithelial cell's center:",
       ha="left", va="bottom", fontsize=12, color=MUTED)

# the epithelial sheet
for y in (-7.0, 0, 7.0):
    A.add_patch(Ellipse((XE, y), 5.0, 6.5, facecolor=EPI, alpha=0.15, edgecolor=INK, lw=1.4))
    d = dots_in(np.array([XE, y]), 2.5, 3.25, 9)
    A.scatter(d[:, 0], d[:, 1], s=34, color=EPI, edgecolors="white", linewidths=0.5, zorder=3)

def t_cell(x, y, n_epcam, faint=False):
    c = np.array([x, y])
    alpha = 0.06 if faint else 0.15
    A.add_patch(Ellipse(c, 4.4, 4.4, facecolor=TCELL, alpha=alpha, edgecolor=MUTED if faint else INK, lw=1.4))
    A.plot(*c, marker="+", color=MUTED if faint else INK, ms=9, mew=1.6, zorder=6)
    d = dots_in(c, 2.2, 2.2, 6)
    A.scatter(d[:, 0], d[:, 1], s=30, color=OTHER, edgecolors="white", linewidths=0.5, zorder=3,
              alpha=0.4 if faint else 1)
    if n_epcam:
        e = dots_in(c, 2.2, 2.2, n_epcam, side="left")
        A.scatter(e[:, 0], e[:, 1], s=34, color=EPI, edgecolors="white", linewidths=0.5, zorder=4)
        A.scatter(e[:, 0], e[:, 1], s=150, facecolors="none", edgecolors=WRONG, linewidths=1.8, zorder=5)

t_cell(XE + 4.8, 3.6, 1)
t_cell(XE + 7.6, -4.2, 1)
t_cell(MID0 + 3.0, -2.0, 0, faint=True)
t_cell(FAR0 + 4.0, 3.6, 0)
t_cell(FAR0 + 8.8, -4.2, 0)

A.set_xlim(XE - 3.6, FAR1 + 0.4); A.set_ylim(BOTTOM - 2.6, TOP + 4.2)
A.set_aspect("equal"); A.axis("off")
A.set_title("A · Two groups of T cells in one section", loc="left", fontsize=15, color=INK, pad=4)
handles = [Ellipse((0, 0), 1, 1, facecolor=EPI, alpha=0.3, edgecolor=INK),
           Ellipse((0, 0), 1, 1, facecolor=TCELL, alpha=0.3, edgecolor=INK),
           plt.Line2D([0], [0], marker="o", ls="", ms=9, color=EPI, mec="white"),
           plt.Line2D([0], [0], marker="o", ls="", ms=9, color=OTHER, mec="white"),
           plt.Line2D([0], [0], marker="o", ls="", ms=12, mfc="none", mec=WRONG, mew=1.8)]
fig.legend(handles, ["epithelial cell", "T cell", "EPCAM dot", "dot of another gene",
                     "EPCAM counted for a T cell"],
           loc="lower center", bbox_to_anchor=(0.25, -0.07), ncol=3, fontsize=11.5, frameon=False,
           handler_map={Ellipse: matplotlib.legend_handler.HandlerPatch()})

# B · the comparison in that section (made-up numbers)
genes = ["EPCAM", "ITGAE", "CD3E"]
near = np.array([0.64, 0.40, 12.0])
far = np.array([0.08, 0.20, 12.0])
x = np.arange(len(genes))
B.bar(x - 0.2, near, 0.38, color=TCELL, label="near")
B.bar(x + 0.2, far, 0.38, color="#c9bfe0", label="far")
for xi, n, f in zip(x, near, far):
    B.text(xi, max(n, f) * 1.35, f"{n / f:.0f}×", ha="center", va="bottom", fontsize=13, color=INK,
           weight="bold")
B.set_yscale("log"); B.set_ylim(0.04, 60)
B.set_yticks([]); B.minorticks_off()
B.set_xticks(x, genes, fontsize=12)
B.set_ylabel("per 1,000 RNA molecules (log scale)", fontsize=12)
for sp in ("top", "right"):
    B.spines[sp].set_visible(False)
B.legend(frameon=False, fontsize=11.5, loc="upper left")
B.set_title("B · Near against far, gene by gene", loc="left", fontsize=15, color=INK)
B.text(0.5, -0.11, "times higher near = near ÷ far", transform=B.transAxes, ha="center", va="top",
       fontsize=12, color=MUTED)

# C · the same in every section: one dot per section, and the median (made-up numbers)
folds = np.array([1.2, 2.5, 4, 5, 6, 7, 7.5, 8.5, 9, 10, 12, 16, 18, 30])
yj = rng.uniform(-0.12, 0.12, len(folds))
q1, med, q3 = np.percentile(folds, [25, 50, 75])
C.add_patch(plt.Rectangle((q1, -0.25), q3 - q1, 0.5, facecolor=EPI, alpha=0.2, edgecolor=MUTED))
C.plot([med, med], [-0.25, 0.25], color=INK, lw=2)
C.scatter(folds, yj, s=30, color=EPI, zorder=3)
C.axvline(1, color=INK, lw=0.8)
C.text(med, 0.36, f"median: {med:.0f}×", ha="center", va="bottom", fontsize=12, color=INK, weight="bold")
C.set_xscale("log"); C.set_xlim(0.7, 45); C.set_ylim(-0.8, 0.9)
C.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"]); C.minorticks_off()
C.set_yticks([0], ["EPCAM"], fontsize=12)
C.set_xlabel("times higher near, one dot per section", fontsize=12)
for sp in ("top", "right", "left"):
    C.spines[sp].set_visible(False)
C.set_title("C · Repeat in every section", loc="left", fontsize=15, color=INK)
C.text(5, -0.62, "14 sections, so 14 dots per gene", ha="center", va="center",
       fontsize=12, color=MUTED)

fig.savefig(os.path.join(OUT, "de_schematic.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", os.path.join(OUT, "de_schematic.png"))

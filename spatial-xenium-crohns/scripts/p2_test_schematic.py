#!/usr/bin/env python3
"""Part 2, section 3 schematic: how we test whether cells hold genes from the cells next to them.

  A  the test: an epithelial sheet (which makes EPCAM) and non-epithelial cells at growing distances,
     in the same distance groups the code uses; EPCAM dots counted for the wrong cell are ringed
  B  what we expect: EPCAM in non-epithelial cells by distance group, if RNA is counted for the wrong
     cell (high next to the epithelium, low far away) and if it isn't (the same everywhere)

Synthetic and deterministic (fixed seed); the colors match p2_schematic.py.
Usage: python p2_test_schematic.py [out_dir]      -> test_schematic.png
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
EPI, TCELL, FIB = "#1C8577", "#7A5DA9", "#A2701A"
rng = np.random.default_rng(4)

def dots_in(c, a, b, n, side=None):
    # n uniform dots inside an ellipse; side="left" keeps them in the left third (near the epithelium)
    out = []
    while len(out) < n:
        p = rng.uniform(-1, 1, 2)
        if p @ p <= 0.8 and (side != "left" or p[0] < -0.45):
            out.append(c + p * [a, b])
    return np.array(out)

fig, (A, B) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.12))

# A · the test
XE = -2.6                                  # centers of the epithelial cells (a column)
bands = [0, 5, 10, 15, 20]                 # distance groups, from the epithelial centers
TOP, BOTTOM = 11.0, -11.0                  # the bands span the drawing, not the legend
def band(x0, x1, shade, label):
    A.fill_between([x0, x1], BOTTOM, TOP, color="#eef0f2" if shade else "#f8f9fa", zorder=0, lw=0)
    A.text((x0 + x1) / 2, TOP + 0.5, label, ha="center", va="bottom", fontsize=12, color=INK)
for k in range(len(bands) - 1):
    band(XE + bands[k], XE + bands[k + 1], k % 2 == 0, f"{bands[k]}–{bands[k + 1]}")
FAR = 31.0                                 # the far group, after a break in the axis
band(FAR - 3.2, FAR + 3.2, True, "over 100")
A.text((XE + 20 + FAR - 3.2) / 2, 0, "…", ha="center", va="center", fontsize=22, color=MUTED)
A.text(XE - 3.4, TOP + 2.6, "distance from each cell's center to the nearest epithelial cell's center (µm):",
       ha="left", va="bottom", fontsize=12, color=MUTED)

# the epithelial sheet: tall columns, full of EPCAM
for y in (-7.0, 0, 7.0):
    A.add_patch(Ellipse((XE, y), 5.0, 6.5, facecolor=EPI, alpha=0.15, edgecolor=INK, lw=1.4))
    d = dots_in(np.array([XE, y]), 2.5, 3.25, 9)
    A.scatter(d[:, 0], d[:, 1], s=34, color=EPI, edgecolors="white", linewidths=0.5, zorder=3)

# non-epithelial cells at growing distances; (distance, kind, EPCAM dots counted for it)
cells = [(4.7, TCELL, 2), (8.2, FIB, 1), (12.6, TCELL, 0), (17.4, FIB, 0)]
for dist, color, n_epcam in cells:
    c = np.array([XE + dist, 0.0])
    a, b = (2.3, 2.3) if color == TCELL else (2.6, 1.7)
    A.add_patch(Ellipse(c, 2 * a, 2 * b, facecolor=color, alpha=0.15, edgecolor=INK, lw=1.4))
    # the cell's center, which sets its distance group
    A.plot(*c, marker="+", color=INK, ms=9, mew=1.6, zorder=6)
    d = dots_in(c, a, b, 6)
    A.scatter(d[:, 0], d[:, 1], s=30, color=OTHER, edgecolors="white", linewidths=0.5, zorder=3)
    if n_epcam:
        e = dots_in(c, a, b, n_epcam, side="left")
        A.scatter(e[:, 0], e[:, 1], s=34, color=EPI, edgecolors="white", linewidths=0.5, zorder=4)
        A.scatter(e[:, 0], e[:, 1], s=150, facecolors="none", edgecolors=WRONG, linewidths=1.8, zorder=5)
c = np.array([FAR, 0.0])
A.add_patch(Ellipse(c, 4.6, 4.6, facecolor=TCELL, alpha=0.15, edgecolor=INK, lw=1.4))
A.plot(*c, marker="+", color=INK, ms=9, mew=1.6, zorder=6)
d = dots_in(c, 2.3, 2.3, 6)
A.scatter(d[:, 0], d[:, 1], s=30, color=OTHER, edgecolors="white", linewidths=0.5, zorder=3)

# the distance d for the first cell, center to center
A.plot([XE, XE + 4.7], [0, 0], color=INK, lw=0.9, ls=":", zorder=6)
A.plot(XE, 0, marker="+", color=INK, ms=9, mew=1.6, zorder=6)
A.annotate("", (XE + 4.7, -3.0), (XE, -3.0), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
A.text(XE + 2.35, -3.3, "d", ha="center", va="top", fontsize=13, style="italic", color=INK)

A.set_xlim(XE - 3.6, FAR + 3.4); A.set_ylim(BOTTOM - 0.3, TOP + 4.2)
A.set_aspect("equal"); A.axis("off")
A.set_title("A · The test: EPCAM in the cells that don't make it", loc="left", fontsize=15, color=INK,
            pad=4)
handles = [Ellipse((0, 0), 1, 1, facecolor=EPI, alpha=0.3, edgecolor=INK),
           plt.Line2D([0], [0], marker="o", ls="", ms=9, color=EPI, mec="white"),
           plt.Line2D([0], [0], marker="o", ls="", ms=9, color=OTHER, mec="white"),
           plt.Line2D([0], [0], marker="o", ls="", ms=12, mfc="none", mec=WRONG, mew=1.8),
           Ellipse((0, 0), 1, 1, facecolor=TCELL, alpha=0.3, edgecolor=INK),
           Ellipse((0, 0), 1, 1, facecolor=FIB, alpha=0.3, edgecolor=INK)]
fig.legend(handles, ["epithelial cell", "EPCAM dot", "dot of another gene",
                     "EPCAM counted for the wrong cell", "T cell", "fibroblast"],
           loc="lower center", bbox_to_anchor=(0.36, -0.06), ncol=3, fontsize=11.5, frameon=False,
           handler_map={Ellipse: matplotlib.legend_handler.HandlerPatch()})

# B · what we expect
groups = ["0-5", "5-10", "10-15", "15-20", "…", "over 100"]
x = np.arange(len(groups))
wrong = [20, 10, 3.2, 2.0, 1.3, 1.0]
B.plot(x, wrong, color=WRONG, marker="o", lw=2,
       label="if RNA is counted for the wrong cell:\nhigh next to the epithelium")
B.plot(x, np.ones(len(x)), color=MUTED, marker="o", lw=2, ls="--",
       label="if it isn't: the same at every distance")
B.set_yscale("log"); B.set_ylim(0.6, 40)
B.set_yticks([]); B.minorticks_off()
B.set_xticks(x, groups, fontsize=11.5)
B.set_xlabel("distance to the nearest epithelial cell (µm)", fontsize=12)
B.set_ylabel("EPCAM per 1,000 RNA molecules\nin non-epithelial cells", fontsize=12)
for sp in ("top", "right"):
    B.spines[sp].set_visible(False)
B.legend(frameon=False, fontsize=11.5, loc="upper right")
B.set_title("B · What we expect", loc="left", fontsize=15, color=INK)

fig.savefig(os.path.join(OUT, "test_schematic.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", os.path.join(OUT, "test_schematic.png"))

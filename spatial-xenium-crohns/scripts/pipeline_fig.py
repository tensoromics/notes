#!/usr/bin/env python3
"""Intro schematic in two figures, sized for the text column:
  pipeline_a.png  tissue -> thin slice on a slide -> transcripts in the slice (x, y, z)
  pipeline_b.png  flattened to dots (x, y) -> segmentation boundaries -> cell-by-gene matrix

Synthetic and deterministic (fixed seed): ~9 made-up cells, each a cluster of transcripts
colored by gene, with a z-depth. No data needed.

Usage: python pipeline_fig.py [out_dir]
"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                      os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

GENES = ["#1C8577", "#7A5DA9", "#A2701A", "#2C6E9B"]     # 4 gene colors
GNAME = ["gene A", "gene B", "gene C", "gene D"]
INK, GRID, CELL = "#1b2024", "#c9c7c0", "#6b727a"
TIS, TIS_SIDE, SLAB = "#E9BDB6", "#D49C95", "#C9798A"     # tissue block, its sides, the slice

# ---- synthetic cells: ~9 clusters of transcripts, each favoring one gene ----
centers = np.array([[7, 8], [12, 20], [20, 10], [19, 22], [27, 17],
                    [30, 7], [24, 26], [10, 27], [33, 24]], float)
dom = [0, 1, 2, 1, 3, 0, 2, 3, 1]
pts_xy, pts_z, pts_g, cell_of = [], [], [], []
for ci, (c, d) in enumerate(zip(centers, dom)):
    n = rng.integers(16, 26)
    p = c + rng.normal(0, 2.1, (n, 2))
    z = rng.uniform(1.5, 9.5, n)
    g = np.where(rng.random(n) < 0.7, d, rng.integers(0, 4, n))
    pts_xy.append(p); pts_z.append(z); pts_g.append(g); cell_of += [ci] * n
pts_xy = np.vstack(pts_xy); pts_z = np.concatenate(pts_z)
pts_g = np.concatenate(pts_g).astype(int); cell_of = np.array(cell_of)
cols = np.array(GENES)[pts_g]

# an irregular tissue outline (top view), reused for the block and the slice
th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
rad = 1 + 0.10 * np.sin(3 * th + 0.4) + 0.06 * np.cos(5 * th) + 0.04 * np.sin(7 * th + 1)
outline = np.c_[1.5 * rad * np.cos(th), 1.0 * rad * np.sin(th)]

FIGSIZE = (12, 4.9)
fig = plt.figure(figsize=FIGSIZE)
W, H = 0.31, 0.66
pos = [(0.01, 0.17), (0.345, 0.17), (0.68, 0.17)] * 2

PANEL = iter(range(6))
def title(ax, s, sub, sub_y=None):
    k = next(PANEL); x0, y0 = pos[k]
    fig.text(x0 + W / 2, y0 + H + 0.012, s, fontsize=16, ha="center", va="bottom", color=INK)
    fig.text(x0 + W / 2, y0 - 0.012, sub, fontsize=13, ha="center", va="top", color=INK)

# ---- 1 · Tissue: a block with a thin slice lifted off the top ----
ax = fig.add_axes([*pos[0], W, H], projection="3d")
def slab(z0, z1, top, side):
    n = len(outline); faces = []
    lo = np.c_[outline, np.full(n, z0)]; hi = np.c_[outline, np.full(n, z1)]
    for i in range(n):
        j = (i + 1) % n
        faces.append([lo[i], lo[j], hi[j], hi[i]])
    ax.add_collection3d(Poly3DCollection(faces, facecolor=side, edgecolor=side, lw=0.3))
    ax.add_collection3d(Poly3DCollection([hi], facecolor=top, edgecolor="#a8706b", lw=0.8))
slab(0, 2.2, TIS, TIS_SIDE)
slab(3.0, 3.25, SLAB, "#b36474")
ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_zlim(-0.4, 3.6)
ax.view_init(elev=24, azim=-58); ax.set_axis_off()
ax.text(1.75, 0.2, 3.15, "thin section", fontsize=10, color="#8a3f50")
title(ax, "1 · Tissue", "a piece of ileum, fixed, set in paraffin")

# ---- 2 · The slice on a glass slide ----
ax = fig.add_axes([*pos[1], W, H])
ax.add_patch(FancyBboxPatch((-3.2, -1.25), 6.4, 2.5, boxstyle="round,pad=0.02,rounding_size=0.08",
                            facecolor="#eef4f6", edgecolor="#9fb3ba", lw=1.4))
ax.add_patch(Rectangle((-3.2, -1.25), 1.2, 2.5, facecolor="#dfe7ea", edgecolor="#9fb3ba", lw=1.0))
ax.add_patch(Polygon(outline * 0.85 + [0.5, 0], closed=True, facecolor=SLAB, alpha=0.85,
                     edgecolor="#8a3f50", lw=1.0))
ax.add_patch(Rectangle((0.75, -0.05), 0.42, 0.42, fill=False, ec=INK, lw=1.4))
ax.text(1.25, 0.42, "imaged area", fontsize=9.5, color=INK, va="bottom")
ax.set_xlim(-3.4, 3.4); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal"); ax.set_axis_off()
title(ax, "2 · Section on a slide", "a 5 µm section on a Xenium slide")

# ---- 3 · Transcripts in the slice (x, y, z) ----
ax = fig.add_axes([*pos[2], W, H], projection="3d")
ax.scatter(pts_xy[:, 0], pts_xy[:, 1], pts_z, c=cols, s=26, depthshade=True,
           edgecolors="white", linewidths=0.3)
ax.set_xlabel("x", fontsize=8, labelpad=-8); ax.set_ylabel("y", fontsize=8, labelpad=-8)
ax.set_zlabel("z", fontsize=8, labelpad=-8)
ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
ax.set_zlim(0, 11); ax.view_init(elev=18, azim=-60)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_facecolor((1, 1, 1, 0)); pane.pane.set_edgecolor(GRID)
    pane._axinfo["grid"]["color"] = (0.85, 0.85, 0.82, 1)
title(ax, "3 · Transcripts  (x, y, z)", "one dot per mRNA, colored by gene (4 of 5,000)")

def arrows(f):
    for x in (0.338, 0.673):
        f.text(x, 0.5, "→", fontsize=34, color=CELL, ha="center", va="center")
arrows(fig)
fig.savefig(os.path.join(OUT, "pipeline_a.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
fig = plt.figure(figsize=FIGSIZE)

# ---- 4 · Flattened to dots (x, y) ----
ax4 = fig.add_axes([*pos[3], W, H])
ax4.scatter(pts_xy[:, 0], pts_xy[:, 1], c=cols, s=40, edgecolors="white", linewidths=0.5)
title(ax4, "4 · Flattened  (x, y)", "z dropped, leaving a flat cloud of dots")

# ---- 5 · Segmentation: a boundary around each cell ----
ax5 = fig.add_axes([*pos[4], W, H])
ax5.scatter(pts_xy[:, 0], pts_xy[:, 1], c=cols, s=28, alpha=0.6, edgecolors="none", zorder=3)

def clip(poly, a, b):
    # keep the part of a convex polygon that is closer to point a than to point b
    n, mid = b - a, (a + b) / 2
    inside = lambda q: (q - mid) @ n <= 0
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        if inside(p):
            out.append(p)
        if inside(p) != inside(q):
            s = ((mid - p) @ n) / ((q - p) @ n)
            out.append(p + s * (q - p))
    return np.array(out)

# outline = the cell's (slightly expanded) convex hull, trimmed to its own territory, so that
# no two outlines overlap, as in a real segmentation mask
outlines = []
for ci in range(len(centers)):
    P = pts_xy[cell_of == ci]; poly = P[ConvexHull(P).vertices]
    cen = poly.mean(0); poly = cen + 1.16 * (poly - cen)
    for cj in range(len(centers)):
        if cj != ci:
            poly = clip(poly, centers[ci], centers[cj])
    outlines.append(poly)
    ax5.add_patch(Polygon(poly, closed=True, facecolor=CELL, alpha=0.14, edgecolor=CELL, lw=2.0, zorder=2))

# point-in-polygon: each dot goes to the outline that contains it (or to none)
from matplotlib.path import Path
assigned = np.full(len(pts_xy), -1)
for ci, poly in enumerate(outlines):
    assigned[Path(poly).contains_points(pts_xy) & (assigned < 0)] = ci
    ax5.text(*centers[ci], str(ci + 1), fontsize=9, ha="center", va="center", color=INK,
             fontweight="bold", zorder=4)
title(ax5, "5 · Boundaries → cells", "each dot goes to the boundary around it")
for a in (ax4, ax5):
    a.set_xlim(2, 38); a.set_ylim(2, 31); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_edgecolor(GRID)

# ---- 6 · Cell-by-gene matrix ----
ax6 = fig.add_axes([pos[5][0] + 0.05, pos[5][1] + 0.01, W - 0.09, H - 0.06])
M = np.zeros((len(centers), 4), int)
keep = assigned >= 0
np.add.at(M, (assigned[keep], pts_g[keep]), 1)
print(f"dots assigned: {keep.sum()} of {len(keep)}; assigned to a different cell than they came from: "
      f"{(keep & (assigned != cell_of)).sum()}")
ax6.imshow(M, cmap="Greys", vmin=0, vmax=M.max() * 4.5, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax6.text(j, i, M[i, j], ha="center", va="center", fontsize=9.5,
                 color=GENES[j] if M[i, j] else "#b0b4b8", fontweight="bold")
ax6.set_xticks(range(4)); ax6.set_xticklabels(GNAME, fontsize=9)
ax6.set_yticks(range(len(centers))); ax6.set_yticklabels([f"cell {i+1}" for i in range(len(centers))], fontsize=9)
ax6.xaxis.tick_top(); ax6.tick_params(length=0)
for s in ax6.spines.values():
    s.set_visible(False)
title(ax6, "6 · Cell-by-gene matrix", "transcripts counted per cell and gene")

arrows(fig)
fig.savefig(os.path.join(OUT, "pipeline_b.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote pipeline_a.png, pipeline_b.png to", OUT, "| transcripts:", len(pts_xy), "cells:", len(centers))

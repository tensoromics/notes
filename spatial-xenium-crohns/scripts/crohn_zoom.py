#!/usr/bin/env python3
"""One section of Crohn's ileum at three scales: tissue -> region -> cells.

Writes crohn_zoom.png: (A) a whole section, every cell at its centroid, colored by the
authors' megagut level-1 type; (B) a ~1 mm region of it; (C) a ~150 um patch inside that
region, over the ulcer, with each cell drawn as an equal-area circle, r = sqrt(cell_area / pi), so the
packing is visible at true size. Boxes and connector lines tie each panel to the next.

Usage: python crohn_zoom.py <crohn_obs.csv.gz> [out_dir]
"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, ConnectionPatch

CSV = sys.argv[1]
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else
                      os.path.join(os.path.dirname(__file__), "..", "figures"))
TYPE = "megagut_lvl1_majority_voting"; INK, BOX = "#1b2024", "#1b2024"
SEC, REG, PATCH = "0043587_affected", 1000.0, 150.0     # section, region and patch widths (um)
df = pd.read_csv(CSV, low_memory=False)
types = list(df[TYPE].value_counts().index)                # same order/palette as the other figures
tab = plt.get_cmap("tab10"); colmap = {t: tab(i % 10) for i, t in enumerate(types)}

g = df[df.biosample_id == SEC].reset_index(drop=True)
xy = g[["x_centroid", "y_centroid"]].to_numpy(); lab = g[TYPE].to_numpy()
r = np.sqrt(g.cell_area.to_numpy() / np.pi)

# The 1 mm region and 150 um patch over the ulcer, chosen by choose_ulcer_patch.py: the patch is the
# 150 um window, fully inside tissue, with the most COL7A1hi inflammatory fibroblasts (the authors'
# annot_new labels); the region is the 1 mm square around it, patch >= 150 um from every edge,
# with the most tissue in it.
rx0, ry0 = 3679.39, 606.96
px0, py0 = 4354.39, 756.96
inR = (xy[:, 0] >= rx0) & (xy[:, 0] < rx0 + REG) & (xy[:, 1] >= ry0) & (xy[:, 1] < ry0 + REG)
pad = 12.0                                                # include cells whose disks reach in
inP = ((xy[:, 0] > px0 - pad) & (xy[:, 0] < px0 + PATCH + pad) &
       (xy[:, 1] > py0 - pad) & (xy[:, 1] < py0 + PATCH + pad))
print(f"section {SEC}: {len(g):,} cells | region {inR.sum():,} cells at ({rx0:.0f},{ry0:.0f}) | "
      f"patch {int(((xy[:,0]>=px0)&(xy[:,0]<px0+PATCH)&(xy[:,1]>=py0)&(xy[:,1]<py0+PATCH)).sum())} "
      f"cells at ({px0:.0f},{py0:.0f})")

fig = plt.figure(figsize=(12.5, 5.9))
axA = fig.add_axes([0.00, 0.17, 0.36, 0.74]); axB = fig.add_axes([0.37, 0.17, 0.30, 0.74])
axC = fig.add_axes([0.69, 0.17, 0.30, 0.74])

def scatter(ax, m, s):
    for t in types:
        k = m & (lab == t)
        if k.any():
            ax.scatter(xy[k, 0], xy[k, 1], s=s, color=colmap[t], linewidths=0,
                       rasterized=True, alpha=0.85)

def finish(ax, x0, y0, w, h, bar, txt, title):
    ax.set_xlim(x0, x0 + w); ax.set_ylim(y0 + h, y0)        # image orientation: y down
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    bx, by = x0 + 0.04 * w, y0 + 0.965 * h
    ax.plot([bx, bx + bar], [by, by], color=INK, lw=2.6, solid_capstyle="butt")
    ax.text(bx + bar / 2, by - 0.015 * h, txt, ha="center", va="bottom", fontsize=12, color=INK,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_title(title, fontsize=15, color=INK, loc="left")

# A: whole section
scatter(axA, np.ones(len(g), bool), 0.35)
mn, mx = xy.min(0), xy.max(0); span = (mx - mn).max() * 1.02
finish(axA, mn[0] - 0.01 * span, mn[1] - 0.01 * span, span, span, 1000, "1 mm",
       f"A · whole section · {len(g):,} cells")
axA.add_patch(Rectangle((rx0, ry0), REG, REG, fill=False, ec=BOX, lw=1.4))

# B: region
scatter(axB, inR, 3.0)
finish(axB, rx0, ry0, REG, REG, 200, "200 µm", f"B · 1 mm square · {inR.sum():,} cells")
axB.add_patch(Rectangle((px0, py0), PATCH, PATCH, fill=False, ec=BOX, lw=1.4))

# C: patch, cells at true size
for a in np.where(inP)[0]:
    axC.add_patch(Circle((xy[a, 0], xy[a, 1]), r[a], facecolor=colmap.get(lab[a], "#ccc"),
                         edgecolor="white", lw=0.6, alpha=0.75))
nP = int(((xy[:, 0] >= px0) & (xy[:, 0] < px0 + PATCH) & (xy[:, 1] >= py0) &
          (xy[:, 1] < py0 + PATCH)).sum())
finish(axC, px0, py0, PATCH, PATCH, 20, "20 µm", f"C · 150 µm patch · {nP} cells")

# connectors: box corners in one panel -> panel edges in the next
for (src, x0, y0, w, dst) in [(axA, rx0, ry0, REG, axB), (axB, px0, py0, PATCH, axC)]:
    for (sx, sy), (dx, dy) in [((x0 + w, y0), (0, 1)), ((x0 + w, y0 + w), (0, 0))]:
        fig.add_artist(ConnectionPatch((sx, sy), (dx, dy), "data", "axes fraction",
                                       axesA=src, axesB=dst, color=BOX, lw=0.8, alpha=0.6))

handles = [plt.Line2D([0], [0], marker="o", ls="", ms=10, color=colmap[t]) for t in types]
fig.legend(handles, types, loc="lower center", ncol=4, fontsize=13, frameon=False,
           bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.4)
fig.savefig(os.path.join(OUT, "crohn_zoom.png"), dpi=200, bbox_inches="tight", facecolor="white")
print("wrote crohn_zoom.png to", OUT)

#!/usr/bin/env python3
"""Part 2, Figure 5: packed cells in the stain images, and how the equal-area circles compare.

The Crohn's atlas didn't share its stain images or outlines, so this figure uses a public 10x sample
made the same way: Xenium Prime FFPE Human Skin (skin melanoma; the Xenium Prime 5K panel, with all
5,001 genes of the atlas; the same three-stain segmentation).
https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-skin

The code below is the note's block "circles against real outlines (skin sample)", unchanged. It
expects, in <data_dir>/Xenium_Prime_Human_Skin_FFPE/:
  Xenium_Prime_Human_Skin_FFPE_cells.parquet
  Xenium_Prime_Human_Skin_FFPE_cell_boundaries.parquet
  morphology_focus/  (from Xenium_Prime_Human_Skin_FFPE_xe_outs.zip, 3.1 GB)

Usage: python p2_stain_fig.py <data_dir> [out_dir]      -> p2_stain.png
Needs: numpy, pandas, scipy, matplotlib (3.7+), pyarrow, shapely, tifffile, zarr
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
os.makedirs(OUT, exist_ok=True)
os.chdir(sys.argv[1])

import shapely
import tifffile
import zarr
from matplotlib.patches import Polygon

# the folder with the downloaded files (see Reproduce); every file name starts with the sample name
sample = "Xenium_Prime_Human_Skin_FFPE"
skin_cells = pd.read_parquet(f"{sample}/{sample}_cells.parquet")
outlines = pd.read_parquet(f"{sample}/{sample}_cell_boundaries.parquet")

# the outlines come as one row per corner (in µm); join each cell's corners into a polygon,
# in the same order as the cell table
corners = {c: g[["vertex_x", "vertex_y"]].to_numpy() for c, g in outlines.groupby("cell_id", sort=False)}
skin_ids = skin_cells.cell_id.to_numpy()
polygon = np.array([shapely.polygons(corners[c]) for c in skin_ids], dtype=object)

# the same test as for the atlas: the nearest cell, and whether the equal-area circles overlap
# (one section, so one tree)
sxy = skin_cells[["x_centroid", "y_centroid"]].to_numpy()
sr = np.sqrt(skin_cells.cell_area.to_numpy() / np.pi)
dist, j = cKDTree(sxy).query(sxy, k=2)
snn, sd = j[:, 1], dist[:, 1]
circles = sd < sr + sr[snn]
# the test the atlas doesn't allow: are the real outlines of the two cells less than 0.25 µm apart?
touch = shapely.distance(polygon, polygon[snn]) <= 0.25

print(f"{len(skin_cells):,} cells; median distance to the nearest cell {np.median(sd):.1f} µm, "
      f"median diameter {np.median(2 * sr):.1f} µm")
print(f"circles overlap: {circles.mean():.0%}; outlines touch: {touch.mean():.0%}; "
      f"the two agree for {(circles == touch).mean():.0%} of cells")
print(f"when the circles overlap, the outlines touch {touch[circles].mean():.0%} of the time; "
      f"when they don't, {touch[~circles].mean():.0%}")

# the stain images for one 50 µm square: pixel = µm / 0.2125; channels 0-2 are DAPI (nuclei),
# the boundary stain, and 18S (the cell interior)
x0, y0, w, px = 8400, 1600, 50, 0.2125
image = zarr.open(tifffile.imread(f"{sample}/morphology_focus/morphology_focus_0000.ome.tif",
                                  aszarr=True), mode="r")["0"]
crop = np.asarray(image[:3, round(y0 / px):round((y0 + w) / px),
                        round(x0 / px):round((x0 + w) / px)]).astype(float)
# scale each stain from its 1st to its 99.7th percentile, so that faint and bright stains both show
lo, hi = np.percentile(crop, [1, 99.7], axis=(1, 2))
dapi, boundary, interior = np.clip((crop - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1)
# red: 18S, green: boundary stain, blue: DAPI
rgb = np.dstack([0.85 * interior, boundary, 0.95 * dapi])
extent = [x0, x0 + w, y0 + w, y0]

# the cells in the square, and the cells around it (so that circles at the edge are complete)
def in_box(margin):
    return np.flatnonzero((sxy[:, 0] >= x0 - margin) & (sxy[:, 0] < x0 + w + margin) &
                          (sxy[:, 1] >= y0 - margin) & (sxy[:, 1] < y0 + w + margin))
inside, around = in_box(0), in_box(12)
print(f"the square: {len(inside)} cells")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), layout="constrained")
# A: the stain images; B: the outlines on a darker copy; C: the circles on a darker copy still
axes[0].imshow(rgb, extent=extent)
axes[1].imshow(rgb * 0.55, extent=extent)
for i in around:
    axes[1].add_patch(Polygon(corners[skin_ids[i]], fill=False, edgecolor="white", lw=1.3))
axes[2].imshow(rgb * 0.35, extent=extent)
for i in around:
    axes[2].add_patch(Circle(sxy[i], sr[i], facecolor="white", alpha=0.18, edgecolor="white", lw=1))
# a line from each cell in the square to its nearest cell: red if the circles overlap
for i in inside:
    axes[2].plot(sxy[[i, snn[i]], 0], sxy[[i, snn[i]], 1], lw=1.6,
                 color="#ff4d4d" if circles[i] else "#bbbbbb")
titles = ["A · Stain images", "B · Outlines Xenium traced from the stains",
          "C · Equal-area circles, as in Figures 3 and 4"]
for ax, title in zip(axes, titles):
    ax.set_xlim(x0, x0 + w)
    ax.set_ylim(y0 + w, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left")
    # a 10 µm scale bar
    ax.plot([x0 + 3, x0 + 13], [y0 + w - 3, y0 + w - 3], color="white", lw=3)
    ax.text(x0 + 8, y0 + w - 4, "10 µm", ha="center", va="bottom", color="white")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#2a2af2", "#22dd22", "#d92a2a")]
handles += [plt.Line2D([0], [0], color=c, lw=2) for c in ("#ff4d4d", "#bbbbbb")]
fig.legend(handles, ["DAPI (nuclei)", "boundary stain", "18S (cell interior)",
                     "to the nearest cell: circles overlap", "circles don't overlap"],
           loc="outside lower center", ncol=5, frameon=False)
fig.savefig(os.path.join(OUT, "p2_stain.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", os.path.join(OUT, "p2_stain.png"))

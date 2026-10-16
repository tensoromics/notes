#!/usr/bin/env python3
"""Choose the zoom targets for Note 1 in section 0043587_affected:

  patch : the 150 um window, fully inside tissue (every 50 um sub-square holds cells), with the
          most COL7A1hi inflammatory fibroblasts (the authors' annot_new labels)
  region: the 1 mm square around that patch, with the patch >= 150 um from every edge,
          covering the most tissue (share of occupied 50 um squares)

Usage: python choose_ulcer_patch.py <crohn_obs.csv.gz>   (from extract_crohn_obs.py)
"""
import sys
import numpy as np, pandas as pd

SEC, IF = "0043587_affected", "M COL7A1hi inflammatory fibroblast"
d = pd.read_csv(sys.argv[1], low_memory=False,
                usecols=["biosample_id", "x_centroid", "y_centroid", "annot_new"])
s = d[d.biosample_id == SEC]
xy = s[["x_centroid", "y_centroid"]].to_numpy()
is_if = (s.annot_new == IF).to_numpy()

# patch: 25 um grid, sliding 150 um windows (6 x 6 bins), counted with 2D cumulative sums
G, kP = 25.0, 6
o = xy.min(0)
b = np.floor((xy - o) / G).astype(int)
N = np.zeros(b.max(0) + 1); F = np.zeros_like(N)
np.add.at(N, (b[:, 0], b[:, 1]), 1); np.add.at(F, (b[is_if, 0], b[is_if, 1]), 1)
def winsum(A, k):
    c = np.pad(A.cumsum(0).cumsum(1), [(1, 0), (1, 0)])
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
occ50 = (winsum((N > 0).astype(float), 2) > 0).astype(float)      # 50 um squares holding cells
full = winsum(occ50, kP - 1) == (kP - 1) ** 2
i, j = np.unravel_index(np.argmax(np.where(full, winsum(F, kP), -1)), full.shape)
px, py = o[0] + i * G, o[1] + j * G

# region: 1 mm squares that keep the patch >= 150 um from every edge, scored by tissue cover
def cover(x0, y0, w=1000.0, g=50.0):
    m = (xy[:, 0] >= x0) & (xy[:, 0] < x0 + w) & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + w)
    return len({tuple(v) for v in np.floor((xy[m] - [x0, y0]) / g).astype(int)}) / (w / g) ** 2
cands = [(cover(rx, ry), rx, ry) for rx in np.arange(px + 300 - 1000, px - 150 + 1, 25)
         for ry in np.arange(py + 300 - 1000, py - 150 + 1, 25)]
c, rx, ry = max(cands)
print(f"patch  x0={px:.2f} y0={py:.2f}  inflammatory fibroblasts: "
      f"{int(winsum(F, kP)[i, j])} of {int(winsum(N, kP)[i, j])} cells")
print(f"region x0={rx:.2f} y0={ry:.2f}  tissue cover {c:.0%}")

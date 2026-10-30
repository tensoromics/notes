#!/usr/bin/env python3
"""Part 2 method schematic: how we decide that a cell overlaps its nearest neighbor.

  1  real outlines (made up): find the cell whose center is closest to cell A
  2  replace each cell by a circle with the same area, around the same center
  3  A and its nearest cell overlap when the distance d is less than r_A + r_B

Synthetic: cells A and B share a straight edge (like two touching outlines); C and D are farther.
All numbers printed on the figure are computed from these shapes.
Usage: python p2_method_schematic.py [out_dir]      -> method_schematic.png
"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                      os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(OUT, exist_ok=True)
INK, CELL, RED, MUTED = "#1b2024", "#6b727a", "#d62728", "#9aa1a8"

def blob(c, R, n=200, wobble=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = 1 + wobble * (np.sin(3 * t + rng.uniform(0, 6)) * 0.6 + np.cos(5 * t + rng.uniform(0, 6)) * 0.4)
    return np.c_[c[0] + R * k * np.cos(t), c[1] + R * k * np.sin(t)]

def clip(poly, x0, keep_left):
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        pin, qin = (p[0] <= x0, q[0] <= x0) if keep_left else (p[0] >= x0, q[0] >= x0)
        if pin:
            out.append(p)
        if pin != qin:
            out.append(p + (x0 - p[0]) / (q[0] - p[0]) * (q - p))
    return np.array(out)

def area_centroid(P):
    x, y = P[:, 0], P[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cr.sum() / 2
    return abs(a), np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * a)

# four made-up cells, in µm: A and B touch along the line x = 3.4
cells = {"A": clip(blob((0, 0), 4.6, wobble=0.05, seed=1), 3.4, True),
         "B": clip(blob((6.2, 0.4), 3.7, wobble=0.05, seed=2), 3.4, False),
         "C": blob((-3.0, 9.6), 3.4, wobble=0.10, seed=3),
         "D": blob((-9.8, -4.6), 3.6, wobble=0.10, seed=4)}
geo = {k: area_centroid(P) for k, P in cells.items()}
ctr = {k: g[1] for k, g in geo.items()}
rad = {k: np.sqrt(g[0] / np.pi) for k, g in geo.items()}
dist = {k: np.linalg.norm(ctr[k] - ctr["A"]) for k in "BCD"}
assert min(dist, key=dist.get) == "B"
d, rA, rB = dist["B"], rad["A"], rad["B"]
print({k: round(g[0], 1) for k, g in geo.items()}, {k: round(v, 2) for k, v in dist.items()},
      round(rA, 2), round(rB, 2))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw=dict(wspace=0.08))
LIM = dict(x=(-14.5, 11.5), y=(-9.5, 14.5))
BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)

def frame(ax, title, lim=LIM):
    ax.set_xlim(*lim["x"]); ax.set_ylim(*lim["y"]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=15, color=INK, loc="left")

def label(ax, k, dy=0.0):
    ax.text(ctr[k][0], ctr[k][1] + 0.9 + dy, k, ha="center", va="bottom", fontsize=13,
            color=INK, fontweight="bold")

# 1 · the real outlines and the distances from A's center
ax = axes[0]
for k, P in cells.items():
    ax.add_patch(Polygon(P, closed=True, facecolor=CELL, alpha=0.13, edgecolor="none"))
    ax.add_patch(Polygon(P, closed=True, facecolor="none", edgecolor=CELL, lw=1.8))
for k, off, ha in [("C", (-0.6, 0.0), "right"), ("D", (0.0, -1.1), "center")]:
    ax.plot(*zip(ctr["A"], ctr[k]), color=MUTED, lw=1.4, ls="--")
    m = (ctr["A"] + ctr[k]) / 2
    ax.text(m[0] + off[0], m[1] + off[1], f"{dist[k]:.1f} µm", fontsize=11, color=MUTED,
            ha=ha, va="center", bbox=BOX)
ax.plot(*zip(ctr["A"], ctr["B"]), color=RED, lw=2.2)
ax.text((ctr["A"][0] + ctr["B"][0]) / 2, ctr["A"][1] - 5.6, f"nearest: d = {d:.1f} µm", ha="center",
        va="top", fontsize=12, color=RED, fontweight="bold")
for k in cells:
    ax.plot(*ctr[k], "o", color=INK, ms=5)
    label(ax, k)
frame(ax, "1 · Find the nearest cell")

# 2 · each cell as a circle with the same area
ax = axes[1]
for k, P in cells.items():
    ax.add_patch(Polygon(P, closed=True, facecolor="none", edgecolor=MUTED, lw=1.0, ls=":"))
    ax.add_patch(Circle(ctr[k], rad[k], facecolor=CELL, alpha=0.13, edgecolor="none"))
    ax.add_patch(Circle(ctr[k], rad[k], facecolor="none", edgecolor=CELL, lw=1.8))
    ax.plot(*ctr[k], "o", color=INK, ms=5)
    ax.text(ctr[k][0], ctr[k][1] - rad[k] - 0.5, f"{geo[k][0]:.0f} µm²", ha="center", va="top",
            fontsize=11, color=INK, bbox=BOX)
    label(ax, k)
frame(ax, "2 · A circle with the same area")

# 3 · the overlap test for A and B, zoomed in
ax = axes[2]
for k in "AB":
    ax.add_patch(Circle(ctr[k], rad[k], facecolor=CELL, alpha=0.13, edgecolor="none"))
    ax.add_patch(Circle(ctr[k], rad[k], facecolor="none", edgecolor=CELL, lw=1.8))
    ax.plot(*ctr[k], "o", color=INK, ms=5)
# radii: straight up from each center, labelled outside the circles
for k, r, sub, dx, ha in [("A", rA, "A", -0.4, "right"), ("B", rB, "B", 0.4, "left")]:
    ax.annotate("", ctr[k] + [0, r], ctr[k], arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(ctr[k][0] + dx, ctr[k][1] + r * 0.55, f"r$_{sub}$ = {r:.1f}", ha=ha, va="center",
            fontsize=12, color=INK, bbox=BOX)
    ax.text(ctr[k][0], ctr[k][1] + r + 0.5, k, ha="center", va="bottom", fontsize=13, color=INK,
            fontweight="bold")
ax.annotate("", ctr["B"], ctr["A"], arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=2.0,
                                                    shrinkA=0, shrinkB=0))
ax.text((ctr["A"][0] + ctr["B"][0]) / 2, ctr["A"][1] - 0.6, f"d = {d:.1f}", ha="center", va="top",
        fontsize=12, color=RED, fontweight="bold", bbox=BOX)
mid = (ctr["A"][0] + ctr["B"][0]) / 2
ax.text(mid, -6.2, f"d = {d:.1f} µm  <  r$_A$ + r$_B$ = {rA + rB:.1f} µm", ha="center", va="top",
        fontsize=12.5, color=INK)
ax.text(mid, -8.0, "so A overlaps its nearest cell", ha="center", va="top", fontsize=12.5, color=INK)
frame(ax, "3 · Overlap if d < r$_A$ + r$_B$",
      lim=dict(x=(mid - 13, mid + 13), y=(-9.5 - 0.0, 14.5)))

fig.savefig(os.path.join(OUT, "method_schematic.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", os.path.join(OUT, "method_schematic.png"))

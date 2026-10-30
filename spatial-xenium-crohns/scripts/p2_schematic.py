#!/usr/bin/env python3
"""Part 2 intro schematic: from dots to gene counts, and how a dot gets counted for the wrong cell.

  A  what Xenium sees: a stain image, and dots that each know their gene but not their cell;
     where the two cells meet, the stain doesn't show a clear edge
  B  segmentation: outlines traced from the stain alone, the dots are not used
  C  counting: the dots of each gene inside each outline; dots on the other cell's side of the
     edge are counted for the other cell (in both directions); the counts are under the panel

Synthetic and deterministic (fixed seed); the colors match pipeline_fig_v2.py.
Usage: python p2_schematic.py [out_dir]      -> overlap_schematic.png
"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                      os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(OUT, exist_ok=True)
INK, CELL, WRONG, MUTED = "#1b2024", "#6b727a", "#d62728", "#6b727a"
EPI, TCELL = "#1C8577", "#7A5DA9"                 # EPCAM dots (epithelial), CD3E dots (T cell)
rng = np.random.default_rng(3)

def ellipse(c, a, b, n=120):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[c[0] + a * np.cos(t), c[1] + b * np.sin(t)]

def inside_ellipse(p, c, a, b):
    return ((p[:, 0] - c[0]) / a) ** 2 + ((p[:, 1] - c[1]) / b) ** 2 <= 1

def dots(c, a, b, n):
    # uniform dots inside the true cell
    out = []
    while len(out) < n:
        p = c + rng.uniform(-1, 1, 2) * [a, b]
        if inside_ellipse(p[None], c, a, b)[0]:
            out.append(p)
    return np.array(out)

def clip_halfplane(poly, x0, keep_left):
    # keep the part of the polygon with x <= x0 (left) or x >= x0 (right)
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        pin = p[0] <= x0 if keep_left else p[0] >= x0
        qin = q[0] <= x0 if keep_left else q[0] >= x0
        if pin:
            out.append(p)
        if pin != qin:
            t = (x0 - p[0]) / (q[0] - p[0])
            out.append(p + t * (q - p))
    return np.array(out)

# two touching cells: an epithelial cell and a T cell, in µm
A_EPI, B_EPI, A_T, B_T = 5.2, 4.2, 3.6, 3.4        # cell half-widths and half-heights
ce, ct = np.array([0.0, 0.0]), np.array([6.0, 0.0])
de, dt = dots(np.zeros(2), A_EPI, B_EPI, 26) + ce, dots(np.zeros(2), A_T, B_T, 20) + ct

# the outline edge between the cells: halfway between the two cells' far edges
edge = (ce[0] + A_EPI + ct[0] - A_T) / 2
oe = clip_halfplane(ellipse(ce, A_EPI, B_EPI), edge, True)
ot = clip_halfplane(ellipse(ct, A_T, B_T), edge, False)
we, wt = de[:, 0] > edge, dt[:, 0] < edge          # dots on the other cell's side

# a made-up stain image: a blurred ring along each cell's edge, plus its nucleus;
# where the two cells meet the rings run into each other, so there is no clear edge
mid = (ce[0] - A_EPI + ct[0] + A_T) / 2            # center of the pair, for the panel limits
X, Y = np.meshgrid(np.linspace(mid - 11, mid + 11, 440), np.linspace(-B_EPI - 5.4, B_EPI + 2.6, 160))
def ring(c, a, b, width=0.55):
    rho = np.sqrt(((X - c[0]) / a) ** 2 + ((Y - c[1]) / b) ** 2)
    return np.exp(-((rho - 1) * min(a, b) / width) ** 2)
def blob(c, R):
    return np.exp(-(((X - c[0]) ** 2 + (Y - c[1]) ** 2) / R ** 2))
stain = (ring(ce, A_EPI, B_EPI) + ring(ct, A_T, B_T) + 0.6 * blob(ce - [0.8, 0], 1.9)
         + 0.6 * blob(ct + [0.4, 0], 1.5))
# where the two rings cross, don't let them add up to a dark spot
stain = np.minimum(stain, 1.0)
# the shared part is dimmer: two membranes pressed together and out of focus
def soft_inside(c, a, b, k=6.0):
    rho = np.sqrt(((X - c[0]) / a) ** 2 + ((Y - c[1]) / b) ** 2)
    return 1 / (1 + np.exp(-k * (1.12 - rho)))
both = soft_inside(ce, A_EPI, B_EPI) * soft_inside(ct, A_T, B_T)
stain = stain * (1 - 0.55 * both) + 0.25 * both
EXTENT = [X.min(), X.max(), Y.min(), Y.max()]

def show_stain(ax, alpha=1.0):
    ax.imshow(stain, extent=EXTENT, origin="lower", cmap="Greys", vmin=0, vmax=1.6, alpha=alpha,
              interpolation="bilinear")

def show_outlines(ax):
    for o in (oe, ot):
        ax.add_patch(Polygon(o, closed=True, facecolor=CELL, alpha=0.12, edgecolor="none"))
        ax.add_patch(Polygon(o, closed=True, facecolor="none", edgecolor=INK, lw=1.8))

def show_dots(ax):
    ax.scatter(de[:, 0], de[:, 1], s=46, color=EPI, edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter(dt[:, 0], dt[:, 1], s=46, color=TCELL, edgecolors="white", linewidths=0.5, zorder=3)

def frame(ax, title, note):
    ax.set_title(title, fontsize=15, color=INK, loc="left", pad=10)
    ax.text(mid, B_EPI + 1.0, note, ha="center", va="bottom", fontsize=11.5, color=MUTED)
    ax.set_xlim(mid - 11, mid + 11); ax.set_ylim(-B_EPI - 6.0, B_EPI + 2.6)
    ax.set_aspect("equal"); ax.axis("off")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw=dict(wspace=0.06))

# A · what Xenium sees
ax = axes[0]
show_stain(ax)
show_dots(ax)
ax.annotate("no clear edge", (edge, -B_T + 0.2), (edge, -B_EPI - 2.4), ha="center", va="top",
            fontsize=11.5, color=INK, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
frame(ax, "A · What Xenium sees", "stain image, and dots: gene known, cell unknown")

# B · segmentation from the stain alone
ax = axes[1]
show_stain(ax)
show_outlines(ax)
ax.text(edge - 0.6, -0.2, "epithelial\ncell", ha="right", va="center", fontsize=12, color=INK)
ax.text(edge + 0.6, -0.2, "T cell", ha="left", va="center", fontsize=12, color=INK)
frame(ax, "B · Outlines from the stain", "the dots are not used")

# C · counting the dots inside each outline
ax = axes[2]
show_outlines(ax)
show_dots(ax)
for d, w in ((de, we), (dt, wt)):
    ax.scatter(d[w, 0], d[w, 1], s=150, facecolors="none", edgecolors=WRONG, linewidths=1.8, zorder=4)
ne, nt = int(we.sum()), int(wt.sum())
# the counts: a small piece of the count matrix, one row per cell and one column per gene
rows = [("", "EPCAM", "CD3E", "…"), ("epithelial cell", len(de) - ne, nt, "…"), ("T cell", ne, len(dt) - nt, "…")]
y0, x_cols = -B_EPI - 2.0, [mid - 10.2, mid + 0.4, mid + 5.0, mid + 8.8]
for r, row in enumerate(rows):
    for c, v in enumerate(row):
        bad = (r == 1 and c == 2 and nt) or (r == 2 and c == 1 and ne)
        ax.text(x_cols[c], y0 - 1.35 * r, str(v), ha="left" if c == 0 else "center", va="top",
                fontsize=12.5, color=WRONG if bad else (MUTED if c == 3 else INK),
                fontweight="bold" if (r == 0 and c < 3) or bad else None)
# a bracket over the gene columns: 5,001 genes in all
yb = y0 + 0.35
ax.plot([x_cols[1] - 1.6, x_cols[1] - 1.6, x_cols[3] + 1.0, x_cols[3] + 1.0],
        [yb - 0.25, yb, yb, yb - 0.25], color=MUTED, lw=1.0)
ax.text((x_cols[1] + x_cols[3]) / 2 - 0.3, yb + 0.15, "5,001 genes", ha="center", va="bottom",
        fontsize=11, color=MUTED)
frame(ax, "C · Dots counted per outline", "each outline's dots, gene by gene")

handles = [plt.Line2D([0], [0], marker="o", ls="", ms=9, color=EPI, mec="white"),
           plt.Line2D([0], [0], marker="o", ls="", ms=9, color=TCELL, mec="white"),
           plt.Line2D([0], [0], marker="o", ls="", ms=12, mfc="none", mec=WRONG, mew=1.8),
           Polygon([[0, 0]], facecolor=CELL, alpha=0.25, edgecolor=INK)]
fig.legend(handles, ["EPCAM dot (made by the epithelial cell)", "CD3E dot (made by the T cell)",
                     "counted for the wrong cell", "cell outline (from the stain image)"],
           loc="lower center", ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.savefig(os.path.join(OUT, "overlap_schematic.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", os.path.join(OUT, "overlap_schematic.png"))

#!/usr/bin/env python3
"""Dataset summary for Note 1, sized for the text column: cells per section (inflamed vs
adjacent, labeled patient · piece), cell-size distribution, and cell-type composition.

Reads the cell table straight from the h5ad with h5py (the expression matrix is never loaded).
Writes crohn_summary.png.

Usage: python crohn_summary.py <xenium5k_...SCP.h5ad> [out_dir]
"""
import sys, os, re
import numpy as np, pandas as pd, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

H5AD = sys.argv[1]
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else
                      os.path.join(os.path.dirname(__file__), "..", "figures"))
INK = "#1b2024"; PIECE = {"inflamed": "#AC4130", "adjacent": "#2C6E9B"}
plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#7A828C"})

obs = h5py.File(H5AD, "r")["obs"]
def column(name):
    g = obs[name]
    return g[:] if isinstance(g, h5py.Dataset) else g["categories"].asstr()[:][g["codes"][:]]
cells = pd.DataFrame({"section": column("biosample_id"), "area": column("cell_area"),
                      "cell_type": column("megagut_lvl1_majority_voting")})

# ---- per-section label "patient · piece"; patient 0060032 has two specimens (gi9662, gi10056) ----
def label(sec):
    piece = "inflamed" if re.search(r"inf|affected", sec, re.I) else "adjacent"
    pid = sec[:7]
    spec = re.search(r"gi\d+", sec)
    if pid == "0060032" and spec:
        pid += f" ({spec.group(0)})"
    return f"{pid} · {piece}", piece
per = cells.section.value_counts()
lab = pd.DataFrame([(*label(s), n) for s, n in per.items()], columns=["label", "piece", "n"])
lab = lab.sort_values("n")

types = list(cells.cell_type.value_counts().index)
tab = plt.get_cmap("tab10"); colmap = {t: tab(i % 10) for i, t in enumerate(types)}
diam = 2 * np.sqrt(cells.area.to_numpy() / np.pi)

fig, ax = plt.subplots(1, 3, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})

# (A) cells per section
ax[0].barh(range(len(lab)), lab.n / 1000, color=[PIECE[p] for p in lab.piece], height=0.72)
ax[0].set_yticks(range(len(lab))); ax[0].set_yticklabels(lab.label, fontsize=10)
ax[0].set_xlabel("cells per section (thousands)")
ax[0].set_title(f"16 sections · {len(cells) / 1e6:.1f} million cells", fontsize=13)
ax[0].legend([plt.Rectangle((0, 0), 1, 1, color=PIECE[k]) for k in PIECE], list(PIECE),
             loc="lower right", frameon=False, fontsize=11)

# (B) cell size
ax[1].hist(diam, bins=60, range=(0, 24), color="#8FB7C7", edgecolor="none")
med = np.median(diam)
ax[1].axvline(med, color=INK, ls="--", lw=1.3)
ax[1].text(med + 0.7, ax[1].get_ylim()[1] * 0.9, f"median {med:.1f} µm", fontsize=11, color=INK)
ax[1].set_xlabel("cell diameter (µm)")
ax[1].set_ylabel("cells (thousands)")
ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
ax[1].set_title("How big is a cell?", fontsize=13)

# (C) cell-type composition
comp = cells.cell_type.value_counts(normalize=True) * 100
ax[2].barh(range(len(comp)), comp.values[::-1], color=[colmap[t] for t in comp.index[::-1]], height=0.72)
ax[2].set_yticks(range(len(comp))); ax[2].set_yticklabels(comp.index[::-1], fontsize=11)
for i, v in enumerate(comp.values[::-1]):
    ax[2].text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=10, color=INK)
ax[2].set_xlabel("% of all cells"); ax[2].set_xlim(0, comp.max() * 1.18)
ax[2].set_title("Cell types", fontsize=13)

for a in ax:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout(w_pad=2.5)
fig.savefig(os.path.join(OUT, "crohn_summary.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote crohn_summary.png to", OUT, f"| median diameter {med:.2f} µm")

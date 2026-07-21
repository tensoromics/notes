"""Part 4 figure: the SAME embedding t-SNE, colored two ways — by cell lineage
vs by disease state. Reuses cached coords (data/tsne_coords.npz); no recompute.
Shows the map is organized by identity (clean lineage islands) while disease
state is spread across it (magnitude, not a distinct cluster structure).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from celltype_tsne import lineage, LINEAGE_ORDER, LCOLOR

DIS_ORDER = ["Healthy", "Non-inflamed", "Inflamed"]
DCOLOR = {"Healthy": "#2563eb", "Non-inflamed": "#9aa3af", "Inflamed": "#dc2626"}


def main():
    z = np.load("data/tsne_coords.npz")
    Ze, idx = z["Ze"], z["idx"]
    meta = np.load("data/meta.npz", allow_pickle=True)
    lin = np.array([lineage(c) for c in meta["celltype"][idx]])
    dis = np.array(meta["label"][idx], dtype=object)
    import collections
    print("disease counts (subsample):", dict(collections.Counter(dis)))

    INK, MUTED = "#0f172a", "#64748b"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4))

    # left: by lineage
    ax = axes[0]
    for lg in LINEAGE_ORDER:
        m = lin == lg
        if m.any():
            ax.scatter(Ze[m, 0], Ze[m, 1], s=3.5, c=LCOLOR[lg], linewidths=0,
                       alpha=0.65, rasterized=True)
    ax.set_title("coloured by cell lineage", fontsize=12, fontweight="bold",
                 color=INK, loc="left", pad=6)
    h1 = [plt.Line2D([0], [0], marker="o", ls="", ms=6.5, mfc=LCOLOR[lg], mec="none",
          label=lg) for lg in LINEAGE_ORDER if (lin == lg).any()]
    ax.legend(handles=h1, loc="upper left", frameon=False, fontsize=8.2,
              handletextpad=0.2, labelspacing=0.25, borderpad=0.1)

    # right: by disease state (same coords)
    ax = axes[1]
    order = np.argsort([DIS_ORDER.index(d) for d in dis])  # inflamed drawn on top
    for d in DIS_ORDER:
        m = dis == d
        if m.any():
            ax.scatter(Ze[m, 0], Ze[m, 1], s=3.5, c=DCOLOR[d], linewidths=0,
                       alpha=0.55, rasterized=True)
    ax.set_title("coloured by disease state", fontsize=12, fontweight="bold",
                 color=INK, loc="left", pad=6)
    h2 = [plt.Line2D([0], [0], marker="o", ls="", ms=6.5, mfc=DCOLOR[d], mec="none",
          label=d) for d in DIS_ORDER]
    ax.legend(handles=h2, loc="upper left", frameon=False, fontsize=9,
              handletextpad=0.2, labelspacing=0.3, borderpad=0.1)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#e2e8f0")
    fig.suptitle("The same map, two labels: organised by cell type, not by disease",
                 fontsize=14, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.945, f"one t-SNE of the frozen embedding ({len(idx):,} cells), "
             f"coloured by lineage (left) and disease state (right)",
             fontsize=10.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    fig.savefig("disease-tsne.png", dpi=140, facecolor="white", bbox_inches="tight")
    print("saved disease-tsne.png")


if __name__ == "__main__":
    main()

"""Part 3 figure: t-SNE of the frozen nano-Geneformer cell embedding vs PCA-50,
colored by the 7 major cell lineages. Visualizes the identity structure the
cell-type probe measured (embed acc 0.799 vs PCA 0.823). Uses the cached 100k
embedding; t-SNE coords are cached to data/tsne_coords.npz so re-rendering the
figure is free. Random 20k subsample (seeded) of all cells.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SUB = 20000
SEED = 0

# ---- 51 fine types -> 7 lineages (substring rules; Smillie 2019 compartments) --
def lineage(ct):
    s = ct.lower()
    if any(k in ct for k in ["TA", "Enterocyte", "Goblet", "Stem", "Tuft", "Best4",
                             "Secretory", "M cells", "MT-hi", "Enteroendocrine"]):
        return "Epithelial"
    if any(k in ct for k in ["CD4", "CD8", "Treg", "NK", "ILC", "Cycling T", "IEL"]):
        return "T / NK"
    if any(k in ct for k in ["Plasma", "Follicular", "Cycling B", "GC"]):
        return "B / Plasma"
    if any(k in ct for k in ["Macrophage", "DC1", "DC2", "Mono", "Mast"]):
        return "Myeloid"
    if any(k in ct for k in ["WNT2B", "WNT5B", "Fibroblast", "Myofibroblast", "RSPO3"]):
        return "Fibroblast / Stromal"
    if any(k in ct for k in ["Endothelial", "Venule", "Microvascular", "Pericyte"]):
        return "Endothelial / Vascular"
    if "Glia" in ct:
        return "Glia"
    return "Other"

LINEAGE_ORDER = ["Epithelial", "T / NK", "B / Plasma", "Myeloid",
                 "Fibroblast / Stromal", "Endothelial / Vascular", "Glia", "Other"]
# Okabe-Ito colorblind-safe palette (no yellow)
LCOLOR = {"Epithelial": "#0072B2", "T / NK": "#D55E00", "B / Plasma": "#009E73",
          "Myeloid": "#CC79A7", "Fibroblast / Stromal": "#E69F00",
          "Endothelial / Vascular": "#56B4E9", "Glia": "#111111", "Other": "#bbbbbb"}


def main():
    emb = np.load("data/emb_L320_mps.npy")
    pca = np.load("data/pca.npy")
    meta = np.load("data/meta.npz", allow_pickle=True)
    celltype = meta["celltype"]
    print(f"emb {emb.shape}  pca {pca.shape}  cells {len(celltype)}")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(celltype), size=min(N_SUB, len(celltype)), replace=False)
    lin = np.array([lineage(c) for c in celltype[idx]])
    # report mapping coverage
    import collections
    unmapped = sorted({c for c in celltype[idx] if lineage(c) == "Other"})
    print("lineage counts:", dict(collections.Counter(lin)))
    if unmapped:
        print("WARN unmapped ->Other:", unmapped)

    cache = "data/tsne_coords.npz"
    try:
        z = np.load(cache)
        Ze, Zp, cidx = z["Ze"], z["Zp"], z["idx"]
        assert np.array_equal(cidx, idx)
        print(f"loaded cached t-SNE coords {cache}")
    except Exception:
        from sklearn.manifold import TSNE
        def tsne(X, name):
            import time; t = time.time()
            Z = TSNE(n_components=2, perplexity=30, init="pca",
                     random_state=SEED, n_jobs=-1).fit_transform(X[idx].astype(np.float32))
            print(f"  t-SNE {name}: {time.time()-t:.0f}s")
            return Z
        print("computing t-SNE (embedding, then PCA) ...")
        Ze = tsne(emb, "embedding"); Zp = tsne(pca, "PCA")
        np.savez(cache, Ze=Ze, Zp=Zp, idx=idx)
        print(f"cached {cache}")

    # ---- plot: two panels, shared lineage legend --------------------------------
    INK, MUTED = "#0f172a", "#64748b"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2))
    for ax, Z, title in [(axes[0], Ze, "nano-Geneformer embedding  (256-d)"),
                         (axes[1], Zp, "PCA-50 baseline")]:
        for lg in LINEAGE_ORDER:
            m = lin == lg
            if not m.any():
                continue
            ax.scatter(Z[m, 0], Z[m, 1], s=3.5, c=LCOLOR[lg], linewidths=0,
                       alpha=0.65, rasterized=True)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#e2e8f0")
    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=7, mfc=LCOLOR[lg], mec="none",
               label=lg) for lg in LINEAGE_ORDER if (lin == lg).any()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=10.5, handletextpad=0.3, columnspacing=1.2,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("The frozen embedding separates cell lineages as cleanly as PCA",
                 fontsize=14, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.945, f"t-SNE of a {len(idx):,}-cell random subsample, colored by major lineage "
             f"(51 fine types grouped into 7)", fontsize=10.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig("celltype-tsne.png", dpi=140, facecolor="white", bbox_inches="tight")
    print("saved celltype-tsne.png")


if __name__ == "__main__":
    main()

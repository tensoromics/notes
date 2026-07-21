"""Part 3 figure: cell-type accuracy (51 classes), embedding vs PCA vs floors.

Numbers from celltype_probe.py (100k checkpoint, held-out patients):
  embed acc 0.799 (macro-F1 0.629) | PCA acc 0.823 (macro-F1 0.678).
Floors (majority-class + chance) are computed live from the labels here.
Rendering only -- run celltype_probe.py to regenerate/verify the two acc values.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMB_ACC, PCA_ACC = 0.799, 0.823          # from celltype_probe.py (100k, L=320)

meta = np.load("data/meta.npz", allow_pickle=True)
split, celltype = meta["split"], meta["celltype"]
tr, va = split == "train", split == "val"
classes = sorted(set(celltype))
n_ct = len(classes)
c_map = {c: i for i, c in enumerate(classes)}
yc = np.array([c_map[l] for l in celltype])
maj = np.argmax(np.bincount(yc[tr]))
maj_acc = float((yc[va] == maj).mean())
chance = 1.0 / n_ct
print(f"{n_ct} cell types | majority class '{classes[maj]}' val-acc floor = {maj_acc:.3f} "
      f"| uniform chance = {chance:.3f}")
print(f"embed {EMB_ACC}  PCA {PCA_ACC}")

names = [f"Uniform\nchance", f"Majority\nclass", "nano-Geneformer\nembedding", "PCA-50\nbaseline"]
vals = [chance, maj_acc, EMB_ACC, PCA_ACC]
COLORS = ["#cbd5e1", "#94a3b8", "#2563eb", "#ea580c"]
INK, MUTED, GRID = "#0f172a", "#64748b", "#eef2f7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

fig, ax = plt.subplots(figsize=(8.0, 5.0))
bars = ax.bar(range(4), vals, width=0.62, color=COLORS, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.014, f"{v:.3f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
ax.set_ylabel("cell-type accuracy  (held-out patients)", fontsize=10.5, color=MUTED)
ax.set_ylim(0, 0.95)
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=10.5, color=INK)
ax.set_title("Can a frozen cell embedding name the cell type?",
             fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=10)
ax.text(0, 0.965, f"{n_ct}-way classification over Smillie's fine cell types",
        transform=ax.transAxes, fontsize=10, color=MUTED)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#cbd5e1")
ax.tick_params(colors=MUTED)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("celltype-probe.png", dpi=150, facecolor="white", bbox_inches="tight")
print("saved celltype-probe.png")

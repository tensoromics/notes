"""Part 3 figure: disease-state macro-F1, PCA baseline vs frozen embedding.

Visualizes the numbers reported by `probe.py` (the reproducible compute script:
frozen nano-Geneformer cell embedding vs 50-dim PCA, logistic regression on
TRAIN patients, evaluated on HELD-OUT VAL patients). The majority-class floor is
recomputed live from the labels here. Rendering only -- run `probe.py` to
regenerate/verify the two probe numbers below.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# --- macro-F1 from probe.py output (post-QC 100k checkpoint; verified 2026-07) --
PCA_MACRO = 0.755          # PCA-50 baseline        (acc 0.768)
EMB_MACRO = 0.663          # nano-Geneformer embed  (acc 0.667)

# --- majority-class floor: computed live, no model needed ----------------------
meta = np.load("data/meta.npz", allow_pickle=True)
split, label = meta["split"], meta["label"]
classes = sorted(set(label))
y = np.array([classes.index(l) for l in label])
tr, va = split == "train", split == "val"
maj = np.argmax(np.bincount(y[tr]))
MAJ_MACRO = f1_score(y[va], np.full(va.sum(), maj), average="macro")
print(f"classes: {classes}")
print(f"majority class (train) = '{classes[maj]}'  ->  val macro-F1 floor = {MAJ_MACRO:.3f}")
print(f"PCA-50 {PCA_MACRO:.3f}   nano-Geneformer {EMB_MACRO:.3f}   gap {PCA_MACRO-EMB_MACRO:.3f}")

# --- figure: three macro-F1 bars, floor -> model -> PCA ------------------------
names = ["Majority-class\nbaseline", "nano-Geneformer\nembedding", "PCA-50\nbaseline"]
vals = [MAJ_MACRO, EMB_MACRO, PCA_MACRO]
COLORS = ["#94a3b8", "#2563eb", "#ea580c"]      # floor(grey), model(blue), PCA(orange)
INK, MUTED, GRID = "#0f172a", "#64748b", "#eef2f7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

fig, ax = plt.subplots(figsize=(7.4, 5.0))
bars = ax.bar(range(3), vals, width=0.6, color=COLORS, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.014, f"{v:.3f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
ax.set_ylabel("macro-F1  (held-out patients)", fontsize=10.5, color=MUTED)
ax.set_ylim(0, 0.85)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=10.5, color=INK)
ax.set_title("Can a frozen cell embedding read disease state?",
             fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=10)
ax.text(0, 0.965, "3-way classification: Healthy / Inflamed / Non-inflamed",
        transform=ax.transAxes, fontsize=10, color=MUTED)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#cbd5e1")
ax.tick_params(colors=MUTED)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("probe-disease.png", dpi=150, facecolor="white", bbox_inches="tight")
print("saved probe-disease.png")

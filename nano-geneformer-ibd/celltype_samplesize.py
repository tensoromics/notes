"""Part 3 figure: per-type F1 and AUC vs training-cell count (embed probe).
Shows naming a type (F1) needs examples — it rises with sample size — while
separating a type (AUC) stays near-perfect regardless of how rare it is.
From the cached 100k embedding; no re-embed.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

emb = np.load("data/emb_L320_mps.npy")
m = np.load("data/meta.npz", allow_pickle=True)
ct, split = m["celltype"], m["split"]
tr, va = split == "train", split == "val"
names = sorted(set(ct)); cmap = {c: i for i, c in enumerate(names)}
yc = np.array([cmap[c] for c in ct])

clf = LogisticRegression(max_iter=2000).fit(emb[tr], yc[tr])
pred = clf.predict(emb[va]); proba = clf.predict_proba(emb[va])
f1s = f1_score(yc[va], pred, average=None, labels=range(len(names)))
aucs = roc_auc_score(yc[va], proba, multi_class="ovr", average=None, labels=clf.classes_)
ntr = np.array([(yc[tr] == i).sum() for i in range(len(names))])
nva = np.array([(yc[va] == i).sum() for i in range(len(names))])
p = nva > 0

BLUE, ORANGE, INK, MUTED = "#2563eb", "#ea580c", "#0f172a", "#64748b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8.6, 5.4))
ax.scatter(ntr[p], aucs[p], s=42, c=ORANGE, alpha=0.85, zorder=3, label="AUC — can it separate the type?")
ax.scatter(ntr[p], f1s[p], s=42, c=BLUE, alpha=0.85, zorder=3, label="F1 — can it name the type?")
ax.set_xscale("log")
ax.set_xlabel("training cells of that type  (log scale)", fontsize=10.5, color=MUTED)
ax.set_ylabel("score, per cell type (held-out)", fontsize=10.5, color=MUTED)
ax.set_ylim(-0.03, 1.05)

# annotate the telling cases
def note(name, dx, dy, ha="left"):
    i = cmap[name]
    ax.annotate(name, (ntr[i], f1s[i]), (ntr[i]*dx, f1s[i]+dy), ha=ha, fontsize=8.5,
                color=INK, arrowprops=dict(arrowstyle="-", color="#cbd5e1", lw=0.8))
note("Glia", 1.15, 0.06)
note("CD4+ PD1+", 0.6, -0.16, "right")
note("RSPO3+", 1.1, -0.10)
note("Plasma", 0.55, 0.02, "right")

ax.legend(frameon=False, fontsize=10, loc="center right")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#cbd5e1")
ax.tick_params(colors=MUTED); ax.grid(axis="y", color="#eef2f7"); ax.set_axisbelow(True)
fig.subplots_adjust(top=0.85, left=0.09, right=0.97, bottom=0.12)
fig.suptitle("Naming a rare cell type needs examples; separating it doesn't",
             fontsize=13.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.965)
fig.text(0.02, 0.895, "each dot = one of 51 cell types  (Spearman ρ: F1 vs size 0.45, AUC vs size −0.21)",
         fontsize=9.5, color=MUTED, ha="left")
fig.savefig("celltype-samplesize.png", dpi=150, facecolor="white", bbox_inches="tight")
print("saved celltype-samplesize.png")
print(f"big(>=1000) mean F1 {f1s[ntr>=1000].mean():.2f} AUC {aucs[ntr>=1000].mean():.2f} | "
      f"rare(<200) mean F1 {f1s[(ntr<200)&p].mean():.2f} AUC {aucs[(ntr<200)&p].mean():.2f}")

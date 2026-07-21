"""Quick previews of additional cell-type evals for Part 3, from the cached
100k embedding (no re-embed, no MPS). Reports:
  1. coarse (7-lineage) vs fine (51-type) probe accuracy — does it nail broad
     identity but blur fine subtypes?
  2. per-type F1: which types it gets best / worst.
  3. kNN accuracy (non-parametric — a cell's neighbors share its type?).
  4. label-free clustering agreement (ARI / NMI) — find types with NO labels.
Embedding vs PCA throughout, held-out patients where a split applies.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score
from celltype_tsne import lineage

emb = np.load("data/emb_L320_mps.npy")
pca = np.load("data/pca.npy")
meta = np.load("data/meta.npz", allow_pickle=True)
ct, split = meta["celltype"], meta["split"]
tr, va = split == "train", split == "val"
cmap = {c: i for i, c in enumerate(sorted(set(ct)))}
yc = np.array([cmap[c] for c in ct])
lin = np.array([lineage(c) for c in ct])
lmap = {l: i for i, l in enumerate(sorted(set(lin)))}
yl = np.array([lmap[l] for l in lin])
names = sorted(set(ct))
print(f"cells {len(ct)} | 51 types, {len(set(lin))} lineages | train {tr.sum()} val {va.sum()}")


def probe(X, y):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[tr], y[tr]); p = clf.predict(X[va])
    return accuracy_score(y[va], p), f1_score(y[va], p, average="macro"), p


print("\n1) COARSE vs FINE probe accuracy (held-out patients)")
for name, X in [("embed", emb), ("PCA", pca)]:
    af, ff, pf = probe(X, yc)
    ac, fc, _ = probe(X, yl)
    print(f"  {name:6s}  fine(51) acc={af:.3f} macroF1={ff:.3f}  |  coarse(7-lineage) acc={ac:.3f}")

print("\n2) per-type F1 (embed, fine probe) — best & worst")
_, _, pf = probe(emb, yc)
f1s = f1_score(yc[va], pf, average=None, labels=range(len(names)))
order = np.argsort(f1s)
print("   best :", ", ".join(f"{names[i]} {f1s[i]:.2f}" for i in order[::-1][:6]))
print("   worst:", ", ".join(f"{names[i]} {f1s[i]:.2f}" for i in order[:6] if va[yc==i].sum() or True))

print("\n3) kNN accuracy (k=15, non-parametric; fit train, eval val)")
for name, X in [("embed", emb), ("PCA", pca)]:
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X[tr], yc[tr]); acc = accuracy_score(yc[va], knn.predict(X[va]))
    print(f"  {name:6s}  kNN acc={acc:.3f}")

print("\n4) label-free clustering agreement (MiniBatchKMeans k=51, ALL cells, vs true type)")
for name, X in [("embed", emb), ("PCA", pca)]:
    km = MiniBatchKMeans(n_clusters=51, random_state=0, n_init=3, batch_size=1024)
    lab = km.fit_predict(X)
    print(f"  {name:6s}  ARI={adjusted_rand_score(yc, lab):.3f}  NMI={normalized_mutual_info_score(yc, lab):.3f}")

print("\n5) macro AUC + worked example: CD4+ PD1+ (F1=0 but separable)")
from sklearn.metrics import roc_auc_score
import collections
clf = LogisticRegression(max_iter=2000).fit(emb[tr], yc[tr])
proba, pred, yv, cls = clf.predict_proba(emb[va]), clf.predict(emb[va]), yc[va], list(clf.classes_)
print(f"  embed macro OvR AUC (51 types) = "
      f"{roc_auc_score(yv, proba, multi_class='ovr', average='macro', labels=clf.classes_):.3f}")
T = cmap["CD4+ PD1+"]; mask = yv == T; colT = cls.index(T)
print(f"  CD4+ PD1+ held-out cells: {int(mask.sum())}")
print("  the probe names them:", dict(collections.Counter(names[p] for p in pred[mask])))
print(f"  mean P(CD4+ PD1+) on a TRUE cell  = {proba[mask, colT].mean():.3f}")
print(f"  mean P(CD4+ PD1+) on other cells  = {proba[~mask, colT].mean():.4f}")

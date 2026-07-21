"""Probe the frozen embedding for CELL TYPE (identity) vs DISEASE (magnitude).

Part 3's claim: rank encoding preserves identity but discards magnitude. If so,
the frozen embedding should read the 51 fine cell types well, even though it
loses to PCA on disease state. We embed the existing checkpoint with inputs
truncated to L=512 (covers 99.99% of cells; dodges the max_len=1024 MPS hang),
cache the embedding, then run both probes: embedding vs PCA, held-out patients.
"""
import os, signal, sys, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from model import NanoGeneformer

L = 320                         # covers 99.3% of cells fully; keeps MPS attention small
DEV = os.environ.get("NANO_DEVICE", "mps")
BS = 64                         # bs*heads*L^2 ~ 26M elements -> under the MPS hang threshold
CACHE = f"data/emb_L{L}_{DEV}.npy"


@torch.no_grad()
def embed_all(model, tokens, device, bs=BS):
    out = []
    for i in range(0, len(tokens), bs):
        signal.alarm(15)        # per-batch watchdog: catch an MPS hang without freezing
        b = torch.from_numpy(tokens[i:i + bs]).long().to(device)
        out.append(model.cell_embedding(b).cpu().numpy())
        signal.alarm(0)
    return np.concatenate(out)


def probe(name, X, y, tr, va, classes):
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[va])
    acc = accuracy_score(y[va], pred)
    mf1 = f1_score(y[va], pred, average="macro")
    wf1 = f1_score(y[va], pred, average="weighted")
    print(f"    {name:24s} acc={acc:.3f}  macro-F1={mf1:.3f}  weighted-F1={wf1:.3f}")
    return dict(acc=acc, macro_f1=mf1, weighted_f1=wf1)


def main():
    tokens = np.load("data/tokens.npy", mmap_mode="r")[:, :L].copy()   # truncate rank tail
    pca = np.load("data/pca.npy")
    meta = np.load("data/meta.npz", allow_pickle=True)
    split, disease, celltype = meta["split"], meta["label"], meta["celltype"]
    tr, va = split == "train", split == "val"

    if os.path.exists(CACHE):
        emb = np.load(CACHE)
        print(f"loaded cached embedding {CACHE}  {emb.shape}")
    else:
        ck = torch.load("checkpoints/model.pt", map_location=DEV)
        cfg = ck["cfg"]
        model = NanoGeneformer(ck["vocab_size"], cfg["d_model"], cfg["n_heads"],
                               cfg["n_layers"], ck["max_len"], cfg["dropout"], cfg["PAD"])
        model.load_state_dict(ck["model"]); model.to(DEV).eval()
        # per-batch watchdog: if MPS hangs, abort cleanly instead of freezing the session
        signal.signal(signal.SIGALRM, lambda *a: (print("EMBED HUNG — aborting"), sys.exit(124)))
        t = time.time()
        emb = embed_all(model, tokens, DEV)
        print(f"embedded {len(emb)} cells @L={L} on {DEV} in {time.time()-t:.0f}s")
        np.save(CACHE, emb)

    print(f"\nDISEASE (3 classes) — held-out patients:")
    dmap = {c: i for i, c in enumerate(sorted(set(disease)))}
    yd = np.array([dmap[l] for l in disease])
    probe("PCA-50", pca, yd, tr, va, None)
    probe("nano-Geneformer embed", emb, yd, tr, va, None)

    print(f"\nCELL TYPE ({len(set(celltype))} classes) — held-out patients:")
    cmap = {c: i for i, c in enumerate(sorted(set(celltype)))}
    yc = np.array([cmap[l] for l in celltype])
    probe("PCA-50", pca, yc, tr, va, None)
    probe("nano-Geneformer embed", emb, yc, tr, va, None)


if __name__ == "__main__":
    main()

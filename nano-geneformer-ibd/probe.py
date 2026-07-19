"""Linear probe: does the pretrained embedding beat PCA at reading disease state?

Trains logistic regression on TRAIN-patient cells, evaluates on HELD-OUT VAL
patients. Compares (a) frozen nano-Geneformer embeddings vs (b) 50-dim PCA.
The patient-held-out split is the whole point -- see README caveats.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from model import NanoGeneformer
from train import get_device


@torch.no_grad()
def embed_all(model, tokens, device, bs=256):
    out = []
    for i in range(0, len(tokens), bs):
        batch = torch.from_numpy(tokens[i:i + bs]).long().to(device)
        out.append(model.cell_embedding(batch).cpu().numpy())
    return np.concatenate(out)


def report(name, Xtr, ytr, Xva, yva):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    acc = accuracy_score(yva, pred)
    f1 = f1_score(yva, pred, average="macro")
    print(f"  {name:26s}  acc={acc:.3f}  macro-F1={f1:.3f}")
    return f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="checkpoints/model.pt")
    args = ap.parse_args()
    device = get_device()

    tokens = np.load(os.path.join(args.data, "tokens.npy"))
    pca = np.load(os.path.join(args.data, "pca.npy"))
    meta = np.load(os.path.join(args.data, "meta.npz"), allow_pickle=True)
    split, labels = meta["split"], meta["label"]

    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    model = NanoGeneformer(ck["vocab_size"], cfg["d_model"], cfg["n_heads"],
                           cfg["n_layers"], ck["max_len"], cfg["dropout"], cfg["PAD"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    emb = embed_all(model, tokens, device)

    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    tr, va = split == "train", split == "val"
    print(f"classes: {classes}")
    print(f"train cells: {tr.sum()}  val cells: {va.sum()} (held-out patients)\n")

    print("disease-state prediction on held-out patients:")
    report("PCA-50 baseline", pca[tr], y[tr], pca[va], y[va])
    report("nano-Geneformer embed", emb[tr], y[tr], emb[va], y[va])
    print("\nIf the embedding doesn't clearly beat PCA, that's a real result, not a"
          " failure -- rank encoding may be discarding the magnitude that inflammation"
          " lives in. See README.")


if __name__ == "__main__":
    main()

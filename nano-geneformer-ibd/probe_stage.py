"""Probe one trained stage: disease + cell type, embedding vs PCA, on held-out
patients. Also checks whether gene embeddings organize (COL1A1 neighbors) --
the weight-tying promise. Writes metrics.json. Safe MPS embedding (small batch
+ per-batch watchdog) so an MPS hang aborts cleanly instead of freezing.
"""
import argparse, json, os, signal, sys, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from model import NanoGeneformer


@torch.no_grad()
def embed_all(model, tokens, device, bs=64):
    out = []
    for i in range(0, len(tokens), bs):
        signal.alarm(20)
        b = torch.from_numpy(tokens[i:i + bs]).long().to(device)
        out.append(model.cell_embedding(b).cpu().numpy())
        signal.alarm(0)
    return np.concatenate(out)


def probe(X, y, tr, va):
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(X[tr], y[tr]); pred = clf.predict(X[va])
    return dict(acc=round(float(accuracy_score(y[va], pred)), 4),
                macro_f1=round(float(f1_score(y[va], pred, average="macro")), 4),
                weighted_f1=round(float(f1_score(y[va], pred, average="weighted")), 4))


def gene_neighbors(model, vocab, gene="COL1A1", k=8):
    g2i = vocab["gene_to_id"]
    if gene not in g2i:
        return None
    W = model.tok_emb.weight.detach().cpu().numpy()
    v = W[g2i[gene]]
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
    sims = Wn @ (v / (np.linalg.norm(v) + 1e-8))
    i2g = {int(k_): v_ for k_, v_ in vocab["id_to_gene"].items()}
    top = np.argsort(-sims)
    out = [(i2g.get(int(j), str(j)), round(float(sims[j]), 3))
           for j in top if int(j) != g2i[gene] and int(j) in i2g][:k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    signal.signal(signal.SIGALRM, lambda *x: (print("EMBED HUNG — abort"), sys.exit(124)))

    vocab = json.load(open(os.path.join(a.data, "vocab.json")))
    ck = torch.load(a.ckpt, map_location=a.device)
    ML = ck["max_len"]
    tokens = np.load(os.path.join(a.data, "tokens.npy"))[:, :ML].copy()
    pca = np.load(os.path.join(a.data, "pca.npy"))
    meta = np.load(os.path.join(a.data, "meta.npz"), allow_pickle=True)
    split, disease, celltype = meta["split"], meta["label"], meta["celltype"]
    tr, va = split == "train", split == "val"

    c = ck["cfg"]
    model = NanoGeneformer(ck["vocab_size"], c["d_model"], c["n_heads"],
                           c["n_layers"], ML, c["dropout"], c["PAD"])
    if c.get("tie"):
        model.mlm_head.weight = model.tok_emb.weight
    model.load_state_dict(ck["model"]); model.to(a.device).eval()

    # cache the embedding next to the checkpoint so re-probes are instant
    emb_cache = os.path.join(os.path.dirname(a.ckpt), f"emb_L{ML}.npy")
    if os.path.exists(emb_cache):
        emb = np.load(emb_cache)
        print(f"loaded cached embedding {emb_cache} {emb.shape}", flush=True)
    else:
        t = time.time()
        emb = embed_all(model, tokens, a.device)
        np.save(emb_cache, emb)
        print(f"embedded {len(emb)} cells @L={ML} in {time.time()-t:.0f}s "
              f"-> cached {emb_cache}", flush=True)

    dmap = {c: i for i, c in enumerate(sorted(set(disease)))}
    cmap = {c: i for i, c in enumerate(sorted(set(celltype)))}
    yd = np.array([dmap[l] for l in disease])
    yc = np.array([cmap[l] for l in celltype])
    metrics = {
        "ckpt": a.ckpt, "max_len": int(ML),
        "final_loss": float(np.mean(np.load(os.path.join(os.path.dirname(a.ckpt),
                          "loss_history.npy"))[-100:])),
        "disease": {"pca": probe(pca, yd, tr, va), "embed": probe(emb, yd, tr, va)},
        "celltype": {"n_classes": int(len(set(celltype))),
                     "pca": probe(pca, yc, tr, va), "embed": probe(emb, yc, tr, va)},
        "col1a1_neighbors": gene_neighbors(model, vocab),
    }
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Re-probe the overnight checkpoints (scale-365k, scale-365k-tied) cleanly,
no orchestrator timeout. Reports DISEASE first (the headline), then CELL TYPE,
then COL1A1 gene-neighbors (the weight-tying question). Uses cached embeddings
where present. Writes results/overnight/reprobe.json + prints a comparison vs the
published Part 3 numbers (100k: embed disease 0.663, PCA 0.755)."""
import json, os, signal, sys, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from model import NanoGeneformer

DATA = "data_365k"
DEV = "mps"
STAGES = [("scale-365k", False), ("scale-365k-tied", True)]
signal.signal(signal.SIGALRM, lambda *a: (print("EMBED HUNG — abort", flush=True), sys.exit(124)))


@torch.no_grad()
def embed_all(model, tokens, bs=64):
    out = []
    for i in range(0, len(tokens), bs):
        signal.alarm(20)
        b = torch.from_numpy(tokens[i:i+bs]).long().to(DEV)
        out.append(model.cell_embedding(b).cpu().numpy())
        signal.alarm(0)
    return np.concatenate(out)


def lr(X, y, tr, va, max_iter):
    t = time.time()
    clf = LogisticRegression(max_iter=max_iter, C=1.0, n_jobs=-1)
    clf.fit(X[tr], y[tr]); pred = clf.predict(X[va])
    return dict(acc=round(float(accuracy_score(y[va], pred)), 4),
                macro_f1=round(float(f1_score(y[va], pred, average="macro")), 4),
                weighted_f1=round(float(f1_score(y[va], pred, average="weighted")), 4),
                secs=round(time.time()-t, 1))


def col1a1_neighbors(model, vocab, k=8):
    g2i = vocab["gene_to_id"]
    if "COL1A1" not in g2i:
        return None
    W = model.tok_emb.weight.detach().cpu().numpy()
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
    sims = Wn @ Wn[g2i["COL1A1"]]
    i2g = {int(k_): v_ for k_, v_ in vocab["id_to_gene"].items()}
    order = np.argsort(-sims)
    return [(i2g[int(j)], round(float(sims[j]), 3)) for j in order
            if int(j) != g2i["COL1A1"] and int(j) in i2g][:k]


def main():
    vocab = json.load(open(f"{DATA}/vocab.json"))
    pca = np.load(f"{DATA}/pca.npy")
    meta = np.load(f"{DATA}/meta.npz", allow_pickle=True)
    split, disease, celltype = meta["split"], meta["label"], meta["celltype"]
    tr, va = split == "train", split == "val"
    # encode labels once (build the class->index map a single time, not per row)
    d_map = {c: i for i, c in enumerate(sorted(set(disease)))}
    c_map = {c: i for i, c in enumerate(sorted(set(celltype)))}
    yd = np.array([d_map[l] for l in disease])
    yc = np.array([c_map[l] for l in celltype])
    n_ct = len(c_map)
    print(f"cells: {tr.sum()} train / {va.sum()} val | disease 3-class | celltype {n_ct}-class", flush=True)

    # PCA baselines (same for both stages)
    print("\n=== PCA-50 baseline ===", flush=True)
    pca_dis = lr(pca, yd, tr, va, 2000); print(f"  disease : {pca_dis}", flush=True)
    pca_ct = lr(pca, yc, tr, va, 1000);  print(f"  celltype: {pca_ct}", flush=True)

    results = {"data": DATA, "n_celltypes": n_ct,
               "published_part3_100k": {"disease_embed": 0.663, "disease_pca": 0.755},
               "pca_365k": {"disease": pca_dis, "celltype": pca_ct}, "stages": {}}

    for name, tied in STAGES:
        sdir = f"results/overnight/{name}"
        ck = torch.load(f"{sdir}/model.pt", map_location=DEV)
        ML = ck["max_len"]; c = ck["cfg"]
        model = NanoGeneformer(ck["vocab_size"], c["d_model"], c["n_heads"],
                               c["n_layers"], ML, c["dropout"], c["PAD"])
        if c.get("tie"):
            model.mlm_head.weight = model.tok_emb.weight
        model.load_state_dict(ck["model"]); model.to(DEV).eval()

        cache = f"{sdir}/emb_L{ML}.npy"
        if os.path.exists(cache):
            emb = np.load(cache); print(f"\n[{name}] loaded cached embedding {emb.shape}", flush=True)
        else:
            tokens = np.load(f"{DATA}/tokens.npy")[:, :ML].copy()
            t = time.time(); emb = embed_all(model, tokens)
            np.save(cache, emb)
            print(f"\n[{name}] embedded {len(emb)} @L={ML} in {time.time()-t:.0f}s -> cached", flush=True)

        loss = float(np.mean(np.load(f"{sdir}/loss_history.npy")[-100:]))
        dis = lr(emb, yd, tr, va, 2000)
        print(f"[{name}] loss={loss:.3f}  DISEASE embed={dis['macro_f1']:.3f} "
              f"(PCA {pca_dis['macro_f1']:.3f}, Part3-100k embed 0.663)", flush=True)
        ct = lr(emb, yc, tr, va, 1000)
        print(f"[{name}] CELLTYPE embed acc={ct['acc']:.3f} macroF1={ct['macro_f1']:.3f} "
              f"(PCA acc {pca_ct['acc']:.3f})", flush=True)
        nb = col1a1_neighbors(model, vocab)
        print(f"[{name}] COL1A1 neighbors: {nb[:6] if nb else None}", flush=True)

        results["stages"][name] = {"tied": tied, "loss": round(loss, 3),
                                   "disease": dis, "celltype": ct, "col1a1_neighbors": nb}
        json.dump(results, open("results/overnight/reprobe.json", "w"), indent=2)

    print("\nwrote results/overnight/reprobe.json", flush=True)


if __name__ == "__main__":
    main()

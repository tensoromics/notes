"""In-silico gene deletion: which genes, when removed from inflamed cells, push
their embedding toward the healthy state?

For each candidate gene we delete its token from inflamed cells that express it,
recompute the cell embedding, and measure the shift along the (healthy - inflamed)
axis. Genes with the largest healthward shift are nominated as drivers of the
inflamed state. Watch for TNF / OSM / IL1B (literature-validated in IBD).
"""
import argparse
import json
import os

import numpy as np
import torch

from model import NanoGeneformer
from train import get_device

# healthy vs inflamed label strings vary by dataset -- edit to match your labels.
HEALTHY = {"Healthy", "healthy", "Control", "control", "HC"}
INFLAMED = {"Inflamed", "inflamed"}
WATCHLIST = ["TNF", "OSM", "OSMR", "IL1B", "IL6", "IFNG", "S100A8", "S100A9",
             "CXCL8", "IL11", "IL13RA2", "CHI3L1"]


@torch.no_grad()
def embed(model, tok, device):
    return model.cell_embedding(torch.from_numpy(tok).long().to(device)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="checkpoints/model.pt")
    ap.add_argument("--n-cells", type=int, default=300, help="inflamed cells sampled")
    ap.add_argument("--n-genes", type=int, default=250, help="candidate genes tested")
    args = ap.parse_args()
    device = get_device()

    tokens = np.load(os.path.join(args.data, "tokens.npy"))
    meta = np.load(os.path.join(args.data, "meta.npz"), allow_pickle=True)
    labels = meta["label"]
    vocab = json.load(open(os.path.join(args.data, "vocab.json")))
    id_to_gene = {int(k): v for k, v in vocab["id_to_gene"].items()}

    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    model = NanoGeneformer(ck["vocab_size"], cfg["d_model"], cfg["n_heads"],
                           cfg["n_layers"], ck["max_len"], cfg["dropout"], cfg["PAD"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    is_healthy = np.array([l in HEALTHY for l in labels])
    is_inflamed = np.array([l in INFLAMED for l in labels])
    if is_healthy.sum() == 0 or is_inflamed.sum() == 0:
        raise SystemExit(f"Edit HEALTHY/INFLAMED sets. Labels seen: {sorted(set(labels))}")

    # healthward axis in embedding space
    h_cent = embed(model, tokens[is_healthy], device).mean(0)
    i_cent = embed(model, tokens[is_inflamed], device).mean(0)
    axis = h_cent - i_cent
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    # sample inflamed cells; deterministic (first N) to avoid RNG
    inf_idx = np.where(is_inflamed)[0][: args.n_cells]
    cells = tokens[inf_idx]
    base_emb = embed(model, cells, device)
    base_proj = base_emb @ axis

    # candidate genes = most frequently expressed among these cells + the watchlist
    present = cells[cells >= cfg["N_SPECIAL"]]
    freq = np.bincount(present, minlength=ck["vocab_size"])
    candidates = list(np.argsort(-freq)[: args.n_genes])
    for g in WATCHLIST:
        gid = vocab["gene_to_id"].get(g)
        if gid is not None and gid not in candidates:
            candidates.append(gid)

    results = []
    for gid in candidates:
        has = (cells == gid).any(1)
        if has.sum() < 5:
            continue
        perturbed = cells[has].copy()
        perturbed[perturbed == gid] = cfg["PAD"]        # delete the gene token
        shift = (embed(model, perturbed, device) @ axis) - base_proj[has]
        results.append((id_to_gene.get(gid, str(gid)), float(shift.mean()), int(has.sum())))

    results.sort(key=lambda r: -r[1])
    print(f"\nHealthward shift after deleting each gene "
          f"({len(inf_idx)} inflamed cells; +ve = toward healthy):\n")
    print(f"  {'gene':12s} {'shift':>8s}  {'n_cells':>7s}")
    for gene, shift, n in results[:25]:
        star = "  <-- watchlist" if gene in WATCHLIST else ""
        print(f"  {gene:12s} {shift:8.4f}  {n:7d}{star}")


if __name__ == "__main__":
    main()

"""Show what the trained nano-Geneformer predicts for masked genes.

Loads the checkpoint, takes held-out (val) cells, replaces a gene token with
[MASK], and reports the model's top-k guesses at that position. Also computes an
honest aggregate top-1 / top-5 recovery rate so the showcased examples are not
cherry-picked in a vacuum.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F

from config import CONFIG as cfg
from model import NanoGeneformer

torch.manual_seed(0)
np.random.seed(0)
device = "cpu"

# ---- load vocab, data, model ------------------------------------------------
vocab = json.load(open("data/vocab.json"))
id_to_gene = {int(k): v for k, v in vocab["id_to_gene"].items()}
def gene(tid):
    return id_to_gene.get(int(tid), f"<{int(tid)}>")

tokens = np.load("data/tokens.npy")
meta = np.load("data/meta.npz", allow_pickle=True)
split, celltype, label = meta["split"], meta["celltype"], meta["label"]

val_idx = np.where(split == "val")[0]
print(f"{len(val_idx)} held-out val cells\n")

model = NanoGeneformer(
    vocab_size=vocab["vocab_size"], d_model=cfg.d_model, n_heads=cfg.n_heads,
    n_layers=cfg.n_layers, max_len=cfg.max_len, dropout=cfg.dropout, pad_id=cfg.PAD,
)
state = torch.load("checkpoints/model.pt", map_location=device)
model.load_state_dict(state["model"] if "model" in state else state)
model.eval()

PAD, MASK = cfg.PAD, cfg.MASK


@torch.no_grad()
def topk_at(tok_row, pos, k=5):
    """Mask position `pos` in one cell, return (top_ids, top_probs)."""
    x = torch.tensor(tok_row, dtype=torch.long).unsqueeze(0)
    x[0, pos] = MASK
    logits = model(x)[0, pos]           # (vocab,)
    probs = F.softmax(logits, dim=-1)
    p, ids = probs.topk(k)
    return ids.tolist(), p.tolist()


# ---- aggregate top-1 / top-5 recovery on masked positions -------------------
# For each of N val cells, mask ~15% of expressed genes with [MASK], measure how
# often the true gene is the top-1 / within top-5 guess. This is the honest number.
rng = np.random.default_rng(0)
N_EVAL = 400
n_masked = top1 = top5 = 0
from collections import defaultdict
per_gene = defaultdict(lambda: [0, 0])   # gene_id -> [correct_top1, times_masked]
for i in val_idx[:N_EVAL]:
    row = tokens[i]
    expressed = np.where(row >= cfg.N_SPECIAL)[0]
    if expressed.size < 4:
        continue
    chosen = rng.choice(expressed, size=max(1, int(0.15 * expressed.size)), replace=False)
    x = torch.tensor(row, dtype=torch.long).unsqueeze(0)
    truth = x[0, chosen].clone()
    x[0, chosen] = MASK
    with torch.no_grad():
        logits = model(x)[0, chosen]     # (n_chosen, vocab)
    top5_ids = logits.topk(5, dim=-1).indices
    hit1 = (top5_ids[:, 0] == truth)
    top1 += hit1.sum().item()
    top5 += (top5_ids == truth.unsqueeze(1)).any(dim=1).sum().item()
    n_masked += chosen.size
    for g, ok in zip(truth.tolist(), hit1.tolist()):
        per_gene[g][0] += int(ok)
        per_gene[g][1] += 1

print("=== Aggregate recovery on held-out cells (mask -> predict) ===")
print(f"masked positions : {n_masked}")
print(f"top-1 accuracy   : {top1/n_masked:.3f}   (chance = 1/2048 = {1/2048:.5f})")
print(f"top-5 accuracy   : {top5/n_masked:.3f}")
print()

# ---- easy wins: which genes does the model recover most reliably? -----------
# Genes masked >=8 times, ranked by top-1 recovery rate. These are the model's
# "easy wins" -- ubiquitous, tightly co-expressed genes it can almost always fill in.
print("=== Easy wins: most reliably recovered genes (masked >= 8x) ===")
rows = [(gene(g), c / n, c, n) for g, (c, n) in per_gene.items() if n >= 8]
rows.sort(key=lambda r: (-r[1], -r[3]))
print(f"{'gene':<10}{'top-1 recovery':>16}{'masked':>10}")
for name, rate, c, n in rows[:12]:
    print(f"{name:<10}{rate:>15.0%}{n:>10}")
print()

# ---- showcase: single-gene masking on recognizable markers ------------------
# Scan val cells, mask each of the top-ranked genes one at a time, keep cases
# where the model's #1 guess equals the true gene AND the gene is recognizable.
print("=== Showcase: correctly recovered masked genes (top of ranking) ===")
shown = 0
for i in val_idx:
    if shown >= 12:
        break
    row = tokens[i]
    expressed = np.where(row >= cfg.N_SPECIAL)[0]
    if expressed.size < 8:
        continue
    for pos in range(min(6, expressed.size)):        # only the top-ranked genes
        true_id = int(row[pos])
        ids, ps = topk_at(row, pos, k=5)
        if ids[0] == true_id:                        # top-1 correct
            context = " > ".join(gene(t) for t in row[:pos]) or "(start)"
            runners = ", ".join(f"{gene(t)} {p:.0%}" for t, p in zip(ids[1:4], ps[1:4]))
            print(f"[{celltype[i]} / {label[i]}]  rank-{pos+1} gene masked")
            print(f"   context: {context} > [MASK]")
            print(f"   truth = {gene(true_id)}  |  model #1 = {gene(ids[0])} {ps[0]:.0%}"
                  f"  (then {runners})")
            print()
            shown += 1
            break

print("done.")

"""Extract REAL self-attention weights from the trained nano-Geneformer.

Find a fibroblast that expresses COL1A1 and other collagens, blank COL1A1 with
[MASK], and read what that masked position attends to -- per layer, per head.
Prints the top-attended genes so we can see whether the attention actually
concentrates on the biologically related genes (collagens / stromal), or is mush.
"""
import json
import numpy as np
import torch
import torch.nn as nn

from config import CONFIG as cfg
from model import NanoGeneformer

torch.manual_seed(0)
device = "cpu"

vocab = json.load(open("data/vocab.json"))
g2i = vocab["gene_to_id"]
i2g = {int(k): v for k, v in vocab["id_to_gene"].items()}
def gene(tid):
    return i2g.get(int(tid), f"<{int(tid)}>")

tokens = np.load("data/tokens.npy")
meta = np.load("data/meta.npz", allow_pickle=True)
split, celltype = meta["split"], meta["celltype"]

model = NanoGeneformer(
    vocab_size=vocab["vocab_size"], d_model=cfg.d_model, n_heads=cfg.n_heads,
    n_layers=cfg.n_layers, max_len=cfg.max_len, dropout=cfg.dropout, pad_id=cfg.PAD,
)
state = torch.load("checkpoints/model.pt", map_location=device)
model.load_state_dict(state["model"] if "model" in state else state)
model.eval()

COL1A1 = g2i["COL1A1"] + 0   # already a token id in vocab (>=2)
collagens = {g2i[g] for g in g2i if g.startswith("COL")}

# ---- find a val fibroblast that expresses COL1A1 and several collagens -------
val_idx = np.where(split == "val")[0]
best = None
for i in val_idx:
    ct = str(celltype[i])
    if "ibro" not in ct:                       # Inflammatory Fibroblasts / Myofibroblasts
        continue
    row = tokens[i]
    if COL1A1 not in row:
        continue
    ncol = sum(int(t) in collagens for t in row)
    pos = int(np.where(row == COL1A1)[0][0])
    if pos < 60 and ncol >= 4:                 # COL1A1 fairly high, several collagens present
        best = (i, pos, ncol, ct)
        break

if best is None:
    raise SystemExit("no suitable fibroblast found")
i, mask_pos, ncol, ct = best
row = tokens[i].copy()
expressed = np.where(row >= cfg.N_SPECIAL)[0]
print(f"cell {i}  [{ct}]  expresses {expressed.size} genes, {ncol} collagens")
print(f"COL1A1 sits at rank {mask_pos+1}; blanking it.\n")
top_ranked = " > ".join(gene(row[p]) for p in range(min(8, expressed.size)))
print("top of ranking:", top_ranked, "...\n")

# ---- run once with attention captured (force the Python path, need_weights) --
x = torch.tensor(row, dtype=torch.long).unsqueeze(0)
x[0, mask_pos] = cfg.MASK
pad_mask = x.eq(cfg.PAD)

captured = []                                   # per-layer attn: (heads, L, L)
orig_fwd = nn.MultiheadAttention.forward
def patched(self, *a, **kw):
    kw["need_weights"] = True
    kw["average_attn_weights"] = False
    out, w = orig_fwd(self, *a, **kw)
    captured.append(w.detach()[0])              # drop batch dim -> (heads, L, L)
    return out, w
# a no-op forward hook on each layer disables the fused fast-path so patched() runs
handles = [layer.register_forward_hook(lambda m, i, o: None) for layer in model.encoder.layers]
nn.MultiheadAttention.forward = patched
with torch.no_grad():
    model(x)
nn.MultiheadAttention.forward = orig_fwd
for h in handles:
    h.remove()

# ---- report what the masked position attends to, per layer ------------------
L = expressed.size
def show(weights_row, tag):
    order = np.argsort(-weights_row)
    parts = []
    for p in order[:6]:
        parts.append(f"{gene(row[p])}{'*' if int(row[p]) in collagens else ''} {weights_row[p]:.2f}")
    print(f"  {tag}: " + ", ".join(parts))

for li, attn in enumerate(captured):           # attn: (heads, L, L)
    print(f"layer {li}:")
    mean_row = attn.mean(0)[mask_pos].numpy()[:L]   # head-averaged attention FROM the blank
    show(mean_row, "head-avg")
    for hh in range(attn.shape[0]):
        show(attn[hh, mask_pos].numpy()[:L], f"head {hh}")
    print()

print("(* = a collagen gene)")

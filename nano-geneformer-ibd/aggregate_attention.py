"""Is head specialization robust, or a one-cell fluke?

Across many held-out fibroblasts, blank COL1A1 and measure, for every
(layer, head), the fraction of the masked slot's attention that lands on
ECM / collagen genes. Average over cells. If one head is consistently high,
the "ECM head" claim generalizes; if it's noisy, we should soften the figure.
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
def gene(tid): return i2g.get(int(tid), "")

tokens = np.load("data/tokens.npy")
meta = np.load("data/meta.npz", allow_pickle=True)
split, celltype = meta["split"], meta["celltype"]

model = NanoGeneformer(vocab_size=vocab["vocab_size"], d_model=cfg.d_model,
    n_heads=cfg.n_heads, n_layers=cfg.n_layers, max_len=cfg.max_len,
    dropout=cfg.dropout, pad_id=cfg.PAD)
state = torch.load("checkpoints/model.pt", map_location=device)
model.load_state_dict(state["model"] if "model" in state else state)
model.eval()

COL1A1 = g2i["COL1A1"]
collagens = {g2i[g] for g in g2i if g.startswith("COL")}
ECM_NAMES = {"SPARC","LUM","DCN","LTBP4","LTBP1","MMP1","MMP2","MMP3","TIMP1",
    "TIMP2","PLAT","PLAU","CHI3L1","F3","AEBP1","MFAP4","FN1","VCAN","FBLN1",
    "FBN1","C1S","CALD1"}
ecm_ids = collagens | {g2i[g] for g in ECM_NAMES if g in g2i}

def capture_attn(row, mask_pos):
    x = torch.tensor(row, dtype=torch.long).unsqueeze(0)
    x[0, mask_pos] = cfg.MASK
    cap = []
    orig = nn.MultiheadAttention.forward
    def patched(self, *a, **kw):
        kw["need_weights"] = True; kw["average_attn_weights"] = False
        out, w = orig(self, *a, **kw); cap.append(w.detach()[0]); return out, w
    hs = [l.register_forward_hook(lambda m,i,o: None) for l in model.encoder.layers]
    nn.MultiheadAttention.forward = patched
    with torch.no_grad():
        model(x)
    nn.MultiheadAttention.forward = orig
    for h in hs: h.remove()
    return cap    # list per layer: (heads, L, L)

# ---- collect fibroblasts ----------------------------------------------------
val_idx = np.where(split == "val")[0]
n_layers, n_heads = cfg.n_layers, cfg.n_heads
ecm_frac = np.zeros((n_layers, n_heads))   # summed over cells
counts = 0
per_cell_head0_L3 = []
N_MAX = 80
for i in val_idx:
    if counts >= N_MAX: break
    if "ibro" not in str(celltype[i]): continue
    row = tokens[i]
    if COL1A1 not in row: continue
    L = int(np.where(row >= cfg.N_SPECIAL)[0][-1]) + 1
    if sum(int(t) in collagens for t in row) < 4: continue
    mask_pos = int(np.where(row == COL1A1)[0][0])
    if mask_pos >= 60: continue
    cap = capture_attn(row, mask_pos)
    ecm_mask = np.array([int(row[p]) in ecm_ids for p in range(L)])
    for li in range(n_layers):
        attn = cap[li]                       # (heads, L, L)
        for hh in range(n_heads):
            w = attn[hh, mask_pos].numpy()[:L]
            ecm_frac[li, hh] += w[ecm_mask].sum()
    per_cell_head0_L3.append(cap[2][0, mask_pos].numpy()[:L][ecm_mask].sum())
    counts += 1

ecm_frac /= counts
print(f"{counts} held-out fibroblasts, COL1A1 blanked\n")
print("Mean fraction of the masked slot's attention landing on ECM/collagen genes:")
print("           " + "".join(f"head{h:>7}" for h in range(n_heads)))
for li in range(n_layers):
    print(f"  layer {li+1}:  " + "".join(f"{ecm_frac[li,h]:>11.2f}" for h in range(n_heads)))
best = np.unravel_index(np.argmax(ecm_frac), ecm_frac.shape)
print(f"\nStrongest ECM head: layer {best[0]+1}, head {best[1]}  ({ecm_frac[best]:.2f} mean ECM attention)")
arr = np.array(per_cell_head0_L3)
print(f"layer 3 / head 0 ECM attention across cells: mean {arr.mean():.2f}, "
      f"std {arr.std():.2f}, min {arr.min():.2f}, max {arr.max():.2f}")
print(f"  fraction of cells where L3/head0 ECM attention > 0.3: {(arr>0.3).mean():.0%}")

"""Render the 'attention heads specialize' figure — robust aggregate version.

For many held-out fibroblasts, blank COL1A1 and measure how much of the masked
slot's attention lands on the ECM / collagen neighborhood, per (layer, head).
Average over cells. Plots a layer x head grid so the ECM-specialized heads stand
out from the MT-'sink' heads — no cherry-picked single cell.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CONFIG as cfg
from model import NanoGeneformer

torch.manual_seed(0)
device = "cpu"
N_CELLS = 80

vocab = json.load(open("data/vocab.json"))
g2i = vocab["gene_to_id"]
i2g = {int(k): v for k, v in vocab["id_to_gene"].items()}
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
ECM = {"SPARC", "LUM", "DCN", "LTBP4", "LTBP1", "MMP1", "MMP2", "MMP3", "TIMP1",
       "TIMP2", "PLAT", "PLAU", "CHI3L1", "F3", "AEBP1", "MFAP4", "FN1", "VCAN",
       "FBLN1", "FBN1", "C1S", "CALD1"}
ecm_ids = collagens | {g2i[g] for g in ECM if g in g2i}


def capture(row, mask_pos):
    x = torch.tensor(row, dtype=torch.long).unsqueeze(0)
    x[0, mask_pos] = cfg.MASK
    cap = []
    orig = nn.MultiheadAttention.forward
    def patched(self, *a, **kw):
        kw["need_weights"] = True; kw["average_attn_weights"] = False
        out, w = orig(self, *a, **kw); cap.append(w.detach()[0]); return out, w
    hs = [l.register_forward_hook(lambda m, i, o: None) for l in model.encoder.layers]
    nn.MultiheadAttention.forward = patched
    with torch.no_grad():
        model(x)
    nn.MultiheadAttention.forward = orig
    for h in hs:
        h.remove()
    return cap

val = np.where(split == "val")[0]
nL, nH = cfg.n_layers, cfg.n_heads
acc = np.zeros((nL, nH))
count = 0
for i in val:
    if count >= N_CELLS:
        break
    if "ibro" not in str(celltype[i]):
        continue
    row = tokens[i]
    if COL1A1 not in row or sum(int(t) in collagens for t in row) < 4:
        continue
    L = int(np.where(row >= cfg.N_SPECIAL)[0][-1]) + 1
    pos = int(np.where(row == COL1A1)[0][0])
    if pos >= 60:
        continue
    cap = capture(row, pos)
    ecm_mask = np.array([int(row[p]) in ecm_ids for p in range(L)])
    for li in range(nL):
        for hh in range(nH):
            acc[li, hh] += cap[li][hh, pos].numpy()[:L][ecm_mask].sum()
    count += 1
acc /= count
print(f"{count} fibroblasts; mean ECM attention per (layer,head):\n{np.round(acc,2)}")

# ---- plot: layer x head grid of mean ECM attention ---------------------------
ACC, OTHER, INK, MUTED = "#2563eb", "#cbd5e1", "#0f172a", "#64748b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.0), sharex=True, sharey=True)
fig.suptitle("Attention heads specialize: which track COL1A1's collagen neighborhood?",
             fontsize=13.5, fontweight="bold", color=INK, y=0.98)
fig.text(0.5, 0.925, f"mean share of the blank's attention on ECM / collagen genes, "
         f"over {count} held-out fibroblasts", ha="center", fontsize=10, color=MUTED)
for li, ax in enumerate(axes.flat):
    vals = acc[li]
    colors = [ACC if v >= 0.30 else OTHER for v in vals]
    ax.bar(range(nH), vals, color=colors, width=0.66, zorder=3)
    for h, v in enumerate(vals):
        ax.text(h, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5,
                color=(INK if v >= 0.30 else MUTED))
    ax.set_title(f"layer {li+1}", fontsize=11, fontweight="bold", color=INK, loc="left", pad=3)
    ax.set_xticks(range(nH))
    ax.set_xticklabels([f"head {h}" for h in range(nH)], fontsize=9.5, color=MUTED)
    ax.set_ylim(0, 0.66)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#eef2f7")
    ax.set_axisbelow(True)
fig.text(0.5, 0.02, "blue = a collagen/ECM-tracking head (>=0.30);  grey = attends elsewhere (mostly MT 'sink' genes)",
         ha="center", fontsize=9.5, color=MUTED)
fig.tight_layout(rect=[0, 0.045, 1, 0.9])
fig.savefig("attention-heads.png", dpi=150, facecolor="white", bbox_inches="tight")
print("saved attention-heads.png")

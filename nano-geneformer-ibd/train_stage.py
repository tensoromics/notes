"""One overnight training stage: pretrain nano-Geneformer, optional weight tying.

Standalone (does not touch the published train.py). Uses a small max_len / batch
so the MPS attention stays under the hang threshold. Saves a checkpoint + loss
history + config after every epoch, so a killed run still leaves partial results.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import NanoGeneformer


def get_device(pref):
    if pref and pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def mask_tokens(tokens, vocab_size, mlm_prob, MASK, N_SPECIAL, device):
    labels = tokens.clone()
    prob = torch.full(tokens.shape, mlm_prob, device=device)
    prob[tokens < N_SPECIAL] = 0.0
    chosen = torch.bernoulli(prob).bool()
    labels[~chosen] = -100
    inp = tokens.clone()
    to_mask = torch.bernoulli(torch.full(tokens.shape, 0.8, device=device)).bool() & chosen
    inp[to_mask] = MASK
    to_rand = (torch.bernoulli(torch.full(tokens.shape, 0.5, device=device)).bool()
               & chosen & ~to_mask)
    rand = torch.randint(N_SPECIAL, vocab_size, tokens.shape, device=device)
    inp[to_rand] = rand[to_rand]
    return inp, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tie", action="store_true", help="tie mlm_head.weight to tok_emb.weight")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = get_device(a.device)
    N_SPECIAL, MASK, MLM = 2, 1, 0.15

    vocab = json.load(open(os.path.join(a.data, "vocab.json")))
    vocab_size = vocab["vocab_size"]
    ML = min(a.max_len, vocab["max_len"])
    tokens = np.load(os.path.join(a.data, "tokens.npy"))[:, :ML]
    split = np.load(os.path.join(a.data, "meta.npz"), allow_pickle=True)["split"]
    tokens = tokens[split == "train"]
    print(f"device={dev}  tie={a.tie}  max_len={ML}  train_cells={len(tokens)}  vocab={vocab_size}", flush=True)

    ds = TensorDataset(torch.from_numpy(tokens).long())
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True)

    model = NanoGeneformer(vocab_size, 256, 4, 4, ML, 0.1, 0).to(dev)
    # BERT-style small embedding init (std 0.02): required for stable weight
    # tying (default nn.Embedding std~1 blows up tied output logits), applied to
    # BOTH variants so tie-vs-no-tie is a clean ablation.
    torch.nn.init.normal_(model.tok_emb.weight, std=0.02)
    torch.nn.init.normal_(model.pos_emb.weight, std=0.02)
    with torch.no_grad():
        model.tok_emb.weight[0].zero_()                    # keep PAD row zero
    if a.tie:
        model.mlm_head.weight = model.tok_emb.weight       # weight tying
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(dl) * a.epochs
    warm = int(0.05 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    cfg = dict(d_model=256, n_heads=4, n_layers=4, dropout=0.1, PAD=0, MASK=1,
               N_SPECIAL=2, max_len=ML, tie=a.tie, epochs=a.epochs,
               batch_size=a.batch_size, lr=a.lr, vocab_size=vocab_size)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    loss_log = []
    t0 = time.time()
    model.train()
    for ep in range(a.epochs):
        run = 0.0
        for step, (batch,) in enumerate(dl):
            batch = batch.to(dev)
            inp, labels = mask_tokens(batch, vocab_size, MLM, MASK, N_SPECIAL, dev)
            loss = F.cross_entropy(model(inp).view(-1, vocab_size), labels.view(-1),
                                   ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            lv = float(loss.item()); loss_log.append(lv); run += lv
            if (step + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  ep{ep+1} step{step+1}/{len(dl)} loss={lv:.3f} "
                      f"avg={run/(step+1):.3f} lr={sched.get_last_lr()[0]:.1e} "
                      f"elapsed={el/60:.1f}m", flush=True)
        # per-epoch checkpoint (overwrite) + loss so far
        torch.save({"model": model.state_dict(), "cfg": cfg,
                    "vocab_size": vocab_size, "max_len": ML},
                   os.path.join(a.out, "model.pt"))
        np.save(os.path.join(a.out, "loss_history.npy"),
                np.asarray(loss_log, dtype=np.float32))
        print(f"epoch {ep+1}/{a.epochs} done  avg_loss={run/len(dl):.3f}  "
              f"saved checkpoint  ({(time.time()-t0)/60:.1f}m elapsed)", flush=True)

    print(f"DONE {a.out}  final_loss(mean last 100)={np.mean(loss_log[-100:]):.3f}  "
          f"total={(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()

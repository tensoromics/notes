"""Pretrain nano-Geneformer with masked gene prediction (self-supervised)."""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import CONFIG
from model import NanoGeneformer


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def mask_tokens(tokens, vocab_size, cfg):
    """BERT masking: 15% chosen; of those 80% -> [MASK], 10% -> random gene, 10% keep."""
    labels = tokens.clone()
    prob = torch.full(tokens.shape, cfg.mlm_prob, device=tokens.device)
    prob[tokens < cfg.N_SPECIAL] = 0.0                    # never mask PAD/MASK
    chosen = torch.bernoulli(prob).bool()
    labels[~chosen] = -100                                # loss only on chosen

    inp = tokens.clone()
    to_mask = torch.bernoulli(torch.full(tokens.shape, 0.8, device=tokens.device)).bool() & chosen
    inp[to_mask] = cfg.MASK
    to_rand = (torch.bernoulli(torch.full(tokens.shape, 0.5, device=tokens.device)).bool()
               & chosen & ~to_mask)
    rand = torch.randint(cfg.N_SPECIAL, vocab_size, tokens.shape, device=tokens.device)
    inp[to_rand] = rand[to_rand]
    return inp, labels


def main():
    cfg = CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cfg.out_dir)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    args = ap.parse_args()

    torch.manual_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    vocab = json.load(open(os.path.join(args.data, "vocab.json")))
    vocab_size, max_len = vocab["vocab_size"], vocab["max_len"]
    tokens = np.load(os.path.join(args.data, "tokens.npy"))
    split = np.load(os.path.join(args.data, "meta.npz"), allow_pickle=True)["split"]

    if cfg.pretrain_on == "train":
        tokens = tokens[split == "train"]
    print(f"pretraining on {len(tokens)} cells ({cfg.pretrain_on})")

    ds = TensorDataset(torch.from_numpy(tokens).long())
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = NanoGeneformer(vocab_size, cfg.d_model, cfg.n_heads, cfg.n_layers,
                           max_len, cfg.dropout, cfg.PAD).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    total_steps = len(dl) * args.epochs
    warmup = int(cfg.warmup_frac * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    loss_log = []
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        for (batch,) in pbar:
            batch = batch.to(device)
            inp, labels = mask_tokens(batch, vocab_size, cfg)
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1),
                                   ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_log.append(float(loss.item()))
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        ckpt = os.path.join(args.out, "model.pt")
        torch.save({"model": model.state_dict(),
                    "cfg": cfg.to_dict(), "vocab_size": vocab_size,
                    "max_len": max_len}, ckpt)
        print(f"saved {ckpt}")

    np.save(os.path.join(args.out, "loss_history.npy"), np.asarray(loss_log, dtype=np.float32))
    print(f"saved loss history: {len(loss_log)} steps")


if __name__ == "__main__":
    main()

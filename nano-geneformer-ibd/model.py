"""A tiny BERT-style encoder over rank-encoded gene tokens (nano-Geneformer)."""
import torch
import torch.nn as nn


class NanoGeneformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 max_len=1024, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)  # rank position
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.register_buffer("pos_ids", torch.arange(max_len).unsqueeze(0),
                             persistent=False)

    def encode(self, tokens):
        """tokens: (B, L) -> hidden states (B, L, d_model)."""
        L = tokens.size(1)
        pad_mask = tokens.eq(self.pad_id)                 # (B, L) True where pad
        x = self.tok_emb(tokens) + self.pos_emb(self.pos_ids[:, :L])
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.norm(h)

    def forward(self, tokens):
        return self.mlm_head(self.encode(tokens))         # (B, L, vocab)

    @torch.no_grad()
    def cell_embedding(self, tokens):
        """Mean-pool final hidden states over non-pad (expressed) genes."""
        h = self.encode(tokens)
        mask = (~tokens.eq(self.pad_id)).unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)

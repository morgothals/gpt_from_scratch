import torch
import torch.nn as nn
import torch.nn.functional as F


#csak paramétereket tárol, lásd README
class GPTConfig:
    def __init__(self, vocab_size, block_size, n_embd=128, n_head=4, n_layer=2):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer



class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token és pozíció embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)

        # Egyszerű transformer blokkok
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                batch_first=True
            ) for _ in range(config.n_layer)
        ])

        # Kimeneti réteg (logit minden tokenre)
        self.ln = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        tok_emb = self.token_embed(idx)                                # (B, T, C)
        pos = torch.arange(T, device=idx.device)                       # (T,)
        pos_emb = self.pos_embed(pos)                                  # (T, C)
        x = tok_emb + pos_emb                                          # (B, T, C)

        x = self.blocks(x)                                             # (B, T, C)
        x = self.ln(x)
        logits = self.head(x)                                          # (B, T, vocab_size)
        return logits



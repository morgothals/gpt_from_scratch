import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Ez lesz az embedding és egyben a kimenet is
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) – batch, time
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Átalakítjuk a bemeneteket lapos tömbbé (batch*T), hogy loss-t számoljunk
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # csak az utolsó tokenre
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

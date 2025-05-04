import torch
import torch.nn.functional as F

# 1. Karakterkészlet
chars = sorted(list(set("hello world")))
vocab_size = len(chars)

# 2. Token ↔ karakter szótárak
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# 3. Egy minta tanítópár
x_str = "hell"
y_str = "ello"  # következő karakterek

# 4. Tokenizálás
x = torch.tensor([stoi[c] for c in x_str])  # [7, 4, 10, 10]
y = torch.tensor([stoi[c] for c in y_str])  # [4, 10, 10, 6]

# 5. Saját súlymátrix definiálása (vocab_size × vocab_size)
W = torch.randn((vocab_size, vocab_size), requires_grad=True)

# 6. Előrefelé számolás (logit = W[token])
logits = W[x]  # (4, vocab_size)
probs = F.softmax(logits, dim=1)  # (4, vocab_size)

# 7. Loss (cross entropy a softmax és cél között)
loss = F.cross_entropy(logits, y)

print("Loss:", loss.item())

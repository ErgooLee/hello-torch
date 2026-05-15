import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out

x = torch.randn(2, 5, 32)   # batch=2, seq_len=5, embed_dim=32
attn = SelfAttention(32)
y = attn(x)
print(y.shape)  # torch.Size([2, 5, 32])

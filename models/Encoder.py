import torch
import torch.nn as nn
from .FFN import FFN
from .MHAttention import MHAttention

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, drop_p):
        super().__init__()

        self.MHA = MHAttention(input_dim, n_heads)
        self.FFN = FFN(input_dim, hidden_dim, drop_p)
        self.LN1 = nn.LayerNorm(input_dim)
        self.LN2 = nn.LayerNorm(input_dim)

    def forward(self, x):

        x = x + self.MHA(self.LN1(x))
        x = x + self.FFN(self.LN2(x))
        return x
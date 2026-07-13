import torch
import torch.nn as nn

class MHAttention(nn.Module):
    def __init__(self, n_dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = torch.tensor((n_dim / n_heads) ** 0.5)

        self.fc_q = nn.Linear(n_dim, n_dim)
        self.fc_k = nn.Linear(n_dim, n_dim)
        self.fc_v = nn.Linear(n_dim, n_dim)
        self.fc_o = nn.Linear(n_dim, n_dim)


    def forward(self, x):
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)

        Q = Q.view(x.shape[0], x.shape[1], self.n_heads, x.shape[2] // self.n_heads).transpose(1, 2)
        K = K.view(x.shape[0], x.shape[1], self.n_heads, x.shape[2] // self.n_heads).transpose(1, 2)
        V = V.view(x.shape[0], x.shape[1], self.n_heads, x.shape[2] // self.n_heads).transpose(1, 2)

        attention = Q @ K.transpose(-1, -2) / self.d_k

        attention = torch.softmax(attention, dim=-1)
        attention = attention @ V

        out = attention.transpose(1, 2).contiguous()
        out = out.view(*x.shape)
        out = self.fc_o(out)
        return out
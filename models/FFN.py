import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_p):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_p)
        
    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))
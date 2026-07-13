import torch
import torch.nn as nn
from .Encoder import Encoder
import math 

class VIT(nn.Module):
    def __init__(self, img_size, patch_size, n_layer, input_dim, hidden_dim, n_heads, drop_p, n_class, representation_size=None):
        super().__init__()

        self.n_layer = n_layer
        patch_num = (img_size // patch_size) ** 2 # patch_size 비율 맞다고 가정 (resize로 조절함)
        self.pos_embedding = nn.Parameter(0.02*torch.randn(patch_num+1, input_dim)) # cls_token 1추가
        self.input_embedding = nn.Conv2d(3, input_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(input_dim))

        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([Encoder(input_dim, hidden_dim, n_heads, drop_p) for _ in range(n_layer)])

        self.LN = nn.LayerNorm(input_dim)

        if representation_size is None: # fine-tuning
            self.head = nn.Linear(input_dim, n_class)
        else: # pre-training
            self.head = nn.Sequential(
                nn.Linear(input_dim, representation_size),
                nn.Tanh(),
                nn.Linear(representation_size, n_class)
            )


        self.apply(self._init_weights)

        for name, m in self.named_modules():
            if name.endswith(('fc_o', 'fc2')):
                nn.init.normal_(m.weight, mean=0.0, std=0.02 / (2 * self.n_layer) ** 0.5)

        # conv의 patch 개수가 많아지면 분산이 커지는것 방지
        fan_in = 3 * (patch_size ** 2) 
        nn.init.trunc_normal_(self.input_embedding.weight, std=math.sqrt(1 / fan_in))

        if representation_size is None: # fine-tuning
            nn.init.zeros_(self.head.weight)
        else: # pre-training
            nn.init.zeros_(self.head[2].weight)

    def forward(self, x):

        x = self.input_embedding(x).flatten(2).transpose(1, 2) # 개 채 행 열 -> 개 채 차 -> 개 패 차
        x = torch.concat([self.cls_token.expand(x.shape[0], 1, -1), x], dim=1) # 개 (cls+패) 차
        x = x + self.pos_embedding.expand_as(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
        
        x = self.LN(x)
        out = self.head(x[:, 0])
        return out

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
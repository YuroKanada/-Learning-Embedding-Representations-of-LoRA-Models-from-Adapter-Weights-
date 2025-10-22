# model/aggregator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, use_mlp=True, dropout=0.05, temperature=1.0):
        super().__init__()
        self.use_mlp = use_mlp

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True),
                nn.GELU(), 
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim, bias=True)  # ← スカラー → ベクトル重み化
            )
            # 温度パラメータ
            self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)  # 固定でも学習可
        else:
            self.mlp = None

    def forward(self, x):  # x: [B, T, D]
        if self.use_mlp:
            scores = self.mlp(x)                          # [B, T, D]
            attn_weights = torch.softmax(scores / self.temperature.clamp_min(1e-6), dim=1)  # 温度付きsoftmax
            pooled = torch.sum(attn_weights * x, dim=1)   # [B, D]
            return pooled
        else:
            return x.mean(dim=1)

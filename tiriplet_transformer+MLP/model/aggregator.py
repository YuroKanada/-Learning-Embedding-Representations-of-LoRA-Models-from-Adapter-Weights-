# model/aggregator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, use_mlp=True):
        super().__init__()
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x):
        if self.use_mlp:
            weights = self.mlp(x).squeeze(-1)
            attn_weights = torch.softmax(weights, dim=1)
            return torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        else:
            return x.mean(dim=1)

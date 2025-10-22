# model/triplet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletTransformer(nn.Module):
    def __init__(self, encoder, aggregator):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator

    def forward(self, a, p, n, use_aggregator=True):
        if use_aggregator:
            a_out = self.aggregator(self.encoder(a))
            p_out = self.aggregator(self.encoder(p))
            n_out = self.aggregator(self.encoder(n))
        else:
            # aggregatorをスキップ（ablation用）
            a_out = self.encoder(a).mean(dim=1)
            p_out = self.encoder(p).mean(dim=1)
            n_out = self.encoder(n).mean(dim=1)
        
        a_out = F.normalize(a_out, dim=-1)
        p_out = F.normalize(p_out, dim=-1)
        n_out = F.normalize(n_out, dim=-1)
        return a_out, p_out, n_out

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        pos_dist = 1 - F.cosine_similarity(a, p)
        neg_dist = 1 - F.cosine_similarity(a, n)
        return F.relu(pos_dist - neg_dist + self.margin).mean()

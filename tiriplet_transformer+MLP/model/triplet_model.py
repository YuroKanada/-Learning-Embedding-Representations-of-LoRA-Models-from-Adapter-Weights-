# model/triplet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletTransformer(nn.Module):
    def __init__(self, encoder, aggregator):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator

    def forward(self, a, p, n):
        a_out = self.aggregator(self.encoder(a))
        p_out = self.aggregator(self.encoder(p))
        n_out = self.aggregator(self.encoder(n))
        return a_out, p_out, n_out

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        pos_dist = 1 - F.cosine_similarity(a, p)
        neg_dist = 1 - F.cosine_similarity(a, n)
        return F.relu(pos_dist - neg_dist + self.margin).mean()

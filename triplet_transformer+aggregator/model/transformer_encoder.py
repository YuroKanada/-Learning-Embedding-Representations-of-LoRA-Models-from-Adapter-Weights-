# model/transformer_encoder.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, T, _ = Q.size()
        Q = self.q_linear(Q).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(K).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(V).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        context, _ = scaled_dot_product_attention(Q, K, V, mask)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.out(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        # # dropout追加
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # dropout追加
        # return self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.linear2(F.gelu(self.linear1(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        
        #pre-LN
        x1 = self.norm1(x)                     # ← 1回だけ
        attn_out = self.self_attn(x1, x1, x1, mask)
        x = x + self.dropout(attn_out)         # residual

        x2 = self.norm2(x)                     # ← 1回だけ
        ff_out = self.ff(x2)
        x = x + self.dropout(ff_out)           # residual
        return x
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, _ = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.pos_embedding(pos)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, max_len, use_pos_emb=True):
        super().__init__()
        self.pos_embed = LearnablePositionalEmbedding(max_len, d_model) if use_pos_emb else nn.Identity()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

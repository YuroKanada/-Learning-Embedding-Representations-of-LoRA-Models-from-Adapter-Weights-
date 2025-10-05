import os
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from datetime import datetime
import wandb

# === 基本設定 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIMENSION = 256
MAX_EPOCHS = 10
MARGIN = 0.4
BATCH_SIZE = 128

timestamp = datetime.now().strftime("%Y%m%d%H%M")
os.environ["WANDB_API_KEY"] = "d6844f09be56692d62d1768b582872fe142687ef"

# ===============================
#  モデル定義
# ===============================
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
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
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
    def forward(self, x):
        B, T, _ = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.pos_embedding(pos)
    
class TokenAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 重みスカラー出力
        )

    def forward(self, x):  # x: [B, T, D]
        weights = self.mlp(x).squeeze(-1)         # [B, T]
        attn_weights = torch.softmax(weights, dim=1)  # [B, T]
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)  # [B, D]
        return pooled

# class TokenAggregator(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#     def forward(self, x):
#         return x.mean(dim=1)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=DIMENSION, N=6, num_heads=4, d_ff=512, max_len=270):
        super().__init__()
        self.pos_embed = LearnablePositionalEmbedding(max_len, d_model)
        #self.pos_embed = nn.Identity()  # 位置エンコーディングを使わない場合は nn.Identity() に変更
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(N)])
    def forward(self, src, mask=None):
        x = self.pos_embed(src)
        for layer in self.layers:
            x = layer(x, mask)
        return x

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

# ===============================
#  データセット
# ===============================
class TripletModelDataset(torch.utils.data.Dataset):
    def __init__(self, triplets, model_matrix_dict, device):
        self.triplets = triplets
        self.model_matrix_dict = model_matrix_dict
        self.device = device
    def __len__(self):
        return len(self.triplets)
    def __getitem__(self, idx):
        t = self.triplets[idx]
        return (
            self.model_matrix_dict[t["anchor"]],
            self.model_matrix_dict[t["positive"]],
            self.model_matrix_dict[t["negative"]],
        )

# ===============================
#  評価関数
# ===============================
def evaluate_triplet_accuracy_image(model, model_matrix_dict, triplets, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for t in random.sample(triplets, min(2000, len(triplets))):
            a, p, n = (model_matrix_dict[t[k]].unsqueeze(0).to(device) for k in ("anchor", "positive", "negative"))
            a_vec, p_vec, n_vec = model(a, p, n)
            sim_ap = F.cosine_similarity(a_vec, p_vec).item()
            sim_an = F.cosine_similarity(a_vec, n_vec).item()
            total += 1
            if sim_ap > sim_an:
                correct += 1
    acc = correct / total if total > 0 else 0
    print(f"🔍 Triplet Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# ===============================
#  前処理：ベクトル読み込み
# ===============================
category_dir = "compressed_rank32"
all_files = sorted([f for f in os.listdir(category_dir) if f.endswith(".npz")])
category_names = [f.replace("compressed_", "").replace(f"_{DIMENSION}d.npz", "") for f in all_files]

temp = defaultdict(dict)
for fname, cat in zip(all_files, category_names):
    data = np.load(os.path.join(category_dir, fname))
    for mid, vec in zip(data["model_ids"], data["vectors"]):
        temp[str(mid)][cat] = vec

model_matrix_dict = {
    mid: torch.tensor(np.stack([temp[mid][c] for c in category_names]), dtype=torch.float32)
    for mid in temp if len(temp[mid]) == len(category_names)
}
print(f"✅ Model matrix loaded: {len(model_matrix_dict)} models")

# ===============================
#  Triplet読み込み
# ===============================
dataset_dir = "image_base_dataset"
with open(os.path.join(dataset_dir, "triplets_rank32_semi15_easy5_too_easy3.jsonl")) as f:
    train_triplets = [json.loads(l) for l in f]
with open(os.path.join(dataset_dir, "triplets_rank32_semi15_easy5_val.jsonl")) as f:
    val_triplets = [json.loads(l) for l in f]
with open(os.path.join(dataset_dir, "triplets_rank32_semi15_easy5_test.jsonl")) as f:
    test_triplets = [json.loads(l) for l in f]

print(f"Train: {len(train_triplets)}, Val: {len(val_triplets)}, Test: {len(test_triplets)}")

# ===============================
#  学習準備
# ===============================
wandb.init(
    project="triplet-image-sim_base",
    name=f"triplet_run_{timestamp}",
    config={
        # --- 基本設定 ---
        "dimension": DIMENSION,
        "margin": MARGIN,
        "epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": 1e-4,
        "optimizer": "Adam",
        "scheduler": None,  # (例) "cosine" や "sigmoid" を使う場合はここに記録

        # --- Transformer構造 ---
        "num_layers": 6,
        "num_heads": 4,
        "ff_dim": 512,
        "positional_embedding": "learnable",  # or "none"

        # --- Aggregator設定 ---
        "aggregator_type": "MLPWeighted",  # or "mean"
        "aggregator_hidden_dim": 512,
    }
)

encoder = TransformerEncoder().to(DEVICE)
aggregator = TokenAggregator(DIMENSION).to(DEVICE)
model = TripletTransformer(encoder, aggregator).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = TripletLoss(margin=MARGIN)

train_loader = DataLoader(TripletModelDataset(train_triplets, model_matrix_dict, DEVICE), batch_size=BATCH_SIZE, shuffle=True)

# ===============================
#  学習ループ
# ===============================
best_acc = 0.0
for epoch in range(MAX_EPOCHS):
    model.train()
    total_loss = 0
    for a, p, n in train_loader:
        a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
        a_out, p_out, n_out = model(a, p, n)
        loss = loss_fn(a_out, p_out, n_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{MAX_EPOCHS}] Loss={avg_loss:.4f}")

    val_acc = evaluate_triplet_accuracy_image(model, model_matrix_dict, val_triplets, DEVICE)
    wandb.log({"epoch": epoch+1, "loss": avg_loss, "val_acc": val_acc})

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.encoder.state_dict(), f"saved_models/best_encoder_{timestamp}.pt")
        torch.save(model.aggregator.state_dict(), f"saved_models/best_aggregator_{timestamp}.pt")
        print(f"🆕 Improved val_acc={val_acc:.4f}")

# ===============================
#  テスト評価
# ===============================
print("\n🧪 Final test evaluation")
test_acc = evaluate_triplet_accuracy_image(model, model_matrix_dict, test_triplets, DEVICE)
wandb.log({"test_triplet_accuracy": test_acc})
wandb.finish()

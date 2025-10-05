# main_train.py
import os, json, numpy as np, torch, wandb
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader

from config import CONFIG
from model.transformer_encoder import TransformerEncoder
from model.aggregator import TokenAggregator
from model.triplet_model import TripletTransformer, TripletLoss
from utils.evaluate import evaluate_triplet_accuracy

# --- WandB init ---
timestamp = datetime.now().strftime("%Y%m%d%H%M")
wandb.init(
    project="triplet-lora-ablation",
    name=f"run_{timestamp}",
    config=CONFIG
)
DEVICE = CONFIG["device"]

# --- データ読み込み ---
category_dir = CONFIG["category_dir"]
files = sorted([f for f in os.listdir(category_dir) if f.endswith(".npz")])
category_names = [f.replace("compressed_", "").replace(f"_{CONFIG['dimension']}d.npz", "") for f in files]

temp = defaultdict(dict)
for fname, cat in zip(files, category_names):
    data = np.load(os.path.join(category_dir, fname))
    for mid, vec in zip(data["model_ids"], data["vectors"]):
        temp[str(mid)][cat] = vec

model_matrix_dict = {
    mid: torch.tensor(np.stack([temp[mid][c] for c in category_names]), dtype=torch.float32)
    for mid in temp if len(temp[mid]) == len(category_names)
}
print(f"✅ Model matrix loaded: {len(model_matrix_dict)} models")

# --- Tripletデータ ---
dataset_dir = CONFIG["dataset_dir"]
with open(os.path.join(dataset_dir, "triplets_rank32_semi15_easy5_too_easy3.jsonl")) as f:
    train_triplets = [json.loads(l) for l in f]
with open(os.path.join(dataset_dir, "triplets_rank32_semi15_easy5_val.jsonl")) as f:
    val_triplets = [json.loads(l) for l in f]
with open(os.path.join(dataset_dir, "triplets_rank32_semi15_easy5_test.jsonl")) as f:
    test_triplets = [json.loads(l) for l in f]

# --- モデル構築 ---
encoder = TransformerEncoder(
    d_model=CONFIG["dimension"],
    num_layers=CONFIG["num_layers"],
    num_heads=CONFIG["num_heads"],
    d_ff=CONFIG["ff_dim"],
    max_len=CONFIG["max_len"],
    use_pos_emb=CONFIG["use_positional_embedding"]
).to(DEVICE)

aggregator = TokenAggregator(
    CONFIG["dimension"], hidden_dim=CONFIG["ff_dim"], use_mlp=CONFIG["use_mlp_aggregator"]
).to(DEVICE)

model = TripletTransformer(encoder, aggregator).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
loss_fn = TripletLoss(CONFIG["margin"])

train_loader = DataLoader(
    [(model_matrix_dict[t["anchor"]], model_matrix_dict[t["positive"]], model_matrix_dict[t["negative"]]) for t in train_triplets],
    batch_size=CONFIG["batch_size"], shuffle=True
)

# --- 学習ループ ---
best_acc = 0.0
for epoch in range(CONFIG["epochs"]):
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
    val_acc = evaluate_triplet_accuracy(model, model_matrix_dict, val_triplets, DEVICE)
    wandb.log({"epoch": epoch+1, "loss": avg_loss, "val_acc": val_acc})

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.encoder.state_dict(), f"saved_models/best_encoder_{timestamp}.pt")
        torch.save(model.aggregator.state_dict(), f"saved_models/best_aggregator_{timestamp}.pt")
        print(f"🆕 Improved val_acc={val_acc:.4f}")

print("\n🧪 Final test evaluation")
test_acc = evaluate_triplet_accuracy(model, model_matrix_dict, test_triplets, DEVICE)
wandb.log({"test_triplet_accuracy": test_acc})
wandb.finish()

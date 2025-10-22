import os, wandb, torch
from datetime import datetime
from torch.utils.data import DataLoader
from config import CONFIG
from dataset.loader import load_model_matrix, load_triplets
from model.transformer_encoder import TransformerEncoder
from model.aggregator import TokenAggregator
from model.triplet_model import TripletTransformer, TripletLoss
from train.optimizer import build_optimizer_scheduler
from train.trainer import Trainer

# === 基本設定 ===
DEVICE = CONFIG["device"]
timestamp = datetime.now().strftime("%Y%m%d%H%M")
os.environ["WANDB_API_KEY"] = "your_wandb_api_key_here"  # Set your Weights & Biases API key
wandb.init(project="triplet-image_base", name=f"run_{timestamp}", config=CONFIG)

# === データ読み込み ===
model_matrix_dict = load_model_matrix(CONFIG["category_dir"], CONFIG["dimension"])

#データセット変更の場合ここを修正
train_triplets, val_triplets, test_triplets = load_triplets(
    CONFIG["dataset_dir"],
    "triplets_rank32_semi15_easy5_too_easy1.jsonl",
    "triplets_rank32_val_semi10_easy4_too_easy2.jsonl",
    "triplets_rank32_test_semi10_easy4_too_easy2.jsonl"
)

# train_loader = DataLoader(
#     [(model_matrix_dict[t["anchor"]], model_matrix_dict[t["positive"]], model_matrix_dict[t["negative"]])
#      for t in train_triplets],
#     batch_size=CONFIG["batch_size"], shuffle=True
# )
train_loader = DataLoader(
    [(model_matrix_dict[t["anchor"]], model_matrix_dict[t["positive"]], model_matrix_dict[t["negative"]])
     for t in train_triplets],
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=4,        # ← CPU並列処理数
    pin_memory=True        # ← GPU転送を高速化（CUDA利用時は推奨）
)


# === モデル構築 ===
encoder = TransformerEncoder(
    d_model=CONFIG["dimension"],
    num_layers=CONFIG["num_layers"],
    num_heads=CONFIG["num_heads"],
    d_ff=CONFIG["ff_dim"],
    max_len=CONFIG["max_len"],
    use_pos_emb=CONFIG["use_positional_embedding"]
).to(DEVICE)
aggregator = TokenAggregator(CONFIG["dimension"], hidden_dim=CONFIG["aggregator_hidden_dim"], use_mlp=CONFIG["use_mlp_aggregator"]).to(DEVICE)
model = TripletTransformer(encoder, aggregator).to(DEVICE)
loss_fn = TripletLoss(CONFIG["margin"])

# === Optimizer + Scheduler ===
optimizer, scheduler, warmup_steps = build_optimizer_scheduler(
    model, CONFIG["lr1"],CONFIG["lr2"], CONFIG["weight_decay"], CONFIG["batch_size"], CONFIG["epochs"], len(train_triplets)
)
wandb.config.update({
    "optimizer": "AdamW",
    "encoder_lr": CONFIG["lr1"],
    "aggregator_lr": CONFIG["lr2"],
    "weight_decay": CONFIG["weight_decay"],
    "scheduler_type": "warmup_cosine_decay",
    "warmup_steps": warmup_steps,
    "total_steps": len(train_triplets) // CONFIG["batch_size"] * CONFIG["epochs"]
})
# === 学習 ===
# trainer = Trainer(model, optimizer, scheduler, loss_fn, DEVICE)
trainer = Trainer(
    model, optimizer, scheduler, loss_fn, DEVICE,
    grad_threshold=CONFIG["grad_threshold"],
    freeze_epochs=CONFIG["freeze_epochs"],
    freeze_aggregator=CONFIG["freeze_aggregator"],
    aggregator_fixed_lr=CONFIG["aggregator_fixed_lr"]
)
for epoch in range(CONFIG["epochs"]):
    loss, grad_enc, grad_agg = trainer.train_epoch(train_loader, epoch)
    val_acc = trainer.validate(val_triplets, model_matrix_dict)
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": loss,
        "val_acc": val_acc,
        "grad_norm_encoder": grad_enc,
        "grad_norm_aggregator": grad_agg,
        "lr_encoder": optimizer.param_groups[0]["lr"],
        "lr_aggregator": optimizer.param_groups[2]["lr"]
    })
    trainer.maybe_save(epoch, val_acc, timestamp)

# === テスト評価 ===
test_acc = trainer.validate(test_triplets, model_matrix_dict)
wandb.log({"test_triplet_accuracy": test_acc})
wandb.finish()

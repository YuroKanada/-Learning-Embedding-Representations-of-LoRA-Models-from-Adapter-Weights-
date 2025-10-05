# config.py
import torch

CONFIG = {
    # === 実験設定 ===
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dimension": 256,
    "epochs": 10,
    "batch_size": 128,
    "margin": 0.4,
    "lr": 1e-4,

    # === Transformer構造 ===
    "num_layers": 6,
    "num_heads": 4,
    "ff_dim": 512,
    "max_len": 270,

    # === Ablationスイッチ ===
    "use_positional_embedding": True,   # Falseにするとnn.Identity()
    "use_mlp_aggregator": True,         # Falseにすると単純平均

    # === データパス ===
    "category_dir": "compressed_rank32",
    "dataset_dir": "image_base_dataset",
}

# config.py
import torch

CONFIG = {
    # === 実験設定 ===
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dimension": 256,
    "epochs": 20,
    "batch_size": 128,
    "margin": 0.3,
    "lr1": 1e-4,#encoder用の初期学習率
    "lr2": 1e-4,#aggregator用の学習率
    "weight_decay": 1e-4,
    
    # === 凍結制御 ===
    "freeze_aggregator": False,     # Trueで初期凍結
    "freeze_epochs": 0,            # 最低何epochは凍結を維持するか
    "grad_threshold": 0.05,        # encoder平均勾配ノルムがこの値を下回ったら解除
    "aggregator_fixed_lr": 1e-4,   # 解除後に固定するaggregatorの学習率

    # === Transformer構造 ===
    "num_layers": 4,
    "num_heads": 4,
    "ff_dim": 512,
    "max_len": 264,
    "aggregator_hidden_dim": 128, #MLPを追加する場合の隠れ層次元

    # === Ablationスイッチ ===
    "use_positional_embedding": False,   # Falseにするとnn.Identity()
    "use_mlp_aggregator": True,         # Falseにすると単純平均

    
    # === データパス ===
    "category_dir": "/home/kanada/my-jupyterlab/work-dir/compressed_rank32",
    "dataset_dir": "/home/kanada/my-jupyterlab/work-dir/image_base_dataset",
}

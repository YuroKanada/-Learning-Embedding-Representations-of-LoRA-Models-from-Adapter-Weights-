import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def separate_weight_decay_params(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # LayerNorm, bias, Embedding は decay除外
        if (
            "bias" in name
            or "norm" in name.lower()
            or "embedding" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay

def build_optimizer_scheduler(model, lr1, lr2, weight_decay, batch_size, num_epochs, train_len, warmup_ratio=0.2, min_lr_ratio=0.05):
    """
    Warm-up + Cosine Decay スケジューラを構築。
    - encoderとaggregatorで異なる学習率を設定
    - warmup終了後はcosineで緩やかに減衰
    """

    # # --- optimizer: encoderとaggregatorでLRを分離 ---
    # optimizer = AdamW([
    #     {"params": model.encoder.parameters(), "lr": lr1},   # encoderは安定学習
    #     {"params": model.aggregator.parameters(), "lr": lr2} # aggregatorは早く適応
    # ], weight_decay=weight_decay)


    enc_decay, enc_no_decay = separate_weight_decay_params(model.encoder)
    agg_decay, agg_no_decay = separate_weight_decay_params(model.aggregator)

    optimizer = torch.optim.AdamW([
        {"params": enc_decay, "lr": lr1, "weight_decay": weight_decay},  # weightのみ正則化
        {"params": enc_no_decay, "lr": lr1, "weight_decay": 0.0}, # bias, norm, embedding除外
        {"params": agg_decay, "lr": lr2, "weight_decay": weight_decay},    # aggregatorは固定 or 弱め
        {"params": agg_no_decay, "lr": lr2, "weight_decay": 0.0},
    ])
    # --- step数を算出 ---
    steps_per_epoch = math.ceil(train_len / batch_size)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(200, int(total_steps * warmup_ratio))  # 少なくとも200step確保

    # --- scheduler: warm-up + cosine decay ---
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        # 最低学習率をmin_lr_ratioで下支え
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    # scheduler = LambdaLR(optimizer, lr_lambda)
    #aggregatorはlrを固定
    # scheduler = LambdaLR(
    #     optimizer,
    #     lr_lambda=[lr_lambda, lr_lambda, lr_lambda, lambda step: 1.0]
    # )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[lr_lambda, lr_lambda, lr_lambda, lr_lambda]
    )

    return optimizer, scheduler, warmup_steps

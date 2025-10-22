import torch, wandb
import os
from utils.evaluate import evaluate_triplet_accuracy

class Trainer:
    def __init__(
        self, model, optimizer, scheduler, loss_fn, device, save_dir="saved_models",
        grad_threshold=0.05, freeze_epochs=2,
        freeze_aggregator=True, aggregator_fixed_lr=5e-4
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = save_dir
        self.best_acc = 0.0

        # === aggregator制御用 ===
        self.grad_threshold = grad_threshold
        self.freeze_epochs = freeze_epochs
        self.freeze_aggregator = freeze_aggregator
        self.aggregator_fixed_lr = aggregator_fixed_lr
        self.aggregator_frozen = freeze_aggregator

        os.makedirs(self.save_dir, exist_ok=True)

        # aggregatorは常に学習対象
        for p in self.model.aggregator.parameters():
            p.requires_grad = True
        print("🚀 Aggregator and Encoder training simultaneously.")

    def train_epoch(self, dataloader, epoch):           
        self.model.train()
        total_loss, total_grad_enc, total_grad_agg = 0, 0, 0

        for step, (a, p, n) in enumerate(dataloader):
            a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)

            # === forward ===
            a_out, p_out, n_out = self.model(a, p, n)
            loss = self.loss_fn(a_out, p_out, n_out)

            # === backward ===
            self.optimizer.zero_grad()
            loss.backward()

            # === 各レイヤごとの勾配ノルム ===
            layer_grads = {
                name: param.grad.norm().item()
                for name, param in self.model.encoder.named_parameters()
                if param.grad is not None
            }

            # --- 主要層のみ追跡（50stepごと） ---
            if step % 50 == 0:
                tracked_layers = ["self_attn", "ff.linear1", "ff.linear2"]
                tracked = {
                    f"grad/{lname}": g
                    for lname, g in layer_grads.items()
                    if any(k in lname for k in tracked_layers)
                }

                enc_grads = [p.grad.norm().item() for p in self.model.encoder.parameters() if p.grad is not None]
                grad_enc_mean = sum(enc_grads) / len(enc_grads) if enc_grads else 0.0
                agg_grads = [p.grad.norm().item() for p in self.model.aggregator.parameters() if p.grad is not None]
                grad_agg_mean = sum(agg_grads) / len(agg_grads) if agg_grads else 0.0

                wandb.log({
                    "step_loss": loss.item(),
                    "lr_encoder": self.optimizer.param_groups[0]["lr"],
                    "lr_aggregator": self.optimizer.param_groups[2]["lr"],
                    "grad_norm_encoder_step": grad_enc_mean,
                    "grad_norm_aggregator_step": grad_agg_mean,
                    "aggregator_frozen": self.aggregator_frozen,
                    **tracked,
                })

            # --- 勾配分布（200stepごと） ---
            if step % 200 == 0 and len(layer_grads) > 0:
                wandb.log({
                    "encoder_grad_distribution": wandb.Histogram(list(layer_grads.values())),
                })

            # === optimizer step ===
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

            # === grad統計 ===
            enc_grads = [p.grad.norm().item() for p in self.model.encoder.parameters() if p.grad is not None]
            grad_enc_mean = sum(enc_grads) / len(enc_grads) if enc_grads else 0.0
            total_grad_enc += grad_enc_mean

            agg_grads = [p.grad.norm().item() for p in self.model.aggregator.parameters() if p.grad is not None]
            grad_agg_mean = sum(agg_grads) / len(agg_grads) if agg_grads else 0.0
            total_grad_agg += grad_agg_mean

        # === epoch平均 ===
        avg_grad_enc = total_grad_enc / len(dataloader)
        avg_grad_agg = total_grad_agg / len(dataloader)
        avg_loss = total_loss / len(dataloader)

        wandb.log({
            "epoch_loss": avg_loss,
            "avg_grad_enc": avg_grad_enc,
            "avg_grad_agg": avg_grad_agg,
        })

        return avg_loss, avg_grad_enc, avg_grad_agg

    def validate(self, triplets, model_matrix_dict):
        acc = evaluate_triplet_accuracy(self.model, model_matrix_dict, triplets, self.device)
        return acc

    def maybe_save(self, epoch, val_acc, timestamp):
        """ベストモデルと定期保存"""
        # --- 定期保存（例：5エポックごと） ---
        if (epoch + 1) % 5 == 0:
            torch.save(self.model.encoder.state_dict(), f"{self.save_dir}/encoder_epoch{epoch+1}_{timestamp}.pt")
            torch.save(self.model.aggregator.state_dict(), f"{self.save_dir}/aggregator_epoch{epoch+1}_{timestamp}.pt")
            print(f"💾 Regular checkpoint saved at epoch {epoch+1}")

        # --- ベスト更新時 ---
        if val_acc > self.best_acc:
            # 古いベストを削除
            if hasattr(self, "best_encoder_path") and os.path.exists(self.best_encoder_path):
                os.remove(self.best_encoder_path)
                print(f"🗑️ Removed old best encoder: {os.path.basename(self.best_encoder_path)}")

            if hasattr(self, "best_aggregator_path") and os.path.exists(self.best_aggregator_path):
                os.remove(self.best_aggregator_path)
                print(f"🗑️ Removed old best aggregator: {os.path.basename(self.best_aggregator_path)}")

            # 新しいベストを保存
            best_encoder_path = f"{self.save_dir}/best_encoder_epoch{epoch+1}_{timestamp}.pt"
            best_aggregator_path = f"{self.save_dir}/best_aggregator_epoch{epoch+1}_{timestamp}.pt"
            torch.save(self.model.encoder.state_dict(), best_encoder_path)
            torch.save(self.model.aggregator.state_dict(), best_aggregator_path)

            # 記録更新
            self.best_acc = val_acc
            self.best_encoder_path = best_encoder_path
            self.best_aggregator_path = best_aggregator_path

            print(f"🏆 New BEST model saved at epoch {epoch+1} (val_acc={val_acc:.4f})")


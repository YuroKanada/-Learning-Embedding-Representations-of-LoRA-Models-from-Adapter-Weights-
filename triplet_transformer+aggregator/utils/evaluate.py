# utils/evaluate.py
import torch
import torch.nn.functional as F
import random

def evaluate_triplet_accuracy(model, model_matrix_dict, triplets, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for t in triplets:
            # anchor, positive, negative のテンソル取得
            a, p, n = (model_matrix_dict[t[k]].unsqueeze(0).to(device) for k in ("anchor", "positive", "negative"))

            # 推論
            a_vec, p_vec, n_vec = model(a, p, n)

            # 類似度を計算
            sim_ap = F.cosine_similarity(a_vec, p_vec).item()
            sim_an = F.cosine_similarity(a_vec, n_vec).item()

            total += 1
            if sim_ap > sim_an:
                correct += 1

    acc = correct / total if total > 0 else 0
    # print(f"🔍 Triplet Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


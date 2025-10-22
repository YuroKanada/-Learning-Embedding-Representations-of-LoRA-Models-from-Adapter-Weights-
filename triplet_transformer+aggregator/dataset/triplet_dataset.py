# dataset/triplet_dataset.py
import torch
from torch.utils.data import Dataset

class TripletModelDataset(Dataset):
    """
    LoRAモデルの三つ組データセット
    各要素は anchor, positive, negative モデルIDに対応し、
    それぞれ model_matrix_dict 内の [T, D] ベクトルを返す。
    """
    def __init__(self, triplets, model_matrix_dict, device):
        """
        Args:
            triplets: [{"anchor": id1, "positive": id2, "negative": id3}, ...]
            model_matrix_dict: {model_id: torch.Tensor([T, D])}
            device: torch.device
        """
        self.triplets = triplets
        self.model_matrix_dict = model_matrix_dict
        self.device = device

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]

        # 各モデルIDからテンソルを取得
        a = self.model_matrix_dict[t["anchor"]]
        p = self.model_matrix_dict[t["positive"]]
        n = self.model_matrix_dict[t["negative"]]

        # DataLoaderがCPU上で動作することもあるため、device移動はmain側で行う
        return a, p, n

import os, json, numpy as np, torch
from collections import defaultdict

def load_model_matrix(category_dir, dimension):
    files = sorted([f for f in os.listdir(category_dir) if f.endswith(".npz")])
    category_names = [f.replace("compressed_", "").replace(f"_" + str(dimension) + "d.npz", "") for f in files]
    temp = defaultdict(dict)
    for fname, cat in zip(files, category_names):
        data = np.load(os.path.join(category_dir, fname))
        for mid, vec in zip(data["model_ids"], data["vectors"]):
            temp[str(mid)][cat] = vec
    model_matrix_dict = {
        mid: torch.tensor(np.stack([temp[mid][c] for c in category_names]), dtype=torch.float32)
        for mid in temp if len(temp[mid]) == len(category_names)
    }
    print(f"✅ Loaded {len(model_matrix_dict)} model matrices")
    return model_matrix_dict


def load_triplets(dataset_dir, train_name, val_name, test_name):
    def _load(fname):
        with open(os.path.join(dataset_dir, fname)) as f:
            return [json.loads(l) for l in f]
    return _load(train_name), _load(val_name), _load(test_name)

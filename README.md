# -Learning-Embedding-Representations-of-LoRA-Models-from-Adapter-Weights-

This repository implements a framework for **learning embedding representations of LoRA models** directly from their **adapter weights**.  
The goal is to obtain compact and interpretable representations that reflect the characteristics of each LoRA model, enabling **similarity estimation**, **retrieval**, and **model analysis**.

## Project Structure
├── config.py
├── main_train.py
├── dataset/
│ └── triplet_dataset.py
├── model/
│ ├── transformer_encoder.py
│ ├── aggregator.py
│ └── triplet_model.py
├── utils/
│ └── evaluate.py
└── saved_models/

## 🧩 Module Overview

### **config.py**
Defines global hyperparameters, model settings, and dataset paths.

### **main_train.py**
Main training script.  
Loads LoRA adapters, builds triplet data, trains the model, and logs results to **wandb**.

### **dataset/triplet_dataset.py**
Defines a PyTorch Dataset for triplet samples `(anchor, positive, negative)`.

### **model/transformer_encoder.py**
Implements a lightweight Transformer encoder for LoRA weight sequences.

### **model/aggregator.py**
Implements `TokenAggregator`, which summarizes token-level outputs via MLP weighting or mean pooling.

### **model/triplet_model.py**
Combines the encoder and aggregator into a unified `TripletTransformer` with margin-based Triplet Loss.

### **utils/evaluate.py**
Computes triplet accuracy using cosine similarity.

---

## ⚙️ Training

```bash
python main_train.py

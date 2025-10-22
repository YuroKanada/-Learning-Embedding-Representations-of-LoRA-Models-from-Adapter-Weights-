# -Learning-Embedding-Representations-of-LoRA-Models-from-Adapter-Weights-

This repository implements a framework for **learning embedding representations of LoRA models** directly from their **adapter weights**.  
The goal is to obtain compact and interpretable representations that reflect the characteristics of each LoRA model, enabling **similarity estimation**, **retrieval**, and **model analysis**.

## Repository Structure

**triplet_transformer+aggregator**/
├── config.py                # Global configuration (hyperparameters, paths, etc.)
├── main_train.py            # Entry point for training and evaluation
│
├── **dataset**/
│   ├── loader.py            # NPZ/JSONL data loader
│   └── triplet_dataset.py   # TripletDataset class for (anchor, positive, negative) samples
│
├── **model**/
│   ├── transformer_encoder.py  # Transformer encoder for adapter sequences
│   ├── aggregator.py           # Token aggregator (attention / MLP / mean pooling)
│   └── triplet_model.py        # Combined TripletTransformer model and TripletLoss
│
├── **train**/
│   ├── optimizer.py         # Optimizer and learning rate scheduler
│   └── trainer.py           # Training loop, logging, and checkpoint management
│
└── **utils**/
    └── evaluate.py          # Triplet accuracy and similarity evaluation

--

## ⚙️ Training

```bash
python3 main_train.py

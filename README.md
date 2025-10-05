# -Learning-Embedding-Representations-of-LoRA-Models-from-Adapter-Weights-

This repository implements a framework for **learning embedding representations of LoRA models** directly from their **adapter weights**.  
The goal is to obtain compact and interpretable representations that reflect the characteristics of each LoRA model, enabling **similarity estimation**, **retrieval**, and **model analysis**.

## Repository Structure

- **config/**
  - `config.py` – global settings (hyperparameters, model dimensions, dataset paths)

- **dataset/**
  - `triplet_dataset.py` – PyTorch dataset for triplet samples `(anchor, positive, negative)`

- **model/**
  - `transformer_encoder.py` – Transformer encoder for LoRA adapter sequences
  - `aggregator.py` – token-level aggregator (MLP-based weighting or mean pooling)
  - `triplet_model.py` – TripletTransformer model and TripletLoss definition

- **utils/**
  - `evaluate.py` – triplet accuracy evaluation using cosine similarity

- **scripts/**
  - `main_train.py` – main training script (data loading, training loop, wandb logging)

- **saved_models/** – directory for trained encoder and aggregator checkpoints

- **compressed_rank32/** – directory containing compressed LoRA adapter weight vectors (`.npz`)

- **image_base_dataset/** – directory containing triplet data files (`.jsonl`)

--

## ⚙️ Training

```bash
python main_train.py

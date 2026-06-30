# Retrieval of LoRA Models based on <br>Layer-Wise Weight Embedding without Metadata</br>

This repository implements a framework for **learning embedding representations of LoRA models** directly from their **adapter weights**.  
The goal is to obtain compact and interpretable representations that reflect the characteristics of each LoRA model, enabling **similarity estimation**, **retrieval**, and **model analysis**.

## Publication

This work has been accepted to **ICMR 2026**:

Retrieval of LoRA Models based on Layer-Wise Weight Embedding without Metadata, Yuro Kanada, Yuma Oe, Huu-Long Pham, Makoto P. Kato, Hiroaki Ohshima, Sumio Fujita and Yoshiyuki Shoji, Proc. of The 16th ACM International Conference on Multimedia Retrieval (ICMR2026), to appear, 2026.

## Citation

If you use this repository or the encoder in your research, please cite:

```bibtex
@inproceedings{Kanada2026retrieval,
author = {Kanada, Yuro and Oe, Yuma and Pham, Huu-Long and Kato, Makoto P. and Ohshima, Hiroaki and Fujita, Sumio and Shoji, Yoshiyuki},
title = {Retrieval of LoRA Models based on Layer-Wise Weight Embedding without Metadata},
year = {2026},
doi = {10.1145/3805622.3810428},
booktitle = {Proceedings of the 2026 International Conference on Multimedia Retrieval},
pages = {2818–2825},
series = {ICMR '26}
}
```

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

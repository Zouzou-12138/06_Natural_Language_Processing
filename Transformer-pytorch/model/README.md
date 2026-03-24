# 🚀 PyTorch-Transformer-From-Scratch

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **A from-scratch implementation of the Transformer model ("Attention Is All You Need") using PyTorch.**  
> This project replicates the original architecture featuring Multi-Head Attention, Positional Encoding, Encoder-Decoder stacks, and advanced training utilities like Noam Optimization and Multi-GPU support.

## 🌟 Key Features

- **Complete Architecture**: Implements every component described in the original paper:
  - **Embeddings & Positional Encoding**: Sinusoidal position injections for sequence order.
  - **Multi-Head Attention**: Scaled dot-product attention with multiple heads.
  - **Encoder-Decoder Stack**: 6-layer stacked architecture with residual connections and Layer Normalization.
  - **Feed-Forward Networks**: Position-wise fully connected feed-forward networks.
- **Advanced Training Utilities**:
  - **Noam Learning Rate Scheduler**: Dynamic learning rate adjustment with warmup steps (`warmup * step^-0.5`).
  - **Multi-GPU Support**: Distributed loss computation and gradient aggregation across multiple devices.
  - **Label Smoothing**: Integrated into the loss criterion for better generalization.
- **Modular Design**: Clean separation of model definition (`tf_model.py`) and training logic (`train_utils.py`).
- **Educational Value**: Heavily commented code explaining the mathematical intuition behind each layer.

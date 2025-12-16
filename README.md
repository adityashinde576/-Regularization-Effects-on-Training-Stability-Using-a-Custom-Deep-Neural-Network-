# MNIST Regularization Study using PyTorch

## Overview

This project explores the impact of regularization techniques on a Fully Connected Neural Network (FCNN) trained on the MNIST handwritten digits dataset using PyTorch.

The objective is to compare training stability, generalization, and overfitting behavior under different regularization settings.

## Objectives

- Understand overfitting in deep neural networks
- Apply and compare:
  - L2 Regularization (Weight Decay)
  - Batch Normalization
  - Early Stopping
- Visualize training and validation performance
- Build clean, reusable PyTorch training pipelines

## Model Architecture

Fully Connected Neural Network:

- Input (784)
- Linear(256) → ReLU → (BatchNorm optional)
- Linear(128) → ReLU → (BatchNorm optional)
- Linear(64)  → ReLU → (BatchNorm optional)
- Linear(10)

- Activation: ReLU  
- Loss Function: CrossEntropyLoss

## Dataset

- MNIST
- 60,000 training samples
- 10,000 test samples
- Image size: 28 × 28
- Automatically downloaded via `torchvision.datasets`

## Training Configuration

| Parameter                 | Value            |
|----------------------------|----------------|
| Optimizer                 | Adam            |
| Learning Rate             | 0.001           |
| Batch Size                | 128             |
| Max Epochs                | 50              |
| Early Stopping Patience   | 5               |
| Device                    | CPU / CUDA      |

## Metrics & Visualization

Metrics tracked per epoch:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

Plots generated:

- Loss vs Epochs
- Accuracy vs Epochs

These plots demonstrate overfitting reduction and training stability improvements.

## How to Run

1. **Create Virtual Environment (Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

2.Install Dependencies

pip install torch torchvision matplotlib


3.Run the Training Script

python train_mnist_regularization.py



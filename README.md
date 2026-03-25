# Glx.MLExamples

A collection of PyTorch and PyTorch Lightning machine learning examples, ranging from beginner-friendly regression demos to practical training pipelines for vector recall, variational autoencoders, and Stable Diffusion.

## Repository Overview

This repository is organized into three main areas:

- `examples/`: lightweight Lightning demos for common deep learning tasks
- `project/`: more complete end-to-end ML projects and training pipelines
- `scripts/`: standalone experiments and infrastructure validation scripts

## Subprojects

### `examples/01-lightning-demo-transformer`
A minimal language modeling example built with Lightning's demo `Transformer` and the `WikiText2` dataset. It shows the standard training, validation, and testing workflow for a sequence model with a compact `LightningModule`.

**Highlights**
- Transformer-based language model
- WikiText2 dataset
- Simple Lightning training loop
- Suitable as a starter example for NLP experiments

### `examples/02-lightning-demo-resnet50`
An image classification demo that uses transfer learning for CIFAR-10 with a ResNet-style backbone. It demonstrates how to structure a Lightning image training workflow with a separate dataset module, model definition, and configuration file.

**Highlights**
- Transfer learning for image classification
- CIFAR-10 data pipeline
- Config-driven training entry point
- Multi-device and precision settings through Lightning

### `examples/03-lightning-demo-autoencoder`
A classic MNIST autoencoder example based on PyTorch Lightning. It includes a `LightningCLI` setup and an image sampling callback for visualizing reconstructions during training.

**Highlights**
- MNIST autoencoder training
- Reconstruction monitoring with saved image grids
- `LightningCLI` integration
- Good reference for unsupervised learning workflows

### `examples/04-lightning-demo-Backbone_Image_Classifier`
A compact MNIST classifier that separates a reusable backbone network from the Lightning training wrapper. It is useful for understanding how to structure classification models with clean module boundaries.

**Highlights**
- Backbone-based image classifier
- MNIST dataset
- Clean separation of model and training logic
- Supports training, testing, and prediction flows

### `project/01/01-LinearRegression`
A set of introductory PyTorch scripts covering simple supervised learning tasks. These examples progress from fitting a linear relation to learning a nonlinear function and a small binary classification task.

**Included examples**
- `example01.py`: fits a noisy linear function similar to $y = 2x + 1$
- `example02.py`: learns a nonlinear noisy signal with a deeper MLP
- `example03.py`: solves a small OR-style binary classification problem

**Highlights**
- Beginner-friendly PyTorch basics
- Dataset creation, model definition, training, and inference
- Visualizations with Matplotlib

### `project/02/02-Vector Recall`
A practical vector recall system for recommendation scenarios. The project trains a two-tower retrieval model and uses an HNSW index for fast approximate nearest neighbor search over large-scale item embeddings.

**Highlights**
- Two-tower user/item encoder architecture
- Contrastive learning with vector similarity objectives
- HNSW-based ANN retrieval
- Training, indexing, and recall service components
- Designed for high-throughput recommendation recall workloads

### `project/03/sd_vae_lightning`
A PyTorch Lightning training framework for the Stable Diffusion 1.5 `AutoencoderKL` VAE. It focuses on image-to-latent and latent-to-image learning, with support for reconstruction, perceptual, KL, and adversarial losses.

**Highlights**
- Stable Diffusion VAE training and inference
- Lightning-based modular training pipeline
- Optional discriminator for adversarial training
- TensorBoard logging and checkpoint management
- Hugging Face dataset and pretrained weight integration

### `project/04/StableDiffusionTrainer`
A Stable Diffusion 1.5 training project built with PyTorch Lightning and Diffusers. The current implementation centers on UNet denoiser training while keeping the VAE and text encoder frozen.

**Highlights**
- Latent diffusion UNet training
- Diffusers-based Stable Diffusion pipeline integration
- Resume training, checkpoint export, and TensorBoard logging
- Validation image generation and inference scripts
- Configurable training for practical experimentation

### `scripts/lightning-FSDP-test`
A focused experiment for validating Fully Sharded Data Parallel training with Lightning Fabric. It initializes a large Transformer model, applies FSDP wrapping, and records memory-related behavior during execution.

**Highlights**
- Lightning Fabric + FSDP strategy test
- Large Transformer initialization
- BF16 precision experiment
- Useful for distributed training and memory profiling checks

## Suggested Audience

This repository is useful for:

- beginners learning PyTorch and Lightning fundamentals
- practitioners exploring project structure for ML training code
- researchers prototyping diffusion, retrieval, and representation learning workflows
- engineers validating distributed training strategies

## Notes

Some subprojects include their own `README.md`, configuration files, and dependency lists. For setup and usage details, please refer to the documentation inside each subproject directory.

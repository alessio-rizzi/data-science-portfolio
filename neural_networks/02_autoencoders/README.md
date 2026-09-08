# Exercise: Autoencoders (Undercomplete & Variational)

## Overview
These two notebooks explore representation learning on the MNIST dataset through convolutional **autoencoders**. Both share the same encoder/decoder backbone, but differ in how the latent space is structured:

* **02_undercomplete_autoencoders**: A classic undercomplete autoencoder that compresses each digit image into a low-dimensional latent vector and learns to reconstruct it, forcing the network to discover a compact, informative representation.
* **03_VAE_autoencoders**: A Variational Autoencoder (VAE) built on the same architecture, which instead learns a *probabilistic* latent space (via mean and variance parameters), enabling smoother interpolation and proper generation of new digit images.

## Dataset
Both notebooks use the **MNIST** dataset (28x28 grayscale handwritten digits), downloaded automatically via `torchvision.datasets.MNIST` and loaded through `DataLoader` with a batch size of 64.

## Process and Methodology

### 1. Shared Convolutional Architecture
Both models use the same encoder/decoder backbone:
* **Encoder**: three convolutional layers (`Conv2d`) with ReLU activations — (1→16, 16→32, 32→64 channels), progressively downsampling the 28x28 image down to a 1x1 feature map.
* **Latent bottleneck**: a fully connected layer mapping the 64-dimensional feature vector into a small latent space (`z_size`).
* **Decoder**: a fully connected layer followed by three transposed convolutions (`ConvTranspose2d`) that upsample the latent vector back into a 28x28 reconstructed image.

### 2. Undercomplete Autoencoder (02)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hwxhMGR-SJeX_fQ29kIjoaH7Rwjeklt9?usp=sharing)

* The encoder maps directly to a single latent vector `z` via `encoder4` (a `Linear` layer).
* The model is trained end-to-end with **MSE reconstruction loss** (`torch.nn.MSELoss`) using the Adam optimizer.
* After training, the notebook:
  * Visualizes original vs. reconstructed digits side by side.
  * Projects the latent space to 2D using **t-SNE** to inspect how digit classes cluster.
  * Estimates the min/max range of each latent dimension and samples random points within that range to **generate new digits** from the decoder.

### 3. Variational Autoencoder (03)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11xg5F9CIN2GcipzIib5WcrOdMeBNiaVt?usp=sharing)

* The encoder produces two vectors, `mu` and `sigma`, instead of a single latent code, applying the **reparameterization trick** to sample `z` in a differentiable way.
* The loss function combines:
  * **Reconstruction loss**: sum-reduced MSE between input and output.
  * **KL divergence loss**: regularizes the latent distribution towards a standard normal, computed as $0.5 \sum (\sigma^2 + \mu^2 - 1 - \log \sigma^2)$.
* Training tracks both loss components over epochs, and results are visualized through:
  * A training loss curve over iterations.
  * Side-by-side original vs. reconstructed digit comparisons.
  * Direct **generation** of new digits by sampling from the prior and decoding.

## Technologies Used
* **PyTorch:** For building the convolutional encoder/decoder architectures and training via Autograd.
* **Torchvision:** For loading and transforming the MNIST dataset.
* **Scikit-Learn:** For t-SNE dimensionality reduction of the latent space (undercomplete autoencoder).
* **Matplotlib:** For visualizing reconstructions, latent space scatter plots, and generated samples.
* **NumPy:** For numerical operations on latent vectors.

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by clicking the **"Open in Colab"** badge at the top of this page.
The environment will open directly in your browser, pre-configured and ready to use. The file opens in read-only mode: you can run all the cells to view the results, but if you wish to experiment or modify the code, simply click on `File > Save a copy in Drive`.

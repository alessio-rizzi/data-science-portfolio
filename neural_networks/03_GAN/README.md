# Exercise: Generative Adversarial Networks (GAN & WGAN)

## Overview
These two notebooks explore generative modeling on the MNIST dataset through adversarial training, implemented from scratch in PyTorch:

* **04_GAN**: A vanilla convolutional GAN, with a Generator and a Discriminator trained in competition using the standard Binary Cross-Entropy adversarial objective.
* **05_WGAN**: A Wasserstein GAN, replacing the Discriminator with a Critic and the BCE loss with the Wasserstein distance, using weight clipping to enforce the Lipschitz constraint and improve training stability.

## Dataset
Both notebooks use the **MNIST** dataset (28×28 grayscale handwritten digits), downloaded via `torchvision.datasets.MNIST` and loaded through `DataLoader` (batch size 128).

## Process and Methodology

### 1. GAN (04)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hwxhMGR-SJeX_fQ29kIjoaH7Rwjeklt9?usp=sharing)

* **Generator**: maps a 100-dimensional noise vector to a 28×28 image through three `ConvTranspose2d` layers (64→32→16→1 channels), with `BatchNorm` + `ReLU` and a final `Sigmoid` activation.
* **Discriminator**: a convolutional classifier (1→16→32→64→1 channels) using `Conv2d`, `BatchNorm`, and `LeakyReLU`, ending in a `Sigmoid` that outputs the probability of an image being real.
* **Loss**: Binary Cross-Entropy (`BCELoss`) for both networks — the Discriminator is trained to output 1 for real images and 0 for fake ones, while the Generator is trained to make the Discriminator output 1 for its generated images.
* **Optimizer**: Adam with learning rate `0.0002` and betas `(0.5, 0.999)` for both networks.
* **Training loop**: alternates between a Discriminator update (on real + fake batches) and a Generator update (fooling the Discriminator), for 10 epochs.
* **Evaluation**: generated digit samples are plotted in a 4×4 grid and saved to disk after training.

### 2. Wasserstein GAN (05)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uQV5gOlNJ9MZ-TP7iSAK8RTEXaueDvPM?usp=sharing)

* **Generator**: a deeper architecture mapping a 128-dimensional noise vector to a 28×28 image through four `ConvTranspose2d` layers (128→64→32→16→1), with `BatchNorm` + `ReLU` and a final `Sigmoid`.
* **Critic** (replacing the Discriminator): the same convolutional feature extractor as the GAN's Discriminator (1→16→32→64), followed by two fully connected layers (64→256→1) with **no Sigmoid**, producing an unbounded real-valued score instead of a probability.
* **Loss**: the Wasserstein (Earth Mover's) distance, approximated as $\mathbb{E}[C(x_{real})] - \mathbb{E}[C(x_{fake})]$. The Critic is trained to maximize this quantity, while the Generator is trained to maximize the Critic's score on fake images.
* **Lipschitz constraint**: enforced via **weight clipping**, clamping all Critic parameters to $[-0.01, 0.01]$ after every update (an alternative to WGAN-GP's gradient penalty).
* **Training strategy**: the Critic is updated 5 times for every single Generator update, using Adam with a lower learning rate (`0.00005`) for both networks to preserve stability, for 100 epochs.
* **Monitoring**: generated samples, Critic loss, and Generator loss are displayed and saved every 100 steps, allowing visual inspection of image quality as training progresses.

### 3. Key Differences: GAN vs. WGAN
| Aspect | GAN | WGAN |
|---|---|---|
| Second network | Discriminator (probability) | Critic (unbounded score) |
| Loss function | Binary Cross-Entropy | Wasserstein distance |
| Output activation | Sigmoid | None |
| Stability constraint | — | Weight clipping (Lipschitz) |
| Update ratio | 1:1 (D:G) | 5:1 (Critic:G) |

## Technologies Used
* **PyTorch:** For building the Generator/Discriminator (or Critic) architectures and training via Autograd.
* **Torchvision:** For loading and transforming the MNIST dataset.
* **Matplotlib:** For visualizing generated image grids and training progress.
* **NumPy:** For numerical operations on generated samples.

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by clicking the **"Open in Colab"** badge at the top of this page.
The environment will open directly in your browser, pre-configured and ready to use. The file opens in read-only mode: you can run all the cells to view the results, but if you wish to experiment or modify the code, simply click on `File > Save a copy in Drive`.

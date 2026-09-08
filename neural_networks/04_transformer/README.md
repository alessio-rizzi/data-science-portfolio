# Exercise: Transformer for Autoregressive Sequence Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FE27CxKhGsy7pOOwHui87OLwgfS7YE-1?usp=sharing)

## Overview
This notebook implements a **decoder-only Transformer** from scratch in PyTorch to generate text autoregressively, token by token. The model is trained on Dante's *Divina Commedia* (the "comm1-7.txt" corpus) to learn to predict the next token given a context window, and is then used to generate new text from a given prompt.

## Dataset
The training corpus is a plain text file (`comm1-7.txt`) containing the first cantos of the *Divina Commedia*. A custom **Byte-Pair Encoding (BPE)** tokenizer is trained directly on this corpus using the `tokenizers` library, with:
* A vocabulary size of **10,000** tokens.
* **Byte-level** pre-tokenization and decoding, allowing correct handling of Latin/UTF-8 characters.
* Standard special tokens (`[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`).

## Process and Methodology

### 1. Data Preparation
* The corpus is encoded into token IDs using the trained tokenizer.
* A custom `TokenDataset` implements a **sliding window** over the token sequence: for a context window of `SEQ_LEN = 32` tokens, each sample's target is the same window shifted by one position — the standard next-token-prediction setup.

### 2. Positional Encoding
Since PyTorch's Transformer modules don't include positional information by default, a `PositionalEncoding` module is implemented manually using the classic **sinusoidal encoding** from the original Transformer paper (alternating `sin`/`cos` functions of position, added to the token embeddings).

### 3. Model Architecture — `TransformerGenerator`
* **Embedding layer**: maps token IDs to dense vectors of size `d_model = 128`, scaled by $\sqrt{d_{model}}$ before adding positional encodings.
* **Decoder stack**: built from PyTorch's `nn.TransformerDecoderLayer` (4 attention heads, feedforward dimension 512, dropout 0.1), stacked into `num_layers = 3` layers via `nn.TransformerDecoder`.
* **Decoder-only setup**: since there is no separate encoder, the same embedded sequence is passed as both `tgt` and `memory` to the decoder.
* **Causal masking**: a triangular mask (`generate_causal_mask`) prevents each position from attending to future tokens, enforcing autoregressive behavior.
* **Output layer**: a final linear projection maps the decoder output back to vocabulary-sized logits.

### 4. Training
* **Loss**: Cross-Entropy Loss between predicted next-token logits and the shifted target sequence.
* **Optimizer**: Adam with an initial learning rate of `1e-3`.
* **Learning rate schedule**: a custom linear **warmup + decay** schedule (`LambdaLR`) — the learning rate ramps up linearly over the first 10% of training steps, then decays linearly back to zero.
* **Gradient clipping**: gradient norms are clipped to `1.0` to stabilize training.
* The model is trained for `NUM_EPOCHS = 100` epochs.

### 5. Text Generation
The `generate_text` function performs autoregressive sampling:
* Starting from a given prompt, the model repeatedly predicts the next token using only the last `SEQ_LEN` tokens as context.
* Logits are scaled by a **temperature** parameter before applying `softmax`, controlling the randomness of sampling (`torch.multinomial`).
* Several prompts (e.g. *"Nel"*, *"vuolsi"*, *"vate"*, *"spaura"*) are used to qualitatively evaluate the generated continuations.

## Technologies Used
* **PyTorch:** For implementing the Transformer decoder architecture, training loop, and autoregressive generation.
* **Hugging Face `tokenizers` / `transformers`:** For training and applying the custom BPE tokenizer.
* **Math (stdlib):** For computing the sinusoidal positional encoding terms.

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by clicking the **"Open in Colab"** badge at the top of this page.
The environment will open directly in your browser, pre-configured and ready to use. The file opens in read-only mode: you can run all the cells to view the results, but if you wish to experiment or modify the code, simply click on `File > Save a copy in Drive`.

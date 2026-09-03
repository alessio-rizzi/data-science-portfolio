#  Exercise: Least Square

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1REYKVlJrhaJ-76wgI7LS3pnMS6EUs38W?usp=sharing)

## Overview
This notebook demonstrates how to implement a Multiple Linear Regression model from scratch using **PyTorch's Autograd**. Instead of using pre-built machine learning models, it manually applies Gradient Descent to minimize the Least Squares loss function on a real-world housing dataset.

## Dataset
The project uses the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) downloaded dynamically via `kagglehub`. It contains various features about houses (such as area, bedrooms, bathrooms, air conditioning, etc.) and their market prices.

## Process and Methodology

### 1. Mathematical Foundation
The primary goal is to find the optimal weights vector $\mathbf{w}$ that minimizes the Least Squares loss function:
$$f(\mathbf{w}) = \\frac{1}{2} \\| \\mathbf{X}\\mathbf{w} - \\mathbf{y} \\|_2^2$$

### 2. Custom Gradient Descent Optimizer
A custom Python function, `torch_minimize_loss`, is implemented to optimize the weights. It utilizes PyTorch's `requires_grad=True` and `.backward()` methods to automatically compute derivatives (Autograd). The weights are updated in-place during the backward pass using a specified learning rate ($\epsilon$), and the loop stops when the gradient norm falls below a given tolerance threshold ($\delta$).

### 3. Data Preprocessing
To make the real-world dataset compatible with the mathematical model, several crucial preprocessing steps are applied using `pandas` and `scikit-learn`:
* **Binary Encoding:** Categorical text columns containing *"yes"* and *"no"* are mapped to `1` and `0`.
* **One-Hot Encoding:** Applied to the *"furnishingstatus"* column to create dummy variables. The first category is dropped (`drop_first=True`) to avoid perfect multicollinearity (the dummy variable trap) which would break the Least Squares calculation.
* **Standardization:** Features are scaled using `StandardScaler` (Z-score normalization). This step is strictly necessary to ensure Gradient Descent converges properly and to prevent gradients from exploding due to differing feature scales (e.g., house area vs. number of bathrooms).
* **Bias Term Addition:** A column of ones is concatenated to the feature tensor $\mathbf{X}$ to act as the intercept (bias) for the linear equation.
* **Tensor Conversion:** Data matrices are converted to PyTorch `float64` tensors to match the precision of the optimizer.

### 4. Model Evaluation & Visualization
Once the optimal weights are found, the model's performance is visually evaluated. A **True vs. Predicted Prices** scatter plot is generated using `matplotlib`. The plot compares the actual market prices against the model's estimations, with a dashed red diagonal line ($y = x$) representing perfect predictions. 

## Technologies Used
* **PyTorch:** For matrix multiplications and automatic differentiation.
* **Pandas:** For data manipulation, cleaning, and dummy variable creation.
* **Scikit-Learn:** For dataset generation and feature scaling.
* **Matplotlib:** For data visualization.
"""

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by clicking the **"Open in Colab"** badge at the top of this page.
The environment will open directly in your browser, pre-configured and ready to use. The file opens in read-only mode: you can run all the cells to view the results, but if you wish to experiment or modify the code, simply click on `File > Save a copy in Drive`.

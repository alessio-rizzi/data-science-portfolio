# Classifiers Comparison: Decision Trees and K-Nearest Neighbors on the Iris Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Saq5LlJqyCjoVaGoGk7pYW9_wf-jO1Oz?usp=sharing)

This repository contains a Jupyter/Colab notebook comparing and evaluating the behavior of **Decision Trees** and **K-Nearest Neighbors (KNN)** classifiers on the classic **Iris** dataset, with a focus on hyperparameter tuning, weighting strategies, and metric learning.

## 📊 Dataset Overview

The notebook uses the built-in **Iris** dataset from `sklearn.datasets`.

| Dataset | Source | Shape (Rows, Cols) | Target Characteristics | Expected Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Iris** | `sklearn.datasets.load_iris` | `(150, 4)` | 3 balanced classes (Setosa, Versicolor, Virginica), 4 numeric features. | Well-separated classes (especially Setosa), suitable for both tree-based and distance-based classifiers. |

## 🔍 What the Notebook Covers

1. **Decision Tree Classifier** — trained with entropy criterion and evaluated via 5-fold cross-validation.
2. **KNN with uniform weighting** — each neighbor contributes equally to the majority vote; predictions shown in a Pandas DataFrame.
3. **KNN with distance-based weighting** — closer neighbors contribute more to the vote.
4. **Hyperparameter tuning of k** — accuracy scanned across all possible values of k to find the optimal number of neighbors, for both weighting schemes, with comparative plots.
5. **2D visualization of true classes** — scatter plot of the best two features (Petal Length vs Petal Width) colored by ground-truth class.
6. **2D visualization of KNN predictions** — same scatter plot, colored by the model's predicted class, to visually assess misclassifications.
7. **Metric learning extension** — a custom Gaussian-kernel-based distance function is implemented and used inside `KNeighborsClassifier` to evaluate whether it improves accuracy over standard Euclidean distance.

## 🛠️ Technologies Used
* **Python**
* **Scikit-Learn**
* **Pandas & NumPy**
* **Matplotlib**

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by opening the notebook `05_iris_knn.ipynb` directly in **Google Colab** or **Jupyter**. All cells can be run top-to-bottom to reproduce the cross-validation scores, tuning curves, and classification plots. If you wish to experiment or modify the code in Colab, simply click on `File > Save a copy in Drive`.

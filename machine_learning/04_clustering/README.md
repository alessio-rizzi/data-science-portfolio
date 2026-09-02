# Clustering Analysis: K-Means Behavior on Varied Dataset Geometries

This repository contains a Jupyter/Colab notebook comparing and evaluating the behavior of the **K-Means** clustering algorithm across three 2D datasets with distinct geometric properties, shapes, and densities.

## 📌 Project Overview

Clustering algorithms make fundamental assumptions about the structure of data. **K-Means** assumes that clusters are **convex, spherical, and linearly separable**, aiming to minimize the within-cluster sum of squared Euclidean distances (inertia). 

## 📊 Datasets Overview

The notebook loads three separate 2D spatial datasets:

| Dataset | File | Shape (Rows, Cols) | Target Characteristics | Expected Behavior with K-Means |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset 1** | `3-clusters.csv` | `(150, 2)` | Small, compact, 3 linearly separable clusters. | **Optimal**: Successfully identifies the 3 clusters. |
| **Dataset 2** | `dataset-DBSCAN.csv` | `(6118, 2)` | Non-convex, intertwined, continuous density patterns. | **Fails**: Cuts the manifold into Voronoi cells, ignoring connectivity. |
| **Dataset 3** | `CURE-complete.csv` | `(86558, 2)` | Large-scale dataset with complex shapes and varying densities. | **Sub-optimal**: Tends to split large clusters or merge close outliers. |

## 🛠️ Technologies Used
* **Python**
* **Scikit-Learn** 
* **SciPy** 
* **Matplotlib**
* **Pandas & NumPy** 

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by clicking the **"Open in Colab"** badge at the top of this page. 
The environment will open directly in your browser, pre-configured and ready to use. The file opens in read-only mode: you can run all the cells to view the results, but if you wish to experiment or modify the code, simply click on `File > Save a copy in Drive`.
# Exercise: Graph Neural Networks (Graph, Node & Link Prediction)

## Overview
These three notebooks explore **Graph Neural Networks (GNNs)** using `torch_geometric`, applying Graph Convolutional layers (`GCNConv`) to three classic graph learning tasks, each on a different dataset:

* **07_GNN_graph_prediction**: Graph-level classification on the **MUTAG** dataset, predicting whether a molecule is mutagenic.
* **08_GNN_node_prediction**: Node-level classification on the **Cora** citation network, predicting the research topic of each paper.
* **09_GNN_link_prediction**: Edge-level (link) prediction on the **Karate Club** graph, predicting whether a connection between two nodes exists.

## Datasets
* **MUTAG** (graph prediction): a collection of nitroaromatic compounds represented as graphs, where nodes are atoms (7 one-hot features) and edges are bonds (4 one-hot features); the task is to predict each molecule's mutagenicity.
* **Cora** (node prediction): a citation network of 2,708 scientific papers with 1,433-dimensional bag-of-words node features, split into 7 topic classes (e.g. *Neural Networks*, *Reinforcement Learning*, *Genetic Algorithms*), loaded with `NormalizeFeatures` transform.
* **Karate Club** (link prediction): the classic 34-node social network graph, split into train/validation/test edge sets via `RandomLinkSplit`.

## Process and Methodology

### 1. Graph Classification (07)
* Each graph in `MUTAG` is represented as a `torch_geometric.data.Data` object (`x`, `edge_index`, `edge_attr`, `y`); batches of graphs are merged into a single disconnected graph via `DataLoader`, using the `batch` tensor to track node-to-graph membership.
* **Model**: three stacked `GCNConv` layers (num_features → 64 → 64 → 64) with ReLU activations, followed by **global mean pooling** to obtain a graph-level embedding, a dropout layer (p=0.5), and a final linear classifier.
* **Training**: Adam optimizer (lr=0.01), Cross-Entropy loss, run for up to 1000 epochs while tracking loss and test accuracy (smoothed with a rolling average).
* **Evaluation**: per-molecule predictions are inspected individually, comparing predicted class probabilities (via `softmax`) against the true mutagenicity label.

### 2. Node Classification (08)
* **Model**: a 2-layer `GCN` (num_features → 16 hidden channels → num_classes) with ReLU activation and dropout (p=0.5) between layers, operating on the full Cora graph at once.
* **Training**: Adam optimizer (lr=0.01, weight_decay=5e-4), Cross-Entropy loss computed only on the nodes in `train_mask`, trained for 400 epochs.
* **Evaluation**: test accuracy is computed on `test_mask` nodes by comparing predicted vs. true classes.
* **Visualization**: node embeddings are projected to 2D via **t-SNE** and plotted before and after training (and for the test subset only) to visually assess how well the classes separate in embedding space.

### 3. Link Prediction (09)
* The Karate Club graph is split into train/validation/test edge sets using `RandomLinkSplit` (undirected, 20% validation, 10% test edges).
* **Model**: a 2-layer `GCNConv` encoder (in_channels → 16 → 16) with dropout (p=0.8), which produces node embeddings `z`; edge scores are computed as the **dot product** between the embeddings of the two endpoint nodes (`predict`), and scores for all possible edges can be computed at once (`predict_all`).
* **Negative sampling**: since the graph only contains positive (real) edges, `negative_sampling` generates an equal number of fake edges at each training step, which are concatenated with the positive edges and labeled 0/1.
* **Training**: Adam optimizer (lr=0.001, weight_decay=1e-4), Binary Cross-Entropy with logits loss, trained for 800 epochs.
* **Evaluation**: performance is measured with **ROC-AUC** and accuracy on train/validation/test edge splits, since AUC does not require choosing a probability threshold; results are visualized with training curves showing loss and AUC over epochs.

## Technologies Used
* **PyTorch & PyTorch Geometric (`torch_geometric`):** For building GCN-based architectures, handling graph-structured data, batching, and negative sampling.
* **NetworkX:** For visualizing molecular graphs and the Karate Club network.
* **Scikit-Learn:** For t-SNE embedding visualization and ROC-AUC / accuracy evaluation metrics.
* **Matplotlib / Pandas:** For plotting training curves and computing rolling averages.
* **Rich:** For progress-bar tracking during the graph-classification training loop.

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by clicking the **"Open in Colab"** badge at the top of this page.
The environment will open directly in your browser, pre-configured and ready to use. The file opens in read-only mode: you can run all the cells to view the results, but if you wish to experiment or modify the code, simply click on `File > Save a copy in Drive`.
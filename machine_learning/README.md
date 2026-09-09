# Machine Learning Lab Exercises

This directory contains classical machine learning implementations and exploratory data analysis tasks developed using `scikit-learn` and Python.

## Topics Covered

* **01_decision_tree**: Implementation and evaluation of decision tree classifiers, focusing on splitting criteria, depth optimization, and visual tree interpretation.
* **02_ann**: Implementation of feed forward network architecture (Multi-Layer perceptron).
* **03_bayes_classifier**: Probabilistic classification models built on Bayes' theorem, analyzing feature independence and conditional probability thresholds.
* **04_clustering**: Unsupervised learning algorithms, featuring density-based clustering models (like DBSCAN) to discover natural data clusters.
* **05_knn**: K-Nearest Neighbors optimization, featuring distance weighting strategies and metric learning with custom Gaussian kernels.
* **06_svm_scratch**: Implementation of a linear SVM classifier *from scratch*, solving the primal optimization problem directly via `scipy.optimize.minimize` with linear constraints, and visualizing the resulting decision boundary.
* **07_svm_2d**: Exploration of `scikit-learn`'s SVM (`SVC`) with different kernels (polynomial, RBF) on synthetic 2D datasets (circles, moons, and linearly non-separable classification data), visualizing decision surfaces and support vectors.
* **08_svm_kernel**: Design of a custom composite kernel for SVM that combines an RBF kernel for numerical features with a categorical (delta) kernel, applied to a mixed-type synthetic product-success classification dataset.
* **09_ensamble_learning**: Comparison of ensemble methods — Bagging, AdaBoost, and Random Forest — against a single decision tree baseline on the Breast Cancer Wisconsin dataset, evaluating accuracy and ROC AUC on train/test splits and analyzing performance stability via cross-validation as the number of estimators grows.


## Tech Stack & Libraries

* **Language**: Python
* **Core Libraries**: Scikit-Learn, NumPy, Matplotlib, SciPy, Pandas

# Exercise: Ensemble Learning (Bagging, AdaBoost, Random Forest)

## Overview
This notebook explores **ensemble learning methods** for classification, comparing how aggregating multiple weak/base learners affects predictive performance compared to a single decision tree.

* **ensemble-learning**: Trains and compares a single **Decision Tree**, a **Bagging** ensemble, an **AdaBoost** ensemble, and a **Random Forest** on a real-world binary classification dataset, evaluating accuracy and ROC AUC on train/test splits, and studying how performance evolves as the number of estimators grows (via cross-validation).

## Process and Methodology

### Ensemble Methods Comparison

* A shared helper function (`evaluate_model`) computes **accuracy** and **ROC AUC** for any fitted classifier exposing `predict_proba`, used consistently across all models for a fair comparison.

**1. Dataset loading**
* The **Breast Cancer Wisconsin** dataset (`sklearn.datasets.load_breast_cancer`) is used: 30 numerical features describing cell nuclei characteristics, with a binary target (malignant/benign).
* Data is split into train/test (70/30) with `train_test_split`, **stratified** on the target to preserve class balance, and class counts are inspected on the training set.

**2. Classifiers training**
* A **single Decision Tree** (`max_depth=7`) is trained as a baseline for comparison.
* **Bagging** (`BaggingClassifier`): 200 decision trees (`max_depth=7`) trained on bootstrap-resampled subsets of the training data, with `oob_score=True` to obtain an out-of-bag performance estimate without a separate validation set.
* **AdaBoost** (`AdaBoostClassifier`): 200 decision trees (`max_depth=7`) trained sequentially, each one focusing more on the samples misclassified by the previous ones (`learning_rate=1.0`).
* **Random Forest** (`RandomForestClassifier`): 200 trees trained with bootstrap sampling and random feature selection at each split (`max_features="sqrt"`), also with `oob_score=True`.

**3. Model evaluation and comparison**
* Accuracy and ROC AUC are computed for all four models on both the training and test sets.
* Two grouped bar charts (**train vs test**) visualize accuracy and ROC AUC side by side across models, making it easy to spot overfitting (large train/test gaps) versus well-generalizing ensembles.

**4. Comparison as the number of estimators (T) grows**
* A `sweep_n_estimators` helper performs **5-fold cross-validation** for Random Forest, AdaBoost, and Bagging across a range of `n_estimators` values (1, 10, 50, 100, 150, 200, 300, 400, 500), recording the mean and standard deviation of accuracy at each step.
* Results are plotted as **accuracy vs. number of estimators**, with error bars (mean ± std) and a small horizontal jitter between series for readability — illustrating how bagging-based methods (RF, Bagging) tend to plateau quickly, while boosting (AdaBoost) can behave differently as more weak learners are added.

## Technologies Used
* **NumPy / Pandas:** For numerical computation and organizing cross-validation sweep results into a structured DataFrame.
* **Scikit-Learn:** For ensemble models (`BaggingClassifier`, `AdaBoostClassifier`, `RandomForestClassifier`), the base learner (`DecisionTreeClassifier`), dataset loading (`load_breast_cancer`), data splitting (`train_test_split`), cross-validation (`cross_val_score`), and evaluation metrics (`accuracy_score`, `roc_auc_score`, `roc_curve`).
* **Matplotlib:** For visualizing train/test comparison bar charts and accuracy-vs-n_estimators error bar plots.

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by opening the notebook `09_ensamble_learning` directly in **Google Colab** or **Jupyter**. All cells can be run top-to-bottom to reproduce the cross-validation scores, tuning curves, and classification plots. If you wish to experiment or modify the code in Colab, simply click on `File > Save a copy in Drive`.
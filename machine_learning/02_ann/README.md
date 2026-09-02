# 🧠 Exercise: Artificial Neural Network (MLP)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uDCuVv0D1fa0kZ_HB0aLDsMmALFwMi4R?usp=sharing)

## 📝 Description
In this project, contained within the `02_ann.ipynb` notebook, I explore the training and optimization of a **feed-forward Artificial Neural Network** (Multi-Layer Perceptron) using the Iris dataset.

The objective of this exercise is to apply fundamental Deep Learning concepts to tabular data. It involves analyzing the impact of different optimizers (Adam vs. SGD), handling imbalanced classes (via *oversampling*), and preventing overfitting through systematic hyperparameter tuning.

## 🎯 Skills Demonstrated & Workflow
The notebook is structured into the following key steps:
1. **Baseline Model Evaluation**: Initialization of an MLP and evaluation of initial performance via accuracy and *Stratified 5-Fold Cross-Validation*.
2. **Optimizer Analysis & Learning Curves**: Training the model to compare the `adam` and `sgd` solvers, visually analyzing their respective loss curves to understand convergence dynamics.
3. **Multiclass ROC Analysis**: Plotting ROC curves (*One-vs-Rest* approach) and calculating the AUC to measure the network's ability to distinguish between the three classes.
4. **Handling Imbalanced Classes (Oversampling)**: Manually inflating (10x replication) the *Virginica* class in the training set to test the network's behavior on an artificially imbalanced dataset.
5. **Hyperparameter Tuning**: Using `GridSearchCV` to explore various architectures and training parameters in parallel (e.g., `hidden_layer_sizes`, `activation`, `solver`, `alpha`, `learning_rate`) to maximize accuracy[cite: 1].
6. **Final Evaluation**: Predicting on the untouched test set using the optimized model and providing a detailed visualization of the results through a Confusion Matrix and Classification Report.

## 🛠️ Technologies Used
* **Python**
* **Scikit-Learn** (MLPClassifier, GridSearchCV, LearningCurveDisplay, Evaluation Metrics)
* **Matplotlib** (Loss curve plotting, ROC Curves, and Confusion Matrices)
* **NumPy & Pandas** (Data manipulation and result formatting)

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by clicking the **"Open in Colab"** badge at the top of this page. 
The environment will open directly in your browser, pre-configured and ready to use. The file opens in read-only mode: you can run all the cells to view the results, but if you wish to experiment or modify the code, simply click on `File > Save a copy in Drive`.
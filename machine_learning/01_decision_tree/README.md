# 🌳 Exercise: Decision Tree

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gTngdEr8IBNHr0mEMh2bGq7H0qaA6mVq?usp=sharing)

## 📝 Description
In this project, contained within the `01_decision_tree.ipynb` notebook, I explore the training and optimization of a **Decision Tree** model using the renowned Iris dataset.

The main objective goes beyond simple classification to tackle practical Machine Learning challenges, such as **handling imbalanced classes** and **preventing overfitting** through hyperparameter tuning.

## 🎯 Skills Demonstrated & Workflow
The notebook is structured into the following key steps:

1. **Baseline Model Evaluation**: Training with Stratified 5-Fold Cross-Validation and evaluating performance via Confusion Matrix and Classification Report.
2. **Multiclass ROC Analysis**: Target binarization (One-vs-Rest approach) to plot ROC curves and compute the AUC for each class.
3. **Handling Imbalanced Classes (Oversampling)**: Manually inflating (10x replication) the Virginica class in the training set to test the algorithm's behavior in the presence of a heavily predominant class.
4. **Cost-Sensitive Learning**: Using the class_weight parameter to penalize misclassifications of the Virginica class without physically altering the dataset, demonstrating mathematical equivalence (100% match) with the manual inflation method.
5. **Hyperparameter Tuning**: Leveraging GridSearchCV to explore nearly 40,000 hyperparameter combinations in parallel, aiming to reduce overfitting and discover the optimal tree architecture.
6. **Advanced Visualization**: Side-by-side plotting of the final Confusion Matrix and the graphical structure of the optimized decision tree.

## 🛠️ Technologies Used
* **Python**
* **Scikit-Learn** (DecisionTreeClassifier, GridSearchCV, StratifiedKFold, Evaluation Metrics)
* **Matplotlib** (Data Visualization, ROC Curves, Tree Plotting)
* **NumPy** (Array manipulation and sampling)

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by clicking the **"Open in Colab"** badge at the top of this page.
The environment will open directly in your browser, pre-configured and ready to use. The file opens in read-only mode: you can run all the cells to view the results, but if you wish to experiment or modify the code, simply click on `File > Salva una copia in Drive`.

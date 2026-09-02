# 🍉 Exercise: Naive Bayes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wLA6_Xf9pyRDCpd9UPtHGPghBGZCjgtG?usp=sharing)

## 📝 Description
In this project, contained within the `03_naive_bayes_2.ipynb` notebook, I tackle a classification problem using the Watermelon dataset, which features a mix of categorical and continuous variables. The exercise focuses on building a hybrid Naive Bayes model, applying rigorous validation techniques, and using statistical hypothesis testing to rigorously compare model performances.

## 🎯 Skills Demonstrated & Workflow
The notebook is structured into the following key steps:
* **Hybrid Naive Bayes Modeling**: Combining `CategoricalNB` (with Laplace correction) for discrete features and `GaussianNB` for continuous features to handle heterogeneous data.
* **Custom Probability Aggregation**: Calculating and normalizing joint probabilities from both categorical and numerical models to predict the class of a specific test instance.
* **Leave-One-Out Cross-Validation (LOOCV)**: Implementing an exhaustive hold-out validation technique—ideal for small datasets (17 samples)—to evaluate and compare the hybrid Naive Bayes model against a baseline Decision Tree.
* **Performance Visualization**: Generating Confusion Matrices and ROC Curves (with AUC calculation) to graphically compare the predictive power of the two algorithms across all LOOCV iterations.
* **Statistical Hypothesis Testing**: Conducting a Paired T-Test (`scipy.stats.ttest_rel`) to scientifically determine whether the observed differences in accuracy between the Naive Bayes and Decision Tree models are statistically significant.

## 🛠️ Technologies Used
* **Python**
* **Scikit-Learn** (`CategoricalNB`, `GaussianNB`, `DecisionTreeClassifier`, `LeaveOneOut`, Evaluation Metrics)
* **SciPy** (`ttest_rel` for paired statistical testing)
* **Matplotlib** (Confusion Matrices, ROC Curves)
* **Pandas & NumPy** (Data manipulation and custom probability calculations)

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by clicking the **"Open in Colab"** badge at the top of this page. 
The environment will open directly in your browser, pre-configured and ready to use. The file opens in read-only mode: you can run all the cells to view the results, but if you wish to experiment or modify the code, simply click on `File > Save a copy in Drive`.
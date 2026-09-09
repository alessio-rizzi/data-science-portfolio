# Exercise: Support Vector Machines (From Scratch, Kernels & Custom Kernels)

## Overview
These three notebooks explore **Support Vector Machines (SVMs)** from three complementary angles: implementing the underlying optimization problem by hand, visualizing how different kernels shape the decision boundary, and designing a custom kernel for mixed-type data.

* **06_svm_scratch**: Implements a **linear SVM from scratch**, solving the primal optimization problem directly with `scipy.optimize`, without relying on `scikit-learn`'s SVM implementation.
* **07_svm_2d**: Uses `scikit-learn`'s `SVC` with different kernels (polynomial, RBF) to visualize decision surfaces on classic non-linearly-separable 2D toy datasets.
* **08_svm_kernel**: Designs a **custom composite kernel** combining an RBF kernel for numerical features with a categorical (delta) kernel, used to train an SVM on a synthetic mixed-type dataset.

## Process and Methodology

### 1. Linear SVM From Scratch (06)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yW0sG0850PSE2RhQ977KC3wdH6eJcBy7?usp=sharing)

* A synthetic 2D dataset is generated and linearly separated into two classes based on the sign of `x + y - 0.3`.
* The data is converted to **homogeneous coordinates** (a column of ones is appended to `X`) so that the bias term can be folded directly into the weight vector `w`.
* The **primal SVM optimization problem** is formulated as:
$$\min_{\mathbf{w}} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{x}_i \cdot \mathbf{w}) \geq 1 \; \forall i$$
* The objective function (`0.5 * ||w||^2`) is minimized using `scipy.optimize.minimize`, with the classification constraints expressed as a `scipy.optimize.LinearConstraint` (built from the label-weighted feature matrix `A = y * X`, with lower bound 1 and upper bound `+inf`).
* Once solved, the resulting weight vector defines the decision boundary line ($w_1 x + w_2 y + w_3 = 0$), which is plotted together with the original data points.

### 2. Kernel Comparison on 2D Datasets (07)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GPJ7tLJP2LtBVZUVlQJ18D-SfJmjXjiD?usp=sharing)

* A helper function (`plot_svm_decision_surface`) visualizes the SVM's decision function as a filled contour plot over a fine grid, together with the data points and support vectors.
* Three synthetic datasets from `scikit-learn` are used to illustrate different levels of non-linearity:
  * **Circles** (`make_circles`): classified with a **polynomial kernel** (degree 2, low regularization `C=0.01`), showing how a quadratic kernel can separate concentric classes.
  * **Moons** (`make_moons`): classified with an **RBF kernel**, showing how a radial basis function adapts to curved, interleaved class boundaries.
  * **Classification dataset** (`make_classification`): classified with a **polynomial kernel** (degree 3), illustrating a more complex non-linear boundary on data that isn't linearly separable.
* For each dataset, the notebook fits an `SVC` from `scikit-learn` and plots the corresponding decision surface for visual comparison across kernel types.

### 3. Custom Composite Kernel for Mixed Data (08)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tBTNGm5V8t9A3rB1IGmOE3zhIKvURWYr?usp=sharing)

* A synthetic **product-success dataset** is generated with two numerical features (*prezzo* / price and *rating*) and one categorical feature (*categoria*, with 3 possible values), where the success rule differs per category (e.g. low price + high rating for category 0, medium price for category 1, very high rating for category 2).
* Three kernel functions are implemented manually:
  * **RBF kernel** (via `sklearn.metrics.pairwise.rbf_kernel`) applied to the numerical features (price, rating).
  * **Categorical (delta) kernel**: returns 1 if two categorical values match and 0 otherwise.
  * **Composite kernel**: combines the RBF and categorical kernels into a single kernel matrix usable by `SVC`.
* The dataset is split into train/test sets (`train_test_split`, stratified on the target), and an `SVC` is trained using the **custom composite kernel** passed as a callable (`kernel=kernel_func`).
* **Evaluation**: accuracy score and a full `classification_report` are computed on the test set, and results are visualized per category with a 3-panel scatter plot comparing train/test points colored by predicted class.

## Technologies Used
* **NumPy / SciPy:** For numerical computation and, in notebook 06, for solving the constrained optimization problem directly (`scipy.optimize.minimize`, `LinearConstraint`).
* **Scikit-Learn:** For SVM implementations (`SVC`), synthetic dataset generation (`make_circles`, `make_moons`, `make_classification`), kernel functions (`rbf_kernel`), data splitting, and evaluation metrics (`accuracy_score`, `classification_report`).
* **Pandas:** For organizing and inspecting the synthetic product dataset (notebook 08).
* **Matplotlib:** For visualizing datasets, decision boundaries, and decision surfaces.

## 🚀 How to View and Run the Exercise
Open the notebooks (`06_svm_scratch.ipynb`, `07_svm_2d.ipynb`, `08_svm_kernel.ipynb`) in Jupyter, Google Colab, or any compatible environment. All three are self-contained: they generate their own synthetic data on the fly, so no external dataset download is required — simply run all cells in order to reproduce the training and visualizations described above.

## 🚀 How to View and Run the Exercise
The fastest way to explore the code is by opening the notebook `05_iris_knn.ipynb` directly in **Google Colab** or **Jupyter**. All cells can be run top-to-bottom to reproduce the cross-validation scores, tuning curves, and classification plots. If you wish to experiment or modify the code in Colab, simply click on `File > Save a copy in Drive`.
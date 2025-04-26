# Optimization Algorithms for Machine Learning  

![Optimization wallpaper](optimization.gif)

This repository is a comprehensive collection of notebooks and materials exploring **optimization algorithms for machine learning**, covering both theoretical foundations and practical applications. It is structured as a course and lab-based journey, going from core optimization concepts to advanced algorithms tailored to modern machine learning problems.

---

## 🗂 Directory Structure

```
optimisation-algorithms/
├── cours/
│   └── materials_cd/                         # Original course material on coordinate descent
├── cours_coordinate_descent/
│   └── materials_cd/                         # Clean version of coordinate descent course content
├── material_quasi_newton/
│   └── material_quasi_newton/                # Code utilities for quasi-Newton methods (e.g., L-BFGS)
│       ├── optim_utils.py
├── project/                                  # Notebooks and experiments applying optimization methods
```

---

## 📚 Course Overview

The notebooks and materials are structured around the following themes:

### 🔹 1. Introduction to Optimization in Data Science

- **1.1** Machine Learning Optimization Problems + Linear Algebra Recap  
- **1.2** Optimization Problem Properties (Convexity, Smoothness)

### 🔹 2. Smooth Optimization

- **2.1** Gradient Descent & Convergence  
- **3.1** Quadratic Problems and Conjugate Gradient  
- **3.2** Line Search Techniques

### 🔹 3. Non-smooth Optimization

- **4.1** Proximal Operators & Algorithms  
- **4.2** Lab: Lasso & Group Lasso

### 🔹 4. Stochastic Optimization

- **5.1** SGD and Variance Reduction  
- **5.2** Lab: SGD for Logistic Regression

### 🔹 5. Constrained Optimization

- **6.1** Linear Programming (LP), Quadratic Programming (QP), and Mixed Integer Programming

---

### 🔹 6. Coordinate Descent (CD)

Located in `cours_coordinate_descent/materials_cd/`

- **1** Exact Coordinate Descent  
- **2** Coordinate Gradient Descent  
- **3** Proximal Coordinate Descent  
- **4** Applications to ML Estimators (e.g., Lasso, SVM)

---

### 🔹 7. Newton & Quasi-Newton Methods

Located in `material_quasi_newton/`

- **8.1** Second-order optimization: Newton’s method  
- **L-BFGS**: Quasi-Newton method for large-scale optimization  
- Applications: ℓ2 Logistic Regression, Conditional Random Fields

---

### 🔹 8. Beyond Convex Optimization

Located in `fw_non_cvx_autodiff/`

- Nonconvex Regularization  
- Frank-Wolfe Algorithm  
- Difference of Convex (DC) Programming  
- Automatic Differentiation

---

## 🤖 Data and ML Problem Applications

The notebooks also integrate optimization techniques in practical ML scenarios:

### 🧠 1. Data & Preprocessing

- Data visualization and feature analysis  
- Defining your ML problem

### 📊 2. Unsupervised Learning

- Clustering  
- Generative Modeling  
- Dimensionality Reduction  

### 🏷️ 3. Supervised Learning

- Linear Models  
- Decision Trees & Ensemble Methods  
- Bayesian Decision Theory  

### 📈 4. Model Evaluation

- Validation & Performance Metrics  
- Hyperparameter Selection  
- Model Interpretation

---

## 🔍 Notebooks and Code

Every `.ipynb` notebook in this repository contains:

- Interactive implementations of optimization algorithms  
- Visualizations of convergence and error behavior  
- Application to ML tasks like regression and classification  
- Insights into theoretical properties (smoothness, convexity, duality)

Some files also include useful utilities like:
- `optim_utils.py`: Common optimization routines for experiments.

---

## 🧹 Notes

- The `__MACOSX/` and hidden `._*` files are system-generated and can be safely ignored or deleted.
- Python `.py` and notebook files are tested with Python 3.8+ and standard ML libraries like NumPy, SciPy, scikit-learn, and Matplotlib.

---

## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/optimisation-algorithms.git
   cd optimisation-algorithms
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # if available
   ```
3. Launch notebooks:
   ```bash
   jupyter notebook
   ```

---

## 👨‍🔬 Authors & Acknowledgements

Developed as part of a data science optimization course.  
Special thanks to contributors, researchers, and instructors whose materials shaped this work.


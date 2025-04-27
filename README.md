# Optimization Algorithms for Machine Learning  

![Optimization wallpaper](optimization.gif)

This repository is a comprehensive collection of notebooks and materials exploring **optimization algorithms for machine learning**, covering both theoretical foundations and practical applications. It is structured as a course and lab-based journey, going from core optimization concepts to advanced algorithms tailored to modern machine learning problems.

---

## 🗂 Directory Structure

```
├── cours                                     # Original course material on coordinate descent
│   ├── 01_intro_optim_ml.pdf
│   ├── 02_grad_desc.pdf
│   ├── 03_MLinpractice.pdf
│   ├── 04_nonsmooth.pdf
│   ├── 05_stochastic.pdf
│   ├── 06_constrained_std.pdf
│   ├── 3_prox.pdf
│   ├── ex_intro_gd.pdf
│   ├── gradient_descent.ipynb
│   └── materials_cd
│       └── materials_cd                      # Clean version of coordinate descent course content
│           ├── coordinate_descent.ipynb
│           ├── Lab_cd.ipynb
│           ├── notes_cd.pdf
│           └── slides_cd.pdf
├── cours_coordinate_descent                     
│   └── materials_cd
│       ├── coordinate_descent.ipynb
│       ├── Lab_cd.ipynb
│       ├── notes_cd.pdf
│       └── slides_cd.pdf
├── fw_non_cvx_autodiff
│   ├── lecture_fw_non_cvx_autodiff
│   │   ├── 07-line_search_interactive.ipynb
│   │   ├── 08-Line_search_method.ipynb
│   │   ├── adaptive_lasso.ipynb
│   │   ├── automatic_differentiation.ipynb
│   │   ├── coordinate_descent_non_cvx.ipynb
│   │   ├── frank_wolfe.ipynb
│   │   ├── movies
│   │   │   ├── 0_l22_film.mp4
│   │   │   ├── 1_l1_film.mp4
│   │   │   ├── 2_enet_film.mp4
│   │   │   ├── 3_l0_film.mp4
│   │   │   ├── 4_sqrt_film.mp4
│   │   │   ├── 5_log_film.mp4
│   │   │   ├── 6_mcp_film.mp4
│   │   │   └── 7_scad_film.mp4
│   │   ├── movies_prox.py
│   │   ├── optim_utils.py
│   │   ├── prox_and_pen_plotting.py
│   │   ├── prox_collection.py
│   │   ├── slides_autodiff.pdf
│   │   ├── slides_fw.pdf
│   │   ├── slides_linear_search.pdf
│   │   └── slides_non_cvx.pdf
├── gradient_descent.ipynb
├── lab1_chen_bryan_and_devilder_alice.ipynb
├── lab2_chen_bryan_and_devilder_alice.ipynb
├── lab3_chen_bryan_and_devilder_alice.ipynb
├── material_quasi_newton                       # Code utilities for quasi-Newton methods (e.g., L-BFGS)
│   └── material_quasi_newton
│       ├── Newton_1d.ipynb
│       ├── Newton_variable_metric.ipynb
│       ├── notes_quasi_newton.pdf
│       ├── optim_utils.py
│       ├── slides_quasi_newton.pdf
│       └── tp_newton_chen_bryan_and_devilder_alice.ipynb
├── optimization.gif
├── project                                     # Notebooks and experiments applying optimization methods
│   ├── NOxEmissions.csv
│   └── project_chen_bryan_and_devilder_alice.ipynb

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


# Optimization Methods for Machine Learning Projects

## Abstract:
This repository collects two projects from the Optimization Methods for Machine Learning course at Sapienza University.
The goal was to implement classical ML models from scratch without using high-level training libraries.
Implemented models include MLP, RBF, and SVMs with custom optimization routines (gradient descent, decomposition, SMO).
Experiments achieved >99% accuracy on MNIST and low MSE in regression tasks.
The work demonstrates strong foundations in ML theory, algorithmic implementation, and performance evaluation.

---

## Project Overview

This repository contains two separate university projects, each located in its own folder.These projects were part of the *Optimization Methods for Machine Learning* course at Sapienza University of Rome. The main objective was to **manually implement and optimize** classical ML models without relying on external training libraries.

Key highlights:
   - Implemented **Multilayer Perceptron (MLP)** and **Radial Basis Function (RBF)** networks for regression.
   - Developed a **Support Vector Machine (SVM)** classifier for binary and multiclass problems (MNIST).
   - Designed all optimization routines manually (gradient descent, decomposition, Most Violating Pair algorithm).

Each project folder contains:
   - The **project report** detailing the technical implementation.
   - The **project assignment** for reference.
   - The **code**, organized into subfolders by assignment question.

Each question folder includes:
   - A `functions` file with the necessary helper functions.
   - A `run` file to execute the experiments.

To run any experiment, execute the `run` file in the corresponding folder. All helper functions are self-contained within the `functions` files.

---

## Dependencies

The projects require the following external Python libraries:
   - **numpy** – arrays, and numerical operations  
   - **pandas** – dataset loading and manipulation
   - **matplotlib** – 2D and 3D visualizations
   - **scipy** – optimization (`scipy.optimize.minimize`)  
   - **scikit-learn** – preprocessing, splitting, and metrics  
   - **cvxopt** – quadratic programming solver for the SVM formulation  

---

**RESULTS**

### Regression Models

| Model              | Final Validation MSE | Optimization Time |
|--------------------|----------------------|-------------------|
| Full MLP           | 2.10 × 10⁻⁴          | 1.35 s            |
| Full RBF           | 3.28 × 10⁻⁴          | 0.23 s            |
| Extreme MLP        | 4.18 × 10⁻⁴          | 0.09 s            |
| Unsupervised RBF   | 4.72 × 10⁻³          | 0.001 s           |
| Block Decomposition| 2.76 × 10⁻⁴          | 1.62 s            |

### SVM Models
| Model                              | Test Accuracy | CPU Time |
|------------------------------------|---------------|----------|
| SVM (cvxopt)                       | 99.25%        | 5.78 s   |
| SVM Decomposition Algorithm (q=90) | 99.25%        | 0.61 s   |
| SVM SMO-MVP (q=2)                  | 99.25%        | 0.79 s   |
| Multiclass SVM (OAO, OAA)          | 98.33%        | 3.11 s   |

---

Detailed explanations and analyses are provided in the **project reports** included in each folder.
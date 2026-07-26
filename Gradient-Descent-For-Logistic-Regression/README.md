# Gradient Descent for Logistic Regression

This project implements **Logistic Regression** from scratch using **NumPy** and **Gradient Descent**. The model learns the optimal parameters by minimizing the Binary Cross-Entropy (Log Loss) cost function and visualizes the training process.

The project demonstrates the complete workflow of Logistic Regression, including cost computation, gradient calculation, parameter optimization, and decision boundary visualization.

---

## Features

- Binary classification dataset
- Logistic Regression implementation from scratch
- Sigmoid activation function
- Binary Cross-Entropy (Log Loss)
- Gradient Descent optimization
- Cost convergence visualization
- Learned decision boundary visualization
- Built using only NumPy and Matplotlib

---

## Dataset

| Feature 1 | Feature 2 | Class |
|----------:|----------:|------:|
|0.5|1.5|0|
|1.0|1.0|0|
|1.5|0.5|0|
|3.0|0.5|1|
|2.0|2.0|1|
|1.0|2.5|1|

---

## Mathematical Background

### Sigmoid Function

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

where

\[
z=w^Tx+b
\]

---

### Binary Cross-Entropy Loss

\[
J(w,b)=
-\frac1m
\sum
\left[
y\log(\hat y)
+
(1-y)\log(1-\hat y)
\right]
\]

---

### Gradient Descent Update

The parameters are updated iteratively using

\[
w=w-\alpha\frac{\partial J}{\partial w}
\]

\[
b=b-\alpha\frac{\partial J}{\partial b}
\]

where

- α = Learning Rate
- J = Binary Cross-Entropy Loss

---

## Output

The program generates

- Binary classification dataset
- Cost Function vs Iterations
- Learned decision boundary

---

## Installation

Clone the repository

```bash
git clone https://github.com/Ayubumair/Machine-Learning.git
```

Navigate to the project

```bash
cd Machine-Learning/Gradient-Descent-for-Logistic-Regression
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the program

```bash
python gradient_descent_logistic_regression.py
```

---

## Dependencies

- Python 3.x
- NumPy
- Matplotlib

---

## Project Structure

```
Gradient-Descent-for-Logistic-Regression/
│
├── gradient_descent_logistic_regression.py
├── README.md
├── requirements.txt
├── LICENSE
└── images/
```

---

## Learning Objectives

This project demonstrates

- Logistic Regression
- Sigmoid Function
- Binary Cross-Entropy Loss
- Gradient Descent
- Decision Boundary
- Binary Classification

---

## Future Improvements

- Vectorize the cost function completely
- Animate Gradient Descent
- Interactive learning rate slider
- Contour plot of the loss surface
- Add prediction accuracy
- Add confusion matrix
- Add precision, recall, and F1-score
- Train on a real-world dataset

---

## Author

**A. Umair**

Undergraduate

Department of Electronic and Telecommunication Engineering

University of Moratuwa

---

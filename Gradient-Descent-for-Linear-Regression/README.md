# Gradient Descent for Linear Regression

This project demonstrates how **Gradient Descent** is used to train a simple Linear Regression model using **NumPy** and **Matplotlib**.

Instead of manually choosing the slope and intercept, the algorithm automatically learns the optimal values by minimizing the Mean Squared Error (MSE) cost function.

---

## Features

- Linear Regression using Gradient Descent
- Cost Function implementation
- Gradient calculation
- Automatic parameter optimization
- Cost vs Iteration visualization
- Weight (w) vs Iteration visualization
- Bias (b) vs Iteration visualization

---

## Dataset

| Size (1000 sqft) | Price (1000 LKR) |
|-----------------:|-----------------:|
| 1.0 | 250 |
| 1.7 | 300 |
| 2.0 | 480 |
| 2.5 | 430 |
| 3.0 | 630 |
| 3.2 | 730 |

---

## Algorithm

The model follows the equation

```
y = wx + b
```

Gradient Descent updates the parameters using:

```
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
```

where

- α = Learning Rate
- J = Cost Function

The algorithm repeatedly updates **w** and **b** until the cost reaches a minimum.

---

## Output

The program displays

- Cost vs Iteration
- Cost convergence
- Weight vs Iteration
- Bias vs Iteration

These plots illustrate how Gradient Descent converges to the optimal solution.

---

## Installation

Clone the repository

```bash
git clone https://github.com/AyubUmair/Machine-Learning.git
```

Go to the project

```bash
cd Machine-Learning/Gradient-Descent-for-Linear-Regression
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python gradient_descent.py
```

---

## Dependencies

- Python 3.x
- NumPy
- Matplotlib

Install manually

```bash
pip install numpy matplotlib
```

---

## Project Structure

```
Gradient-Descent-for-Linear-Regression/
│
├── gradient_descent.py
├── README.md
├── requirements.txt
├── LICENSE
└── images/
```

---

## Future Improvements

- Adjustable learning rate slider
- Animated Gradient Descent
- Cost Surface Visualization
- Contour Plot
- Multiple Linear Regression
- Stochastic Gradient Descent
- Mini-Batch Gradient Descent

---

## Author

A. Umair

University of Moratuwa

---

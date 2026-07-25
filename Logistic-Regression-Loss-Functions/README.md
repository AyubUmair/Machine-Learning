# Logistic Regression Cost Functions

This project demonstrates why **Mean Squared Error (MSE)** is not an ideal loss function for Logistic Regression and why **Binary Cross-Entropy (Log Loss)** is preferred.

Using a small binary classification dataset, the program compares the two cost functions and visualizes the Logistic Loss Surface in three dimensions.

---

## Features

- Sigmoid activation function
- Mean Squared Error (MSE) for Logistic Regression
- Binary Cross-Entropy (Log Loss)
- Cost vs Weight visualization
- 3D Logistic Loss Surface
- 3D Logarithmic Loss Surface

---

## Dataset

| x | y |
|--:|--:|
|0|0|
|1|0|
|2|0|
|3|1|
|4|1|
|5|1|

---

## Mathematical Background

### Sigmoid Function

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

where

\[
z=wx+b
\]

---

### Mean Squared Error

\[
J(w)=\frac1{2m}\sum(\hat y-y)^2
\]

Although MSE works well for Linear Regression, it creates a **non-convex optimization problem** when combined with the sigmoid function.

---

### Binary Cross-Entropy

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

Binary Cross-Entropy is convex for Logistic Regression, making Gradient Descent more reliable.

---

## Visualizations

The program generates:

- MSE Cost vs Weight
- Logistic Cost vs Weight
- 3D Logistic Loss Surface
- 3D Logarithmic Loss Surface

---

## Installation

Clone the repository

```bash
git clone https://github.com/AyubUmair5/Machine-Learning.git
```

Move into this project

```bash
cd Machine-Learning/Logistic-Regression-Cost-Functions
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python logistic_cost_functions.py
```

---

## Dependencies

- Python 3.x
- NumPy
- Matplotlib

---

## Project Structure

```
Logistic-Regression-Cost-Functions/
│
├── logistic_cost_functions.py
├── README.md
├── requirements.txt
├── LICENSE
└── images/
```

---

## Future Improvements

- Interactive sliders for **w** and **b**
- Decision Boundary visualization
- Gradient Descent animation
- Contour Plot of the loss surface
- Real-world classification dataset

---

## Author

A. Umair

University of Moratuwa

---

## License

MIT License
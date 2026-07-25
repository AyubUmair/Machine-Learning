# Interactive Linear Regression Visualizer

An interactive implementation of **Linear Regression** using **NumPy** and **Matplotlib**.

This project demonstrates how changing the slope (**w**) affects the regression line and the cost function in real time using a slider.

---

## Features

- Plot training data
- Linear regression prediction
- Cost function visualization
- Interactive weight (w) slider
- Reset button
- Built entirely with NumPy and Matplotlib

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

## How It Works

The model follows the linear regression equation

```
y = wx + b
```

where

- **w** = slope (weight)
- **b** = intercept

The slider allows you to modify **w** interactively and observe:

- the regression line
- the corresponding cost value

The cost is computed using Mean Squared Error (MSE):

```
J(w,b) = (1 / 2m) Σ (f(x) − y)²
```

where

- **m** = number of training examples
- **f(x)** = predicted value

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Linear-Regression-Visualizer.git
```

Move into the project directory:

```bash
cd Linear-Regression-Visualizer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python linear_regression.py
```

---

## Dependencies

- Python 3.x
- NumPy
- Matplotlib

Install manually if needed:

```bash
pip install numpy matplotlib
```

---

## Project Structure

```
Linear-Regression-Visualizer/
│
├── linear_regression.py
├── README.md
├── requirements.txt
├── images/
│   ├── scatter_plot.png
│   ├── regression_line.png
│   └── cost_curve.png
└── LICENSE
```

---

## Learning Objectives

This project demonstrates:

- Linear Regression
- Cost Function
- Mean Squared Error
- Data Visualization
- Interactive Widgets in Matplotlib
- NumPy Vectorized Operations

---

## Future Improvements

- Gradient Descent implementation
- Adjustable bias (b) slider
- 3D Cost Surface
- Contour Plot
- Multiple Linear Regression
- Polynomial Regression
- Real-world housing dataset support

---

## Author

A. Umair

University of Moratuwa

---


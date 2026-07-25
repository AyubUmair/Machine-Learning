# Logistic Regression Visualization

This project demonstrates the fundamental concepts of **Logistic Regression** using **NumPy** and **Matplotlib**. It provides intuitive visualizations of binary classification datasets, the sigmoid activation function, and decision boundaries in one, two, and three dimensions.

The goal of this project is to help beginners understand how Logistic Regression separates different classes before implementing the complete training algorithm.

---

## Features

- Binary classification dataset visualization
- One-dimensional (1D) classification
- Two-dimensional (2D) classification
- Three-dimensional (3D) classification
- Sigmoid activation function visualization
- 2D decision boundary visualization
- 3D decision boundary plane visualization
- Built using only NumPy and Matplotlib

---

## Project Overview

### 1. One-Dimensional Classification

Visualizes a simple binary classification dataset and shows how the sigmoid function converts a linear model into probabilities.

---

### 2. Two-Dimensional Classification

Displays two-dimensional training data and the corresponding decision boundary represented by

\[
w_0x_0 + w_1x_1 + b = 0
\]

which separates the two classes.

---

### 3. Sigmoid Function

The sigmoid function is defined as

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

where

\[
z=wx+b
\]

It maps any real-valued input to a probability between **0** and **1**.

---

### 4. Three-Dimensional Decision Boundary

The project also demonstrates Logistic Regression with three input features.

The decision boundary becomes a plane:

\[
w_0x_0+w_1x_1+w_2x_2+b=0
\]

which separates the two classes in 3D space.

---

## Output

The program generates the following visualizations:

- One-dimensional classification plot
- Two-dimensional classification plot
- Sigmoid activation function
- Two-dimensional decision boundary
- Three-dimensional decision boundary plane

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Machine-Learning.git
```

Navigate to the project folder:

```bash
cd Machine-Learning/Logistic-Regression-Visualization
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python logistic_regression_visualization.py
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
Logistic-Regression-Visualization/
│
├── logistic_regression_visualization.py
├── README.md
├── requirements.txt
├── LICENSE
└── images/
    ├── classification_1d.png
    ├── classification_2d.png
    ├── sigmoid_function.png
    ├── decision_boundary_2d.png
    └── decision_boundary_3d.png
```

---

## Learning Objectives

This project demonstrates:

- Binary Classification
- Logistic Regression Fundamentals
- Sigmoid Activation Function
- Decision Boundary
- Multi-dimensional Data Visualization
- Probability Interpretation

---

## Future Improvements

- Implement Binary Cross-Entropy Cost Function
- Add Gradient Descent for Logistic Regression
- Train the model instead of using manually selected parameters
- Interactive sliders for weights and bias
- Decision boundary animation
- Real-world classification datasets
- Confusion matrix and prediction visualization

---

## Author

**A. Umair**

Undergraduate  
Department of Electronic and Telecommunication Engineering  
University of Moratuwa

---

## License

This project is licensed under the MIT License.
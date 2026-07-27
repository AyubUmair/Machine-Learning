# Logistic Regression with Gradient Descent and Regularization

This project implements Logistic Regression from scratch using NumPy. It demonstrates binary classification using gradient descent and extends the model with L2 regularization and polynomial feature mapping to classify non-linearly separable data.

No machine learning libraries such as scikit-learn are used for training.

---

## Features

- Logistic Regression implemented from scratch
- Sigmoid activation function
- Binary Cross-Entropy cost function
- Gradient Descent optimization
- Decision Boundary visualization
- L2 Regularization
- Polynomial Feature Mapping (Degree = 6)
- Non-linear Decision Boundary visualization
- NumPy and Matplotlib implementation only

---

## Datasets

### 1. University Admission Dataset

Predicts whether a student is admitted based on two exam scores.

Features:

- Exam 1 Score
- Exam 2 Score

Output:

- Admitted
- Not Admitted

---

### 2. Microchip Quality Assurance Dataset

Predicts whether a manufactured microchip passes quality assurance tests.

Features:

- Microchip Test 1
- Microchip Test 2

Output:

- Accepted
- Rejected

This dataset demonstrates why feature mapping and regularization are necessary for non-linear classification.

---

## Visualizations

The project generates:

- Training data visualization
- Decision boundary for logistic regression
- Feature mapping demonstration
- Regularized non-linear decision boundary

---

## Installation

Clone the repository

```bash
git clone https://github.com/AyubUmair/Machine-Learning.git
```

Move into the project

```bash
cd Machine-Learning/Logistic-Regression-with-Regularization
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python logistic_regression_regularization.py
```

---

## Project Structure

```
Logistic-Regression-with-Regularization/
│
├── logistic_regression_regularization.py
├── README.md
├── requirements.txt
└── images/
    ├── admission_dataset.jpg
    ├── linear_decision_boundary.png
    ├── microchip_dataset.jpg
    └── regularized_decision_boundary.png
```

---

## Learning Objectives

This project demonstrates:

- Logistic Regression
- Binary Classification
- Sigmoid Function
- Binary Cross-Entropy Loss
- Gradient Descent
- Decision Boundaries
- L2 Regularization
- Feature Engineering
- Polynomial Feature Mapping
- Non-linear Classification

---

## Dependencies

- Python 3.x
- NumPy
- Matplotlib

---

## Future Improvements

- Vectorize all computations for better performance
- Plot Cost vs Iteration
- Plot Accuracy vs Iteration
- Add prediction probabilities
- Add model evaluation metrics (Accuracy, Precision, Recall, F1-score)
- Implement mini-batch gradient descent
- Compare results with scikit-learn

---

## Author

**A. Umair**

Undergraduate

Department of Electronic and Telecommunication Engineering

University of Moratuwa

---

## License

This project is licensed under the MIT License.
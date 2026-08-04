"""
Gradient Descent for Logistic Regression

This project implements Logistic Regression from scratch using
Gradient Descent and Binary Cross-Entropy Loss.

Author: A. Umair
University of Moratuwa
"""

import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

pos = y_train == 1
neg = y_train == 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_logistic_cost(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float
) -> float:
    m = x.shape[0]
    cost = 0
    for i in range(m):
        fw_x = np.dot(w, x[i]) + b
        probabilities = sigmoid(fw_x)
        cost += (-y[i]*np.log(probabilities) - (1-y[i])*np.log(1-probabilities))
    return cost / m

def compute_logistic_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    w_in: np.ndarray,
    b_in: float,
    alpha: float,
    num_iterations: int,
    compute_cost: callable
) -> tuple[np.ndarray, float, list]:
    m = x.shape[0]
    w = w_in
    b = b_in
    j_history = []
    for i in range(num_iterations):
        z = np.dot(x, w) + b
        g_z = sigmoid(z)
        error = g_z - y
        dw = (1/m) * np.dot(x.T, error)
        db = (1/m) * np.sum(error)
        j_history.append(compute_cost(x, y, w, b))
        w -= alpha * dw
        b -= alpha * db


    return w, b, j_history

def main():
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(X_train[pos, 0], X_train[pos, 1], c='r', marker='x')
    ax.scatter(X_train[neg, 0], X_train[neg, 1], c='b', marker='o')
    ax.set_xlabel("x_0")
    ax.set_ylabel("x_1")
    plt.title('Dataset Visualization')
    plt.savefig("Gradient-Descent-For-Logistic-Regression/images/dataset_visualization.png",dpi= 300)
    plt.show()


    w_tmp  = np.zeros_like(X_train[0])
    b_tmp  = 0.
    alph = 0.1
    iters = 10000

    w_out, b_out, J_history = compute_logistic_gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters, compute_logistic_cost) 
    print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

    fig, ax = plt.subplots(1,1,figsize=(5,4))
    ax.plot(range(iters), J_history, c='b', linewidth=2)
    ax.set_xlabel("Number of Iterations")   
    ax.set_ylabel("Cost Function Value")
    ax.set_title("Cost Function vs Number of Iterations")
    plt.savefig("Gradient-Descent-For-Logistic-Regression/images/cost_function_vs_iterations.png", dpi=300)
    plt.show()  

    fig, ax = plt.subplots(1,1,figsize=(5,4))
    ax.scatter(X_train[pos, 0], X_train[pos, 1], c='r', marker='x')
    ax.scatter(X_train[neg, 0], X_train[neg, 1], c='b', marker='o')
    ax.set_xlabel("x_0")
    ax.set_ylabel("x_1")
    plt.title('Logistic Regression Decision Boundary')


    # Plot the decision boundary
    x0 = -b_out/w_out[0]
    x1 = -b_out/w_out[1]
    ax.plot([0,x0],[x1,0], c='g', linewidth=2)
    ax.fill_between([0,x0],[x1,0], alpha=0.2, color='b')
    plt.savefig("Gradient-Descent-For-Logistic-Regression/images/logistic_regression_decision_boundary.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

"""
Logistic Regression Cost Functions

This project compares Mean Squared Error and Binary Cross-Entropy
for Logistic Regression and visualizes the corresponding loss surface.

Author: A. Umair
"""

import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

def sigmoid(z):
    return 1/ (1+np.exp(-z))

# Squared Error Cost Function. This will not fit correctly for logistic regression
def compute_cost(x:np.ndarray,
                 y: np.ndarray,
                 w_range: np.ndarray,
                 b: float):
    m = x.shape[0]
    cost = []
    for w in w_range:
        fw_x = w*x +b
        probabilities = sigmoid(fw_x)
        loss = np.sum((probabilities - y)**2) / (2*m)
        cost.append(loss)
    return np.array(cost)

# Binary Cross Entropy (Logistic Cost) Function.
def compute_logistic_cost(x,y,w_range,b):
    m = x.shape[0]
    cost = []
    for w in w_range:
        fw_x = w * x + b
        probabilities = sigmoid(fw_x)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15) # Prevent log(0), which would result in negative infinity
        loss = np.mean((-y*np.log(probabilities) - (1-y)*np.log(1-probabilities)))
        cost.append(loss)
    return np.array(cost)

def compute_cost_grid(x, y, W, B):  # Computes the cost for a grid of weight and bias values.
    m = x.shape[0]
    # W and B are 2D meshgrid matrices
    cost = np.zeros(W.shape)


    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w = W[i, j]
            b = B[i, j]

            z = w * x + b
            probabilities = sigmoid(z)
            probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)  # To Avoid log(0)

            cost[i, j] = np.sum(-y * np.log(probabilities) - (1 - y) * np.log(1 - probabilities)) / m

    return cost

def main():
    w_range = np.linspace(-10,10,100)
    b = 0
    cost_array = compute_cost(x_train, y_train, w_range,b)
    plt.plot(w_range, cost_array,linewidth=4,color='b')
    plt.grid(True)
    plt.xlabel('W values')
    plt.ylabel('Cost')
    plt.title("Mean Squared Error Cost")
    plt.tight_layout()
    plt.savefig("Logistic-Regression-Loss-Functions/images/W vs MSE Cost.png",dpi = 300)
    plt.show()


    cost_array = compute_logistic_cost(x_train, y_train, w_range,b)
    plt.plot(w_range, cost_array,linewidth=2,color='r')
    plt.grid(True)
    plt.xlabel('W values')
    plt.ylabel('Cost')
    plt.title("Binary Cross Entropy Cost")
    plt.tight_layout()
    plt.savefig("Logistic-Regression-Loss-Functions/images/w vs Binary Cross Entropy Cost.png",dpi = 300)
    plt.show()


    w_vals = np.linspace(-5, 5, 100)
    b_vals = np.linspace(-10, 5, 100)

    W, B = np.meshgrid(w_vals, b_vals)
    J_values = compute_cost_grid(x_train, y_train, W, B)

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5),subplot_kw={'projection': '3d'})
    surf = ax1.plot_surface(W, B, J_values, cmap='viridis', edgecolor='none', alpha=0.8)
    surf = ax2.plot_surface(W, B, np.log(J_values), cmap='viridis', edgecolor='none', alpha=0.8)

    ax1.set_xlabel('Weight ($w$)', fontsize=11)
    ax1.set_ylabel('Bias ($b$)', fontsize=11)
    ax1.set_zlabel('Cost ($J$)', fontsize=11)
    ax1.set_title('3D Logistic Loss Surface $J(w, b)$', fontsize=13)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    ax1.view_init(elev=30, azim=135)

    ax2.set_xlabel('Weight ($w$)', fontsize=11)
    ax2.set_ylabel('Bias ($b$)', fontsize=11)
    ax2.set_zlabel('Cost ($J$)', fontsize=11)
    ax2.set_title('3D log(Logistic Loss Surface) $J(w, b)$', fontsize=13)
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
    ax2.view_init(elev=30, azim=135)
    plt.tight_layout()
    plt.savefig("Logistic-Regression-Loss-Functions/images/3dlogistic loss surface" ,dpi = 300)

    plt.show()

if __name__ == "__main__":
    main()
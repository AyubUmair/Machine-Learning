"""
Gradient Descent for Linear Regression

This program demonstrates how Gradient Descent minimizes the
Mean Squared Error (MSE) cost function to learn the optimal
weight (w) and bias (b) for a simple linear regression model.

Author: A. Umair
University of Moratuwa
"""

import numpy as np
import matplotlib.pyplot as plt
import math

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])
b = 100

def compute_gradiant(x,y,w,b):   # Gradient function
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    derivative = []
    f_wb = w*x + b
    dj_dw += np.sum((f_wb-y)*x) / m
    dj_db += np.sum((f_wb-y)) / m
    derivative.append(dj_dw)
    derivative.append(dj_db)
    return np.array(derivative)
"""
def compute_cost(x,y,w_range,b):
    m = x.shape[0]
    cost_sum = []
    for w in w_range:
        f_wb = w*x+b
        total_cost = np.sum((f_wb-y)**2) /(2*m)
        cost_sum.append(total_cost)
    return np.array(cost_sum) """

def compute_cost(
    x: np.ndarray,
    y: np.ndarray,
    w: float,
    b: float
) -> float:

    m = x.shape[0]
    f_wb = w * x + b
    total_cost = np.sum((f_wb - y) ** 2) / (2 * m)
    return total_cost

def gradient_descent(x,y,w_init,b_init,alpha,no_of_iterations,cost_function,gradient_function):
    m = x.shape[0]

    w = w_init
    b = b_init

    cost_array = []
    p_history = []
    w_history = []
    b_history = []

    for i in range(no_of_iterations):
        dj_dw = gradient_function(x, y, w, b)[0]
        dj_db = gradient_function(x, y, w, b)[1]

        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        cost_array.append(cost_function(x,y,w,b))
        p_history.append([w,b])
        w_history.append(w)
        b_history.append(b)
        if i % math.ceil(no_of_iterations / 10) == 0:
            print(f"Iteration {i:4}: Cost {cost_array[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w,b, cost_array,p_history,w_history,b_history

def main():
    w_init = 0
    b_init = 0
    learning_rate = 0.1  # Learning rate
    iterations = 10000

    w_final, b_final, cost_history, p_history, w_history, b_history = gradient_descent(x_train,
                                                                                       y_train,
                                                                                       w_init,
                                                                                       b_init,
                                                                                       learning_rate,
                                                                                       iterations,
                                                                                       compute_cost,
                                                                                       compute_gradiant)
    print(f"w_final : {w_final}, b_final : {b_final}")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Gradient Descent Convergence", fontsize=16)

    ax1.plot(cost_history[:100])
    ax1.set_title('Cost vs Iteration (start)')
    ax1.set_xlabel('Iteration');
    ax1.set_ylabel('Cost')

    ax2.plot(1000 + np.arange(len(cost_history[1000:])), cost_history[1000:])
    ax2.set_title('Cost vs Iteration (end)')
    ax2.set_xlabel('Iteration');
    ax2.set_ylabel('Cost')

    ax3.plot(w_history, c='orange', linewidth=2, label='w value')
    ax3.set_title('W Value vs iteration')
    ax3.set_xlabel('iteration');
    ax3.set_ylabel('w value')

    ax4.plot(b_history, c='green', linewidth=2, label='w value')
    ax4.set_title('B Value vs iteration')
    ax4.set_xlabel('iteration');
    ax4.set_ylabel('B value')

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    #os.makedirs("images", exist_ok=True)
    plt.savefig("Gradient-Descent-for-Linear-Regression/images/gradient_descent_results.png", dpi=300)
    plt.show()

    print(f"\nFinal Equation:")
    print(f"y = {w_final:.3f}x + {b_final:.3f}")

    plt.figure(figsize=(6, 4))
    plt.scatter(x_train, y_train, color='red', label='Training Data')

    predictions = w_final * x_train + b_final

    plt.plot(x_train, predictions,
             color='blue',
             linewidth=2,
             label='Regression Line')

    plt.title("Linear Regression using Gradient Descent")
    plt.xlabel("Size (1000 sqft)")
    plt.ylabel("Price (1000 LKR)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Gradient-Descent-for-Linear-Regression/images/linearRegressionusingGradientDescent.png", dpi=300)
    plt.show()
if __name__ == "__main__":
    main()




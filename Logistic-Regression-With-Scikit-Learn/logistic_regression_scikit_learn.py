"""
Logistic Regression with Scikit-Learn

This project demonstrates how to train a Logistic Regression model
using scikit-learn and visualize the resulting decision regions.

Author: A. Umair
University of Moratuwa
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

from sklearn.linear_model import LogisticRegression

def main():
    lr_model = LogisticRegression()
    lr_model.fit(X, y)

    y_pred = lr_model.predict(X)
    print("Predicted labels:", y_pred)
    print("Model coefficients:", lr_model.coef_)


    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = lr_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[y==0,0], X[y==0,1],
                color='blue', marker='o')

    plt.scatter(X[y==1,0], X[y==1,1],
                color='red', marker='x')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Logistic Regression Decision Regions")
    plt.legend()
    plt.grid(True)
    plt.savefig("Logistic-Regression-With-Scikit-Learn/images/logistic_regression_decision_regions.png", dpi=300)
    plt.show()
    Z = lr_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.4)
    plt.colorbar(label="Probability of Class 1")
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
    plt.savefig("Logistic-Regression-With-Scikit-Learn/images/logistic_regression_probability_contours.png", dpi=300)
    plt.show()

    w = lr_model.coef_[0]
    b = lr_model.intercept_[0]

    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = -(w[0] * x_vals + b) / w[1]

    plt.plot(x_vals, y_vals, 'k-', linewidth=2, label="Decision Boundary")
    plt.show()

if __name__ == "__main__":
    main()
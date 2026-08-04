"""
Logistic Regression with Gradient Descent and Regularization

This project implements Logistic Regression from scratch
using NumPy and demonstrates both linear and regularized
classification problems.

Author: A. Umair
"""

import numpy as np
import matplotlib.pyplot as plt
import math



def sigmoid(z: np.ndarray) -> np.ndarray :

    return 1 / (1 + np.exp(-z))

def compute_cost(X: np.ndarray, y: np.ndarray, w:np.ndarray, b: float) -> float :
    m = X.shape[0]
    z = np.dot(X, w) + b
    h = sigmoid(z)
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, alpha: float, num_iters: int, compute_cost: callable) -> tuple[np.ndarray, float, list[float], list[np.ndarray]]:
    m = X.shape[0]
    j_history = []
    w_history = []
    for i in range(num_iters):
        z = np.dot(X, w) + b
        h = sigmoid(z)
        dw = (1/m) * np.dot(X.T, (h - y))
        db = (1/m) * np.sum(h - y)
        w -= alpha * dw
        b -= alpha * db

        j_history.append(compute_cost(X, y, w, b))

        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(j_history[-1]):8.2f}   ")

    return w, b, j_history, w_history

def map_feature(X1: np.ndarray, X2: np.ndarray) -> np.ndarray :
    """
    Maps two input features X1 and X2 to polynomial terms 
    up to degree 6 (resulting in 27 features).
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    
    degree = 6
    out = []
    
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))
            
    return np.stack(out, axis=1)

def compute_cost_regularization(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float ) -> float:

    m, n = x.shape
    cost_without_reg = compute_cost(x, y, w, b)
    reg_cost = 0
    for j in range (n):
        reg_cost += w[j]**2
    reg_cost = (reg_cost * lambda_)/ (2*m) 
    total_cost = reg_cost + cost_without_reg

    return total_cost

def compute_gradient_descent_reg(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float, alpha: float, num_iters: int ) -> tuple[np.ndarray, float, list[float], list[np.ndarray]]:
    m, n = x.shape
    j_history = []
    w_history = []


    for i in range (num_iters):
        z = np.dot(x, w) + b
        h = sigmoid(z)
        dw = (1/m) * np.dot(x.T, (h - y))
        db = (1/m) * np.sum(h - y)
        dw += (lambda_ / m) * w

        w -= alpha * dw
        b -= alpha * db
    
        j_history.append(compute_cost_regularization(x, y, w, b, lambda_))
    
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
                w_history.append(w.copy())
                print(f"Iteration {i:4}: Cost {float(j_history[-1]):8.2f}   ")
    
    return w, b, j_history, w_history

def admission_example():
    x_train = np.array([[34.62365962, 78.02469282],
    [30.28671077, 43.89499752],
    [35.84740877, 72.90219803],
    [60.18259939, 86.3085521 ],
    [79.03273605, 75.34437644],
    [45.08327748, 56.31637178],
    [61.10666454, 96.51142588],
    [75.02474557, 46.55401354],
    [76.0987867, 87.42056972],
    [84.43281996, 43.53339331],
    [95.86155507, 38.22527806],
    [75.01365839, 30.60326323],
    [82.30705337, 76.4819633 ],
    [69.36458876, 97.71869196],
    [39.53833914, 76.03681085],
    [53.97105215, 89.20735014],    
    [69.07014406, 52.74046973],
    [67.94685548, 46.67857411],
    [70.66150955, 92.92713789],
    [76.97878373, 47.57596365],
    [67.37202755, 42.83843832],
    [89.67677575, 65.79936593],
    [50.53478829, 48.85581153],
    [34.21206098, 44.2095286 ],
    [77.92409145, 68.97235999],
    [62.27101367, 69.95445795],
    [80.19018075, 44.82162893],
    [93.1143888, 38.80067034],
    [61.83020602, 50.25610789],
    [38.7858038, 64.99568096],
    [61.37928945, 72.80788731],
    [85.40451939, 57.05198398],
    [52.10797973, 63.12762377],
    [52.04540477, 69.43286012],
    [40.23689374, 71.16774802],
    [54.63510555, 52.21388588],
    [33.91550011, 98.86943574],
    [64.17698887, 80.90806059],
    [74.78925296, 41.57341523],
    [34.18364003, 75.23772034],
    [83.90239366, 56.30804622],
    [51.54772027, 46.85629026],
    [94.44336777, 65.56892161],
    [82.36875376, 40.61825516],
    [51.04775177, 45.82270146],
    [62.22267576, 52.06099195],
    [77.19303493, 70.4582    ],
    [97.77159928, 86.72782233],
    [62.0730638, 96.76882412],
    [91.5649745, 88.69629255],
    [79.94481794, 74.16311935],
    [99.27252693, 60.999031  ],
    [90.54671411, 43.39060181],
    [34.52451385, 60.39634246],
    [50.28649612, 49.80453881],
    [49.58667722, 59.80895099],
    [97.64563396, 68.86157272],
    [32.57720017, 95.59854761],
    [74.24869137, 69.82457123],
    [71.79646206, 78.45356225],
    [75.39561147, 85.75993667],
    [35.28611282, 47.02051395],
    [56.2538175, 39.26147251],
    [30.05882245, 49.59297387],
    [44.66826172, 66.45008615],
    [66.56089447, 41.09209808],
    [40.45755098, 97.53518549],
    [49.07256322, 51.88321182],
    [80.27957401, 92.11606081],
    [66.74671857, 60.99139403],
    [32.72283304, 43.30717306],
    [64.03932042, 78.03168802],
    [72.34649423, 96.22759297],
    [60.45788574, 73.0949981 ],
    [58.84095622, 75.85844831],
    [99.8278578, 72.36925193],
    [47.26426911, 88.475865  ],
    [50.4581598,  75.80985953],
    [60.45555629, 42.50840944],
    [82.22666158, 42.71987854],
    [88.91389642, 69.8037889 ],
    [94.83450672, 45.6943068 ],
    [67.31925747, 66.58935318],
    [57.23870632, 59.51428198],
    [80.366756,   90.9601479 ],
    [68.46852179, 85.5943071 ],
    [42.07545454, 78.844786  ],
    [75.47770201, 90.424539  ],
    [78.63542435, 96.64742717],
    [52.34800399, 60.76950526],
    [94.09433113, 77.15910509],
    [90.44855097, 87.50879176],
    [55.48216114, 35.57070347],
    [74.49269242, 84.84513685],
    [89.84580671, 45.35828361],
    [83.48916274, 48.3802858 ],
    [42.26170081, 87.10385094],
    [99.31500881, 68.77540947],
    [55.34001756, 64.93193801],
    [74.775893,   89.5298129 ]])

    y_train = np.array([0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 0., 0.,
    1., 1., 1., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1.,
    1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1.,
    1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1.,
    1., 1., 1., 1.])

    admitted = y_train == 1
    not_admitted = y_train == 0

    plt.scatter(x_train[admitted,0], x_train[admitted, 1], c = 'yellow', marker = 'o', label = 'Admitted')
    plt.scatter(x_train[not_admitted, 0], x_train[not_admitted, 1], c = 'blue', marker = 'x', label = 'Not Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.title('Exam Scores vs Admission')
    plt.legend()
    plt.savefig('Logistic-Regression-with-Regularization/images/Data of Exam Scores.jpg', dpi = 300)
    plt.show()

    m, n = x_train.shape
    intial_w = np.zeros(n)
    intial_b = 0
    intial_cost = compute_cost(x_train, y_train, intial_w, intial_b)
    print(f'Initial cost: {intial_cost:.4f}')

    intial_w = 0.01 * (np.random.rand (2) - 0.5) 
    intial_b = -8
    alpha = 0.001
    num_iters = 10000

    w, b, j_history, w_history = gradient_descent(x_train, y_train, intial_w, intial_b, alpha, num_iters, compute_cost)

    x_0_boundary = np.linspace(20, 100, 100)
    x_1_boundary = -(w[0]*x_0_boundary + b)/w[1]
    plt.scatter(x_train[admitted,0], x_train[admitted, 1], c = 'yellow', marker = 'o', label = 'Admitted')
    plt.scatter(x_train[not_admitted, 0], x_train[not_admitted, 1], c = 'blue', marker = 'x', label = 'Not Admitted')
    plt.plot(x_0_boundary, x_1_boundary, c = 'red', label = 'Decision Boundary')
    plt.xlabel('Exam 1 Score')  
    plt.ylabel('Exam 2 Score')
    plt.title('Exam Scores vs Admission')
    plt.savefig('Logistic-Regression-with-Regularization/images/Boundary of Exam Scores Data.jpg', dpi = 300)
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(j_history[:20])
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iteration")
    plt.grid(True)
    plt.savefig("Logistic-Regression-with-Regularization/images/cost_vs_iteration.png")
    plt.show()

    probabilities = sigmoid(np.dot(x_train, w) + b)
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = np.mean(predictions == y_train) * 100
    print(f"Training Accuracy: {accuracy:.2f}%")
    
# Logistic Regression with Regularization
def regularization_example():

    print("Regularized Example")

    x_train = np.array([[ 0.051267,   0.69956  ],
    [-0.092742,   0.68494  ],
    [-0.21371,    0.69225  ],
    [-0.375,      0.50219  ],
    [-0.51325,    0.46564  ],
    [-0.52477,    0.2098   ],
    [-0.39804 ,   0.034357 ],
    [-0.30588,   -0.19225  ],
    [ 0.016705,  -0.40424  ],
    [ 0.13191,   -0.51389  ],
    [ 0.38537,   -0.56506  ],
    [ 0.52938 ,  -0.5212   ],
    [ 0.63882 ,  -0.24342  ],
    [ 0.73675  , -0.18494  ],
    [ 0.54666  ,  0.48757  ],
    [ 0.322    ,  0.5826   ],
    [ 0.16647  ,  0.53874  ],
    [-0.046659 ,  0.81652  ],
    [-0.17339  ,  0.69956  ],
    [-0.47869  ,  0.63377  ],
    [-0.60541  ,  0.59722  ],
    [-0.62846  ,  0.33406  ],
    [-0.59389  ,  0.005117 ],
    [-0.42108  , -0.27266  ],
    [-0.11578  , -0.39693  ],
    [ 0.20104  , -0.60161  ],
    [ 0.46601  , -0.53582  ],
    [ 0.67339  , -0.53582  ],
    [-0.13882  ,  0.54605  ],
    [-0.29435  ,  0.77997  ],
    [-0.26555  ,  0.96272  ],
    [-0.16187  ,  0.8019   ],
    [-0.17339  ,  0.64839  ],
    [-0.28283  ,  0.47295  ],
    [-0.36348,0.31213  ],
    [-0.30012  ,  0.027047 ],
    [-0.23675 ,  -0.21418  ],
    [-0.06394 ,  -0.18494  ],
    [ 0.062788,  -0.16301  ],
    [ 0.22984 ,  -0.41155  ],
    [ 0.2932  ,  -0.2288   ],
    [ 0.48329 ,  -0.18494  ],
    [ 0.64459 ,  -0.14108  ],
    [ 0.46025 ,   0.012427 ],
    [ 0.6273  ,   0.15863  ],
    [ 0.57546 ,   0.26827  ],
    [ 0.72523 ,   0.44371  ],
    [ 0.22408 ,   0.52412  ],
    [ 0.44297 ,   0.67032  ],
    [ 0.322   ,   0.69225  ],
    [ 0.13767 ,   0.57529  ],
    [-0.0063364,  0.39985  ],
    [-0.092742 ,  0.55336  ],
    [-0.20795  ,  0.35599  ],
    [-0.20795  ,  0.17325  ],
    [-0.43836  ,  0.21711  ],
    [-0.21947  , -0.016813 ],
    [-0.13882  , -0.27266  ],
    [ 0.18376  ,  0.93348  ],
    [ 0.22408  ,  0.77997  ],
    [ 0.29896  ,  0.61915  ],
    [ 0.50634  ,  0.75804  ],
    [ 0.61578  ,  0.7288   ],
    [ 0.60426  ,  0.59722  ],
    [ 0.76555  ,  0.50219  ],
    [ 0.92684  ,  0.3633   ],
    [ 0.82316  ,  0.27558  ],
    [ 0.96141  ,  0.085526 ],
    [ 0.93836  ,  0.012427 ],
    [ 0.86348  , -0.082602 ],
    [ 0.89804  , -0.20687  ],
    [ 0.85196  , -0.36769  ],
    [ 0.82892  , -0.5212   ],
    [ 0.79435  , -0.55775  ],
    [ 0.59274  , -0.7405   ],
    [ 0.51786  , -0.5943   ],
    [ 0.46601  , -0.41886  ],
    [ 0.35081  , -0.57968  ],
    [ 0.28744  , -0.76974  ],
    [ 0.085829 , -0.75512  ],
    [ 0.14919  , -0.57968  ],
    [-0.13306  , -0.4481   ],
    [-0.40956  , -0.41155  ],
    [-0.39228  , -0.25804  ],
    [-0.74366  , -0.25804  ],
    [-0.69758  ,  0.041667 ],
    [-0.75518  ,  0.2902   ],
    [-0.69758  ,  0.68494  ],
    [-0.4038   ,  0.70687  ],
    [-0.38076  ,  0.91886  ],
    [-0.50749  ,  0.90424  ],
    [-0.54781  ,  0.70687  ],
    [ 0.10311  ,  0.77997  ],
    [ 0.057028 ,  0.91886  ],
    [-0.10426  ,  0.99196  ],
    [-0.081221 ,  1.1089   ],
    [ 0.28744  ,  1.087    ],
    [ 0.39689  ,  0.82383  ],
    [ 0.63882  ,  0.88962  ],
    [ 0.82316  ,  0.66301  ],
    [ 0.67339  ,  0.64108  ],
    [ 1.0709   ,  0.10015  ],
    [-0.046659 , -0.57968  ],
    [-0.23675  , -0.63816  ],
    [-0.15035  , -0.36769  ],
    [-0.49021  , -0.3019   ],
    [-0.46717  , -0.13377  ],
    [-0.28859  , -0.060673 ],
    [-0.61118  , -0.067982 ],
    [-0.66302  , -0.21418  ],
    [-0.59965  , -0.41886  ],
    [-0.72638  , -0.082602 ],
    [-0.83007  ,  0.31213  ],
    [-0.72062  ,  0.53874  ],
    [-0.59389  ,  0.49488  ],
    [-0.48445  ,  0.99927  ],
    [-0.0063364,  0.99927  ],
    [ 0.63265  , -0.030612 ]])

    y_train = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    pos = y_train == 1
    neg = y_train == 0

    plt.scatter(x_train[pos,0], x_train[pos, 1], c = 'yellow', marker = 'o', label = 'Admitted')
    plt.scatter(x_train[neg, 0], x_train[neg, 1], c = 'blue', marker = 'x', label = 'Not Admitted')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('Microchip validation')
    plt.legend()
    plt.savefig('Logistic-Regression-with-Regularization/images/Data of Microchip Validation.jpg', dpi = 300)
    plt.show()

    X_mapped = map_feature(x_train[:, 0], x_train[:, 1])
    print("x_train shape before map: " , x_train.shape)
    print("x_train shape after map: " , X_mapped.shape)

    np.random.seed(1) 
    initial_w  = np.random.rand(X_mapped.shape[1]) - 0.5 
    initial_b = 0.5
    alpha = 0.01
    num_iters = 10000
    lambda_ = 0.01
    w, b, j_history, w_history = compute_gradient_descent_reg(X_mapped, y_train, initial_w, initial_b, lambda_, alpha, num_iters)


    plt.scatter(x_train[pos, 0], x_train[pos, 1], c='yellow', marker='o', label='Admitted')
    plt.scatter(x_train[neg, 0], x_train[neg, 1], c='blue', marker='x', label='Not Admitted')

    u = np.linspace(-1, 1.5, 100)
    v = np.linspace(-1, 1.5, 100)
    z = np.zeros((len(u), len(v)))

    # 3. Compute z = w * map_feature(u, v) + b for every point on the grid
    for i in range(len(u)):
        for j in range(len(v)):
            # .flatten() reshapes (1, 27) -> (27,) so np.dot returns a scalar
            mapped_point = map_feature(u[i], v[j]).flatten()
            z[i, j] = np.dot(mapped_point, w) + b

    # 4. Transpose z so that grid axes align with Matplotlib's meshgrid convention
    z = z.T
    # 5. Plot the contour line where z = 0 (the decision boundary where sigmoid(z) = 0.5)
    plt.contour(u, v, z, levels=[0], colors='red', linewidths=2)

    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title(f'Microchip Validation (lambda = {lambda_})')
    plt.legend(['Decision Boundary', 'Admitted', 'Not Admitted'])
    plt.savefig('Logistic-Regression-with-Regularization/images/Boundary For Microchip Validation Dataset.jpg', dpi = 300)
    plt.show()
    
    plt.figure(figsize=(6,4))
    plt.plot(j_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iteration")
    plt.grid(True)
    plt.savefig("Logistic-Regression-with-Regularization/images/cost_vs_iteration(Regularized).png")
    plt.show()

    probabilities = sigmoid(np.dot(x_train, w) + b)
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = np.mean(predictions == y_train) * 100
    print(f"Training Accuracy: {accuracy:.2f}%")

def main():
    admission_example()
    regularization_example()
        
        
if __name__ == "__main__":
    main()
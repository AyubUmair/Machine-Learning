import numpy as np
import matplotlib.pyplot as plt


x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
x_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])

pos = y_train == 1
neg = y_train == 0

fig, ax = plt.subplots(1,3,figsize=(12,4))

ax[0].scatter(x_train[pos],y_train[pos],c='r',marker='x')
ax[0].scatter(x_train[neg],y_train[neg],c='b',marker='o')
ax[0].set_title('One Variable Plot')
ax[0].set_xlabel('x',fontsize = 12)
ax[0].set_ylabel('y', fontsize = 12)
ax[0].set_ylim([-0.08,1.1])

def plot_data(x_train,y_train,ax):
    pos = y_train == 1
    neg = y_train == 0
    ax.scatter(x_train[pos,0],x_train[pos,1],c='r',marker='x')
    ax.scatter(x_train[neg,0],x_train[neg,1],c='b',marker='o')
    ax.set_title('Two Variable Plot')
    ax.set_xlabel('$x_0$',fontsize = 12)
    ax.set_ylabel('$x_1$',fontsize = 12)


plot_data(x_train2,y_train2,ax[1])


def sigmoid(z):
    g_z = 1 / (1 + np.exp(-z))
    return g_z
z_temp = np.linspace(-10,10,50)
y = sigmoid(z_temp)


ax[2].plot(z_temp, y, linewidth = 4, c='b')
ax[2].set_title('Sigmoid Function')
ax[2].set_xlabel('$x$',fontsize = 12)
ax[2].set_ylabel('Y Value', fontsize = 12)


w_in = np.zeros((1))
b_in = 0
g_z1 = sigmoid(w_in * x_train + b_in)
ax[0].plot(x_train,g_z1,linewidth = 4, c='b')
ax[0].fill_between(x_train,g_z1, alpha=0.2)

w_in2d = np.array([1.0, 1.0])
b_in2d = -3
g_z2 = sigmoid(w_in2d * x_train2 + b_in2d)
x_0_boundary = np.linspace(0,3,100)
x_1_boundary = -(w_in2d[0]*x_0_boundary + b_in2d)/w_in2d[1]
ax[1].plot(x_0_boundary, x_1_boundary, linewidth = 4, c='b',label ='Boundary Line')
ax[1].fill_between(x_0_boundary,x_1_boundary, alpha=0.2)

plt.savefig("Logistic-Regression_Visualization/images/logistic_regression_visualization 1d_2d_sigmoid.png", dpi=300)
plt.show()





X_train3 = np.array([
    [0.5, 1.5, 1.0], [1.0, 1.0, 0.5], [1.5, 0.5, 1.0], # Class 0
    [3.0, 0.5, 2.5], [2.0, 2.0, 3.0], [1.0, 2.5, 2.0]  # Class 1
])
y_train3 = np.array([0, 0, 0, 1, 1, 1])


pos = (y_train3 == 1)
neg = (y_train3 == 0)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X_train3[pos, 0], X_train3[pos, 1], X_train3[pos, 2], c='r', marker='x', s=100, label='y=1')
ax.scatter(X_train3[neg, 0], X_train3[neg, 1], X_train3[neg, 2], c='b', marker='o', s=100, label='y=0')


w = np.array([1.2, 1.0, 1.5])
b = -4.5

# 5. Create a 2D grid for x0 and x1 coordinates to span the surface plane
x0_range = np.linspace(0, 4, 20)
x1_range = np.linspace(0, 4, 20)
X0, X1 = np.meshgrid(x0_range, x1_range)

# 6. Solve for X2 using the decision boundary equation: w0*x0 + w1*x1 + w2*x2 + b = 0
X2 = (-b - w[0]*X0 - w[1]*X1) / w[2]


ax.plot_surface(X0, X1, X2, color='green', alpha=0.4, label='Decision Boundary')


ax.set_xlabel('$x_0$', fontsize=12)
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_zlabel('$x_2$', fontsize=12)
ax.set_title('3D Classification Decision Boundary Plane')


ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_zlim(0, 4)

plt.savefig("Logistic-Regression_Visualization/images/logistic_regression_visualization 3d.png", dpi=300)
plt.show()
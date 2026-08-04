import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x_train = np.array([[ 4.33e+00 ,-1.99e+00],
 [ 4.10e+00, -2.31e+00],
 [ 4.50e+00 ,-2.19e+00],
 [-5.40e+00,  3.02e+00],
 [ 4.22e+00 ,-3.10e-02],
 [ 3.76e+00 ,-3.61e+00],
 [ 7.19e+00, -2.76e+00],
 [ 5.44e+00,-1.41e+00],
 [ 1.60e+00,  1.63e+00],
 [-5.03e+00,  3.39e+00],
 [ 1.23e+00 , 6.32e-01],
 [-1.22e+00, -3.79e+00],
 [-3.06e+00 ,-3.28e+00],
 [-1.68e+00 ,-2.99e+00],
 [-3.57e+00,  1.34e+00],
 [ 4.60e+00, -9.70e-01],
 [ 3.83e-02,  2.95e+00],
 [ 4.16e+00 ,-3.48e+00],
 [-7.32e-01, -2.28e+00],
 [ 5.90e-01,  2.58e+00],
 [ 4.08e+00, -1.63e+00],
 [-5.29e+00,  2.54e+00],
 [-3.16e+00 ,-2.02e+00],
 [ 1.93e+00 , 6.09e-01],
 [ 1.47e+00,  2.38e+00],
 [-3.62e+00,  2.76e+00],
 [ 5.06e+00, -6.77e-03],
 [-2.26e+00, -1.43e+00],
 [-5.39e+00,  7.27e-02],
 [-7.08e+00,  2.94e+00],
 [ 4.36e+00, -1.60e+00],
 [-3.10e+00,  1.65e+00],
 [ 3.61e+00, -1.92e+00],
 [ 2.52e+00,  3.24e+00],
 [ 4.94e+00, -1.50e+00],
 [-2.17e+00, -2.22e+00],
 [-2.95e-01, -1.12e+00],
 [-4.74e-01,  1.40e+00],
 [ 4.30e+00, -1.64e+00],
 [-4.19e+00,  9.55e-01],
 [-4.87e+00,  8.93e-01],
 [ 1.44e+00,  2.51e+00],
 [ 4.61e+00, -2.10e+00],
 [-5.10e+00,  2.30e+00],
 [ 4.21e+00, -2.24e+00],
 [-2.41e+00, -3.06e+00],
 [-1.09e+00, -3.39e+00],
 [ 5.23e+00, -6.66e-02],
 [ 7.18e+00, -3.41e+00],
 [-3.42e+00,  2.11e+00],
 [-6.26e+00,  3.53e+00],
 [ 2.02e+00,  2.74e+00],
 [-7.90e+00,  3.62e+00],
 [ 1.38e+00,  2.81e+00],
 [ 1.16e+00,  2.15e+00],
 [-5.97e+00,  2.47e+00],
 [-3.95e+00,  1.46e+00],
 [ 5.59e+00,  7.44e-02],
 [ 3.28e+00, -3.30e-01],
 [ 4.28e-01,  1.19e+00],
 [-5.76e+00,  1.22e+00],
 [-6.10e+00,  6.84e-01],
 [-5.69e+00,  1.15e+00],
 [-1.97e+00, -2.45e+00],
 [ 5.77e-01,  1.83e+00],
 [-3.58e+00, -3.24e+00],
 [-1.42e+00, -2.50e+00],
 [-1.45e+00, -1.05e+00],
 [ 7.10e+00, -2.54e+00],
 [-2.02e+00, -2.07e+00],
 [-2.30e+00, -3.06e+00],
 [ 1.46e+00,  1.62e+00],
 [ 6.51e+00, -1.26e+00],
 [-5.01e+00,  1.53e+00],
 [-1.66e+00, -4.59e+00],
 [-8.84e-01,  2.36e+00],
 [ 1.42e+00,  2.52e+00],
 [-6.73e+00,  3.59e+00],
 [-3.16e+00, -1.10e+00],
 [-2.13e+00, -2.90e+00],
 [ 1.58e+00,  1.39e+00],
 [ 1.76e+00,  2.40e+00],
 [ 4.02e+00,  1.67e+00],
 [-1.12e+00, -3.49e+00],
 [-1.01e+00, -2.59e+00],
 [-3.89e+00,  7.88e-02],
 [-7.58e-01, -2.92e-01],
 [-5.70e+00,  2.85e+00],
 [ 3.41e-01,  2.70e+00],
 [ 4.79e+00, -2.05e+00],
 [ 4.38e+00, -1.63e+00],
 [-5.42e+00,  3.06e+00],
 [ 1.79e+00,  1.74e+00],
 [-1.79e+00, -2.46e+00],
 [ 4.61e-02,  8.88e-01],
 [-9.97e-01, -3.04e+00],
 [-1.67e+00, -1.56e+00],
 [ 4.54e-01,  1.15e+00],
 [ 2.77e+00,  2.41e+00],
 [-4.01e+00,  3.00e-01]] )

y_train = np.array([3, 3, 3, 0, 3 ,3, 3, 3, 2, 0, 2, 1, 1, 1, 0, 3, 2, 3, 1, 2, 3, 0, 1, 2, 2, 0, 3, 1, 0, 0, 3, 0, 3, 2, 3, 1, 1,
 2, 3, 0, 0, 2, 3, 0, 3, 1, 1, 3, 3, 0, 0, 2, 0, 2, 2, 0, 0, 3, 3, 2, 0, 0, 0, 1, 2, 1, 1, 1, 3, 1, 1, 2, 3, 0,
 1, 2, 2, 0, 1, 1, 2, 2, 2, 1, 1, 0, 1, 0, 2, 3, 3, 0, 2, 1, 2, 1, 1, 2, 2, 0])

def main():
    classes = 4
    m = 100 

    print(f"Number of Uniquue classes : {np.unique(y_train)}")
    print(f"Shape of x_train: {x_train.shape}  ,   Shape of y_train: {y_train.shape}")

    tf.random.set_seed(1234)

    model = Sequential(
        [
            Dense(2, activation = 'relu', name = 'layer_1'),
            Dense(4, activation = 'linear', name = 'layer_2')
        ] 
    )

    model.compile(
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                optimizer = tf.keras.optimizers.Adam(0.01),
                metrics = [ "accuracy" ]
            )

    model.summary()

    history = model.fit(
        x_train, y_train ,
        epochs = 200,
        
    )

    plt.plot(history.history['accuracy'])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.savefig("Multiclass-Clssification-With-TensorFlow/images/Training Accuracy.png", dpi = 300)
    plt.grid(True)

    l1 = model.get_layer('layer_1')
    l2 = model.get_layer('layer_2')

    w1, b1 = l1.get_weights()
    w2, b2 = l2.get_weights()

    x_min, x_max = x_train[ :, 0 ].min() - 1, x_train[ :, 0 ].max() + 1
    y_min, y_max = x_train[ :, 1 ].min() - 1, x_train[ :, 1 ].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.05),
        np.arange(y_min, y_max, 0.05)
                                        )
    grid_points = np.c_[ xx.ravel(), yy.ravel() ]
    logits = model.predict(grid_points)
    predictions = np.argmax(logits, axis=1)
    Z = predictions.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    scatter = plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Set1, edgecolors='k')

    plt.title("Neural Network Decision Boundaries")
    plt.xlabel("Feature $x_1$")
    plt.ylabel("Feature $x_2$")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.savefig("Multiclass-Clssification-With-TensorFlow/images/Boundary Plot.png", dpi = 300)
    plt.show()

if __name__ == "__main__":
    main()
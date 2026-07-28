"""
Logistic Regression using TensorFlow

This project demonstrates binary classification using
TensorFlow/Keras with feature normalization,
Binary Cross Entropy loss, and the Adam optimizer.

Author: A. Umair
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.engine import sequential

#logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X_train = np.array([[185.32 , 12.69],
 [259.92 , 11.87],
 [231.01 , 14.41],
 [175.37 , 11.72],
 [187.12 , 14.13],
 [225.91 , 12.1 ],
 [208.41 , 14.18],
 [207.08 , 14.03],
 [280.6  , 14.23],
 [202.87 , 12.25],
 [196.7  , 13.54],
 [270.31 , 14.6 ],
 [192.95 , 15.2 ],
 [213.57 , 14.28],
 [164.47 , 11.92],
 [177.26 , 15.04],
 [241.77 , 14.9 ],
 [237.   , 13.13],
 [219.74 , 13.87],
 [266.39 , 13.25],
 [270.45 , 13.95],
 [261.96 , 13.49],
 [243.49 , 12.86],
 [220.58 , 12.36],
 [163.59 , 11.65],
 [244.76 , 13.33],
 [271.19 , 14.84],
 [201.99 , 15.39],
 [229.93 , 14.56],
 [204.97 , 12.28],
 [173.19 , 12.22],
 [231.51 , 11.95],
 [152.69 , 14.83],
 [163.42 , 13.3 ],
 [215.95 , 13.98],
 [218.04 , 15.25],
 [251.3  , 13.8 ],
 [233.33 , 13.53],
 [280.24 , 12.41],
 [243.02 , 13.72],
 [155.67 , 12.68],
 [275.17 , 14.64],
 [151.73 , 12.69],
 [151.32 , 14.81],
 [164.9  , 11.73],
 [282.55 , 13.28],
 [192.98 , 11.7 ],
 [202.6  , 12.96],
 [220.67 , 11.53],
 [169.97 , 12.34],
 [209.47 , 12.71],
 [232.8  , 12.64],
 [272.8  , 15.35],
 [158.02 , 12.34],
 [226.01 , 14.58],
 [158.64 , 12.24],
 [211.66 , 14.17],
 [271.95 , 14.97],
 [257.16 , 11.71],
 [281.85 , 13.96],
 [161.63 , 12.52],
 [233.8  , 13.04],
 [210.29 , 14.72],
 [261.24 , 13.69],
 [256.98 , 13.12],
 [281.56 , 13.92],
 [280.64 , 11.68],
 [269.16 , 13.74],
 [246.34 ,12.27],
 [224.07 , 12.66],
 [164.24 , 11.51],
 [272.42 , 14.18],
 [177.68 , 12.53],
 [212.86 , 14.77],
 [165.88 , 15.37],
 [277.43 , 12.48],
 [236.51 , 12.94],
 [244.14 , 11.85],
 [213.45 , 13.85],
 [234.57 , 14.27],
 [270.34 , 12.47],
 [170.68 , 13.06],
 [226.79 , 15.34],
 [245.92 , 14.45],
 [281.32 , 12.57],
 [185.03 , 13.19],
 [189.88 , 14.1 ],
 [278.48 , 12.11],
 [219.92 , 14.21],
 [216.58 , 15.15],
 [249.48  ,15.03],
 [165.09 , 12.28],
 [158.87 , 14.82],
 [279.98 , 11.56],
 [256.55 , 14.41],
 [272.61 , 12.58],
 [246.49 , 12.45],
 [160.26 , 14.48],
 [155.7  , 14.3 ],
 [188.27 , 13.45],
 [270.36 , 12.47],
 [213.22 , 12.92],
 [175.7  , 13.39],
 [174.52 , 14.7 ],
 [233.   , 12.63],
 [281.37 , 12.88],
 [240.62 , 14.43],
 [185.81 , 11.55],
 [270.5  , 15.33],
 [172.98 , 12.11],
 [208.41 , 13.89],
 [283.51 , 15.35],
 [283.36 , 12.48],
 [230.85 , 13.24],
 [181.24 , 11.76],
 [172.78 , 12.93],
 [161.88 , 12.1 ],
 [156.03 , 13.99],
 [216.52 , 12.47],
 [221.06 , 13.2 ],
 [238.99 , 15.23],
 [197.69 , 14.08],
 [179.55 , 15.26],
 [233.39 , 12.13],
 [184.7  , 12.14],
 [174.18 , 12.73],
 [261.11 , 13.33],
 [187.42 , 13.18],
 [186.1  , 14.43],
 [157.94 , 12.66],
 [193.64 , 12.23],
 [249.65 , 12.22],
 [190.56 , 11.73],
 [252.   , 12.96],
 [238.55 , 12.37],
 [152.94 , 12.79],
 [255.17 , 14.85],
 [197.09 , 14.89],
 [156.8  , 13.59],
 [184.75 , 13.26],
 [179.92 , 15.07],
 [190.79 , 15.28],
 [164.73 , 13.22],
 [209.87 , 14.34],
 [196.58 , 13.47],
 [159.51 , 12.74],
 [247.87 , 11.92],
 [212.44 , 12.45],
 [172.34 , 11.99],
 [259.87 , 14.25],
 [201.23 , 13.07],
 [248.34 , 13.92],
 [273.66 , 15.18],
 [215.09 , 14.14],
 [223.53 , 12.74],
 [211.22 , 14.38],
 [224.61 , 14.03],
 [215.75 , 15.31],
 [254.82 , 12.02],
 [259.9  , 15.17],
 [260.25 , 12.87],
 [199.67 , 12.47],
 [157.52 , 13.39],
 [264.81 , 14.58],
 [239.4  , 14.89],
 [238.98 , 12.39],
 [258.43 , 12.97],
 [270.16 , 12.81],
 [162.41 , 14.42],
 [164.53 , 14.98],
 [205.61 , 14.62],
 [157.1  , 13.68],
 [241.38 , 12.02],
 [232.13 , 12.07],
 [191.04 , 12.96],
 [233.64 , 12.02],
 [174.95 , 14.63],
 [246.64 , 13.32],
 [188.07 , 14.27],
 [213.16 , 12.75],
 [268.08 , 12.31],
 [258.58 , 13.97],
 [237.21 , 14.23],
 [251.02 , 15.02],
 [274.28 , 12.52],
 [172.12 , 15.09],
 [177.52 , 12.39],
 [258.71 , 15.36],
 [264.01 , 13.57],
 [200.71 , 15.45],
 [249.37 , 14.02],
 [151.5  , 12.28],
 [151.82 , 15.13],
 [181.92 , 12.18],
 [228.65  ,12.31],
 [223.78 , 15.3 ],
 [266.63  ,12.48],
 [273.68  ,13.1 ],
 [220.61 , 12.8 ],
 [284.99, 12.73]])

Y_train = np.array([[1.],
 [0.],
 [0.],
 [0.],
 [1.],
 [1.],
 [0.],
 [0.],
 [0.],
 [1.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [1.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [1.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],
 [1.],
 [1.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [1.],
 [0.],
 [0.],
 [0.],
 [1.],
 [1.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [1.],
 [1.],
 [1.],
 [0.],
 [0.],
 [1.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.],
 [1.],
 [0.],
 [0.],
 [0.],
 [1.],
 [0.],])

def main():
 print(X_train.shape, Y_train.shape)

 pos_mask = (Y_train == 1).reshape(-1)
 neg_mask = (Y_train == 0).reshape(-1)
 plt.scatter(X_train[pos_mask, 0], X_train[pos_mask, 1], color='red', marker='x', s=100, label='Y=1')
 plt.scatter(X_train[neg_mask, 0], X_train[neg_mask, 1], color='blue', marker='o', s=100, label='Y=0')
 plt.title("Data Set")
 plt.xlabel('Temperature')
 plt.ylabel('Duration')
 plt.savefig("Logistic-Regression-With-TensorFlow/images/ Coffee Roasting DataSet.png", dpi =300)
 plt.show()

 # Normalizing Data
 print(f'Max,Min pre Normalizing of Temperature : {np.max(X_train[:, 0]):0.2f} , {np.min(X_train[:, 0]):0.2f}')
 print(f'Max,Min pre Normalizing of Duration : {np.max(X_train[:, 1]):0.2f} , {np.min(X_train[:, 1]):0.2f}')

 norm_1 = tf.keras.layers.Normalization(axis=1)
 norm_1.adapt(X_train)  # Learns Mean and Variance
 Xn = norm_1(X_train)

 Xn_plot = Xn.numpy()
 plt.scatter(Xn_plot[pos_mask, 0], Xn_plot[pos_mask, 1], color='red', marker='x', s=100, label='Y=1')
 plt.scatter(Xn_plot[neg_mask, 0], Xn_plot[neg_mask, 1], color='blue', marker='o', s=100, label='Y=0')
 plt.title("Data Set")
 plt.xlabel('Temperature')
 plt.ylabel('Duration')
 plt.savefig("Logistic-Regression-With-TensorFlow/images/ Normalized Coffee Roasting DataSet.png", dpi =300)
 plt.show()

 print(f"Max, Min post normalization of Temperature : {np.max(Xn[:, 0]):0.2f}, {np.min(Xn[:, 0]):0.2f}")
 print(f"Max, Min post normalization of Duration : {np.max(Xn[:, 1]):0.2f}, {np.min(Xn[:, 1]):0.2f}")

 # Tile/copy our data to increase the training set size and reduce the number of training epochs.
 Xt = np.tile(Xn, (1000, 1))
 Yt = np.tile(Y_train, (1000, 1))
 print(Xt.shape, Yt.shape)

 # TensorFlow Model

 tf.random.set_seed(1234)
 model = Sequential(
  [tf.keras.Input(shape=(2,)),
   Dense(units=5, activation='sigmoid', name='layer_1'),
   Dense(units=1, activation='sigmoid', name='layer_2')
   ]
 )

 model.summary()

 L1_num_params = 2 * 5 + 5  # W1 parameters  + b1 parameters
 L2_num_params = 5 * 1 + 1  # W2 parameters  + b2 parameters
 print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params)

 w1, b1 = model.get_layer(name='layer_1').get_weights()
 w2, b2 = model.get_layer(name='layer_2').get_weights()
 print("Before Model Fitting: ")
 print(f'W1 Shape {w1.shape} , W1 = ', w1, f'b1 shape {b1.shape}  b1 = ', b1)
 print(f'W2 Shape {w2.shape} , W2 = ', w2, f'b2 shape {b2.shape}  b2 = ', b2)

 ''' The model.compile statement defines a loss function and specifies a compile optimization.
 The model.fit statement runs gradient descent and fits the weights to the data.     '''

 model.compile(
  loss=tf.keras.losses.BinaryCrossentropy(),
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
  metrics=['accuracy']
 )

 history = model.fit(
              Xt, Yt,
              epochs=10,
 )

 plt.plot(history.history["accuracy"])
 plt.xlabel("Epoch")
 plt.ylabel("Accuracy")
 plt.title("Training Accuracy")
 plt.grid(True)
 plt.savefig("Logistic-Regression-With-TensorFlow/images/ Training Accuracy.png", dpi = 300)
 plt.show()

 # After fitting, the weights have been updated:
 W1, b1 = model.get_layer("layer_1").get_weights()
 W2, b2 = model.get_layer("layer_2").get_weights()
 print("After Model Fitting: ")
 print("W1:\n", W1, "\nb1:", b1)
 print("W2:\n", W2, "\nb2:", b2)

 ''' Let's start by creating input data. The model is expecting one or more examples where examples are in the rows of matrix. In this case, we have two features so the matrix will be (m,2) where m is the number of examples. Recall, we have normalized the input features so we must normalize our test data as well.
 To make a prediction, you apply the predict method. '''

 X_test = np.array([
  [200, 13.9],  # positive example
  [200, 17],
  [300, 10],
  [150, 20]])  # negative example
 X_testn = norm_1(X_test)
 predictions = model.predict(X_testn)
 print("predictions = \n", predictions)

 # To convert the probabilities to a decision, we apply a threshold:

 yhat = np.zeros_like(predictions)


 yhat = (predictions >= 0.5).astype(int)
 print(f"decisions = \n{yhat}")

if __name__ == "__main__":
 main()
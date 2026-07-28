# Logistic Regression Using TensorFlow

This project demonstrates **binary classification using Logistic Regression implemented with TensorFlow/Keras**.

The model classifies data into two categories using:
- Feature normalization
- Neural network-based logistic regression
- Binary Cross Entropy loss function
- Adam optimizer
- Accuracy evaluation

The dataset used in this project represents a **coffee roasting classification problem**, where the model predicts whether a coffee roast belongs to a specific class based on:

- Temperature
- Duration

---

## Project Overview

In traditional logistic regression, the model learns a decision boundary to separate two classes.

In this implementation, TensorFlow/Keras is used to build a small neural network:

```
Input Layer (2 features)
        |
        ↓
Dense Layer (5 neurons, Sigmoid activation)
        |
        ↓
Output Layer (1 neuron, Sigmoid activation)
        |
        ↓
Binary Classification
```

The output of the model is a probability between **0 and 1**.

Example:

```
Prediction = 0.85 → Class 1
Prediction = 0.20 → Class 0
```

---

## Features

- Data visualization before normalization
- Feature normalization using TensorFlow Normalization layer
- Binary classification model using Keras Sequential API
- Training using Adam optimizer
- Binary Cross Entropy loss calculation
- Accuracy visualization during training
- Prediction on new unseen samples

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## Project Structure

```
Logistic-Regression-With-TensorFlow/
│
├── Logistic_Regression.py
├── README.md
├── requirements.txt
│
└── images/
    ├── Coffee Roasting DataSet.png
    ├── Normalized Coffee Roasting Dataset.png
    └── Training Accuracy.png
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AyubUmair/Logistic-Regression-With-TensorFlow.git
```

### 2. Navigate into the project directory

```bash
cd Logistic-Regression-With-TensorFlow
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate virtual environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run:

```bash
python Logistic_Regression.py
```

The program will:

1. Display the original dataset
2. Normalize input features
3. Train the TensorFlow model
4. Display training accuracy
5. Predict new examples

---

## Data Normalization

The input features have different ranges:

Before normalization:

| Feature | Range |
|---|---|
| Temperature | ~150 - 285 |
| Duration | ~11 - 15 |

Since these values have different scales, normalization is applied:

```python
norm_1 = tf.keras.layers.Normalization(axis=1)
norm_1.adapt(X_train)

Xn = norm_1(X_train)
```

After normalization, features are transformed approximately into:

```
Mean ≈ 0
Standard deviation ≈ 1
```

This improves the convergence speed during training.

---

## Model Architecture

The neural network is created using:

```python
model = Sequential(
[
    tf.keras.Input(shape=(2,)),
    Dense(5, activation='sigmoid'),
    Dense(1, activation='sigmoid')
]
)
```

### Layers

| Layer | Neurons | Activation |
|---|---|---|
| Input | 2 | - |
| Hidden Layer | 5 | Sigmoid |
| Output Layer | 1 | Sigmoid |

---

## Model Compilation

The model uses:

```python
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)
```

### Loss Function

Binary Cross Entropy is used because this is a binary classification problem.

### Optimizer

Adam optimizer is used to update model weights efficiently during training.

---

## Training

The dataset is duplicated using:

```python
np.tile()
```

to increase the training examples and reduce the number of epochs required.

Training:

```python
model.fit(
    Xt,
    Yt,
    epochs=10
)
```

---

## Results

The model learns a decision boundary between the two classes.

Example predictions:

```python
X_test = np.array([
    [200,13.9],
    [200,17],
    [300,10],
    [150,20]
])
```

Output:

```
Prediction probabilities:
[
  [0.xx],
  [0.xx],
  [0.xx],
  [0.xx]
]
```

A threshold of 0.5 is used:

```
Probability >= 0.5 → Class 1
Probability < 0.5  → Class 0
```

---

## Generated Visualizations

### Original Dataset

Shows the distribution of the two classes using original feature values.

![Dataset](images/Coffee%20Roasting%20DataSet.png)

---

### Normalized Dataset

Shows the same data after feature scaling.

![Normalized Dataset](images/Normalized%20Coffee%20Roasting%20Dataset.png)

---

### Training Accuracy

Shows how the model accuracy improves during training.

![Accuracy](images/Training%20Accuracy.png)

---

## Requirements

Example `requirements.txt`:

```
tensorflow
numpy
matplotlib
```

---

## Author

**A. Umair**

Electronic and Telecommunications Engineering Undergraduate  
Interested in:
- Machine Learning
- Computer Vision
- Signal Processing
- Artificial Intelligence

---

## License

This project is for educational purposes and can be freely used for learning and experimentation.
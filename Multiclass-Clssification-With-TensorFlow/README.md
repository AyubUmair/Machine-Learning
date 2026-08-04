# Multiclass Classification with TensorFlow

This project demonstrates multiclass classification using TensorFlow/Keras on a synthetic dataset. A simple feedforward neural network is trained using the Adam optimizer and Sparse Categorical Cross-Entropy loss. The project also visualizes the training accuracy and the learned decision boundaries.

---

## Features

- Multiclass classification (4 classes)
- Neural network built using TensorFlow/Keras
- ReLU hidden layer
- Sparse Categorical Cross-Entropy loss
- Adam optimizer
- Training accuracy visualization
- Decision boundary visualization

---

## Dataset

- Samples: 100
- Features: 2
- Classes: 4

Each sample contains two numerical features, while the target labels belong to one of four classes.

---

## Model Architecture

Input (2 Features)
↓
Dense Layer (2 Neurons, ReLU)
↓
Dense Layer (4 Neurons, Linear Logits)
↓
SparseCategoricalCrossentropy(from_logits=True)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.01 |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 200 |
| Metrics | Accuracy |

---

## Visualizations

The project generates:

- Training Accuracy vs Epoch
- Neural Network Decision Boundary

Example outputs are saved inside the `images/` directory.

```
images/
├── Training Accuracy.png
└── Boundary Plot.png
```

---

## Requirements

- Python 3.11 or 3.12
- TensorFlow
- NumPy
- Matplotlib

Install dependencies using:

```bash
pip install tensorflow numpy matplotlib
```

---

## Run

```bash
python multiclass_classification_tensorflow.py
```

---

## Project Structure

```
Multiclass-Classification-With-TensorFlow/
│
├── multiclass_classification_tensorflow.py
├── README.md
├── requirements.txt
├── .gitignore
└── images/
    ├── Training Accuracy.png
    └── Boundary Plot.png
```

---

## Learning Objectives

This project demonstrates:

- Multiclass classification
- Feedforward neural networks
- Dense layers
- ReLU activation
- Logits
- Sparse Categorical Crossentropy
- Adam optimization
- Decision boundary visualization
- TensorFlow/Keras workflow

---

## Author

**A. Umair**

Electronic and Telecommunication Engineering Undergraduate

University of Moratuwa

## License

This project is for educational purposes and can be freely used for learning and experimentation.
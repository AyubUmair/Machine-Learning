# Handwritten Digit Recognition using Random Forest (Scikit-Learn)

A machine learning project that recognizes handwritten digits using the **Random Forest Classifier** from **Scikit-Learn**. The project loads handwritten digit images from the original Optical Digits dataset, preprocesses them, trains a Random Forest model, and evaluates its performance using multiple metrics and visualizations.

---

## Features

- Load handwritten digit images from the original `.cv` dataset
- Convert 32×32 bitmap images into feature vectors
- Normalize pixel values
- Split data into training and testing sets
- Train a Random Forest classifier
- Evaluate model performance using:
  - Accuracy Score
  - Classification Report
  - Confusion Matrix
- Visualize:
  - Sample handwritten digit
  - Confusion Matrix
  - Feature Importance Heatmap
- Save generated plots as high-quality images

---

## Dataset

This project uses the **Optical Recognition of Handwritten Digits** dataset.

Each sample consists of:

- **32 × 32** binary image
- Flattened into **1024 input features**
- Label representing one of the digits **0–9**

The dataset is stored as:

```
optdigits-orig.cv
```

---

## Project Structure

```
HandWritten-Digit-Recognition-With-Scikit-Learn/
│
├── images/
│   ├── Sample Figure.png
│   ├── Confusion Matrix Plot.png
│   └── Feature Importance Plot.png
│
├── optdigits-orig.cv
├── random_forest.py
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/HandWritten-Digit-Recognition-With-Scikit-Learn.git
```

Navigate to the project directory

```bash
cd HandWritten-Digit-Recognition-With-Scikit-Learn
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## Requirements

```
numpy
matplotlib
scikit-learn
```

or install manually

```bash
pip install numpy matplotlib scikit-learn
```

---

## Running the Project

Run the Python script

```bash
python random_forest.py
```

The program performs the following steps:

1. Loads the handwritten digit dataset.
2. Converts each 32×32 image into a 1024-dimensional feature vector.
3. Displays a sample handwritten digit.
4. Normalizes pixel values.
5. Splits the dataset into training and testing sets.
6. Trains a Random Forest classifier.
7. Predicts the test dataset.
8. Prints model accuracy and classification report.
9. Displays and saves the confusion matrix.
10. Displays and saves the feature importance heatmap.

---

## Machine Learning Pipeline

```
Dataset
    │
    ▼
Load Dataset
    │
    ▼
Convert Images to Feature Vectors
    │
    ▼
Normalize Pixel Values
    │
    ▼
Train-Test Split (80% / 20%)
    │
    ▼
Random Forest Classifier
    │
    ▼
Prediction
    │
    ▼
Model Evaluation
    │
    ├── Accuracy Score
    ├── Classification Report
    ├── Confusion Matrix
    └── Feature Importance Heatmap
```

---

## Example Output

```
X shape: (1934, 1024)
y shape: (1934,)

X_train Shape: (1547, 1024)
X_test Shape: (387, 1024)

Model Accuracy: 0.98

Classification Report

              precision    recall    f1-score

...
```

*The reported accuracy may vary slightly depending on the random seed and dataset split.*

---

## Sample Images

### Sample Handwritten Digit

Displays one image from the dataset before training.

![Sample Digit](images/Sample%20Figure.png)

---

### Confusion Matrix

Visualizes the classification performance across all digit classes.

![Confusion Matrix](images/Confusion%20Matrix%20Plot.png)

---

### Feature Importance Heatmap

Shows which pixels contribute the most to the Random Forest model's predictions.

![Feature Importance](images/Feature%20Importance%20Plot.png)

---

## Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-Learn

### Machine Learning Algorithm

- Random Forest Classifier

---

## Performance

The Random Forest classifier achieves high classification accuracy on the Optical Digits dataset while requiring minimal preprocessing.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Future Improvements

- Hyperparameter tuning using `GridSearchCV`
- K-Fold Cross Validation
- Save the trained model using `joblib`
- Compare with other machine learning algorithms:
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Logistic Regression
  - Multi-layer Perceptron (MLP)
  - Convolutional Neural Network (CNN)
- Create a GUI for digit prediction
- Predict handwritten digits from user-provided image files

---

## Author

**A. Umair**

Electronic and Telecommunication Engineering Undergraduate

University of Moratuwa

---

## License

This project is intended for educational and learning purposes.
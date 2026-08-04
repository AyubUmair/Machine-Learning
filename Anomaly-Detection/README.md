# Gaussian Anomaly Detection from Scratch

A Python implementation of **Anomaly Detection** using the **Multivariate Gaussian Distribution**. This project estimates the probability distribution of normal data, determines the optimal anomaly threshold using cross-validation, and detects anomalous observations.

## Overview

This project demonstrates how anomaly detection works without using machine learning libraries such as Scikit-learn. Instead, the Gaussian probability density function is implemented manually using NumPy.

The algorithm:

1. Estimates the mean and variance of each feature.
2. Computes the multivariate Gaussian probability density.
3. Selects the optimal anomaly threshold (`ε`) using cross-validation.
4. Detects anomalies based on the probability threshold.
5. Visualizes the Gaussian distribution and detected outliers.

---

## Features

- Manual implementation of Multivariate Gaussian Distribution
- Gaussian contour visualization
- Automatic threshold selection using F1 Score
- Outlier detection
- NumPy-based implementation
- Matplotlib visualizations

---

## Dataset

The dataset consists of two numerical features:

- **Latency (ms)**
- **Throughput (mb/s)**

Training data contains mostly normal examples, while the validation set contains both normal and anomalous examples.

---

## Project Structure

```
Gaussian-Anomaly-Detection/
│
├── anomaly_detection.py
├── README.md
├── images/
│   ├── dataset.png
│   ├── gaussian_contours.png
│   └── detected_outliers.png
```

---

## Mathematical Background

The probability of each example is calculated using the multivariate Gaussian distribution:

\[
p(x)=\frac{1}{(2\pi)^{k/2}|\Sigma|^{1/2}}
\exp\left(
-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)
\right)
\]

where

- **μ** = Mean vector
- **Σ** = Covariance matrix
- **k** = Number of features

Examples with probabilities below a threshold are considered anomalies.

---

## Threshold Selection

Instead of manually selecting the threshold, this project searches over possible values of ε and computes:

- Precision
- Recall
- F1 Score

The threshold with the highest F1 score is selected.

---

## Results

The program prints:

- Mean of each feature
- Variance of each feature
- Best epsilon
- Best F1 score

It also generates:

- Scatter plot of the dataset
- Gaussian contour plot
- Detected anomalies highlighted with red circles

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Gaussian-Anomaly-Detection.git
```

Move into the project

```bash
cd Gaussian-Anomaly-Detection
```

Install dependencies

```bash
pip install numpy matplotlib
```

Run

```bash
python anomaly_detection.py
```

---

## Dependencies

- Python 3.10+
- NumPy
- Matplotlib

---

## Future Improvements

- Estimate the full covariance matrix instead of assuming independent features.
- Load datasets from CSV files.
- Interactive visualization.
- ROC Curve and Precision-Recall Curve.
- Compare against Scikit-learn's EllipticEnvelope and Isolation Forest.
- Save detected anomalies to CSV.
- Command-line interface for custom datasets.

---

## Learning Outcomes

This project demonstrates understanding of:

- Probability distributions
- Gaussian Density Estimation
- Unsupervised Learning
- Anomaly Detection
- Precision
- Recall
- F1 Score
- NumPy vectorization
- Data visualization

---

## Author

**A. Umair**

Electronic and Telecommunication Engineering Undergraduate

University of Moratuwa
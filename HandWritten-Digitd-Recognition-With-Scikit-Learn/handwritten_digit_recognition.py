import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("HandWritten-Digitd-Recognition-With-Scikit-Learn/optdigits-orig.cv")
print("Shape of DataSet: ", df.shape)

def load_dataset(path,grid_height):
    X_list = []
    y_list = []

    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        if not lines[i][0].isdigit():
            i = i+1
            continue
        if i+grid_height >= len(lines):
            break
        
        pixel_rows = lines[i : i + grid_height]
        label_line = lines[i + grid_height]
        
        if label_line.strip().lstrip('-').isdigit():
            pixels = [int(char) for row in pixel_rows for char in row if char.isdigit()]
            if len(pixels) == grid_height * grid_height:
                X_list.append(pixels)
                y_list.append(int(label_line.strip()))
                i += grid_height + 1
            else:
                i += 1
        else:
            i += 1
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

grid_height = 32
X, y = load_dataset("HandWritten-Digitd-Recognition-With-Scikit-Learn/optdigits-orig.cv", grid_height)
print("X shape: ", X.shape)
print("y shape: ", y.shape)
print("Few elements of y: ",y[:10])

X = X /255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)

model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

y_prediction = model.predict(X_test)

print("Model Accuracy: ", accuracy_score(y_test,y_prediction))
print("\nClassification Report\n", classification_report(y_test, y_prediction))





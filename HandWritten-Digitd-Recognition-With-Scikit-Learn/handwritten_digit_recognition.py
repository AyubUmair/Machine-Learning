import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



def load_dataset(path,grid_height):
    """
    Loads the handwritten digit dataset from the original text file.

    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
    """
        
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

def main():
    DATASET_PATH = "HandWritten-Digitd-Recognition-With-Scikit-Learn/optdigits-orig.cv"
    grid_height = 32
    X, y = load_dataset(DATASET_PATH, grid_height)
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("Few elements of y: ",y[:10])

    plt.imshow(X[0].reshape(32,32), cmap='grey')
    plt.title(f"label {y[0]}")
    plt.axis(False)
    plt.savefig("HandWritten-Digitd-Recognition-With-Scikit-Learn/images/Sample Figure.png", dpi = 300)
    plt.show()

    X = X /255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    model.fit(X_train,y_train)

    y_prediction = model.predict(X_test)

    print("Model Accuracy: ", accuracy_score(y_test,y_prediction))
    print("\nClassification Report\n", classification_report(y_test, y_prediction))


    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.savefig("HandWritten-Digitd-Recognition-With-Scikit-Learn/images/Confusion MAtrix Plot.png", dpi = 300)
    plt.show()

    importance = model.feature_importances_
    plt.figure(figsize=(8,8))
    plt.imshow(importance.reshape(32,32), cmap="hot")
    plt.colorbar()
    plt.title("Feature Importance")
    plt.savefig("HandWritten-Digitd-Recognition-With-Scikit-Learn/images/Feature Importance Plot.png", dpi = 300)
    plt.show()

if __name__ == "__main__":
    main()

import os
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import pandas as pd
from tqdm import tqdm

model = ResNet50(weights='imagenet', include_top=False, pooling = 'avg')

def extract_feature(img_path):
    try:
        img = image.load_img(img_path, target_size = (224,224))
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        X = preprocess_input(X)

        feature = model.predict(X, verbose = 0).squeeze()
        return feature/np.linalg.norm(feature)

    except Exception as e:
        return None

features = []
valid_ids = []

df = pd.read_csv(
            'Fashion-Recomendation-System-Visual-Similarities/styles.csv',
            on_bad_lines='skip'
                )

for img_id in tqdm(df['id'].values):
    img_path = os.path.join('Fashion-Recomendation-System-Visual-Similarities/images', f'{img_id}.jpg')
    if os.path.exists(img_path):
        feature_vec = extract_feature(img_path)
        if feature_vec is not None:
            features.append(feature_vec)
            valid_ids.append(img_id)
print(np.array(features).shape)
np.save("Fashion-Recomendation-System-Visual-Similarities/image_features.npy", np.array(features))
np.save("Fashion-Recomendation-System-Visual-Similarities/valid_ids.npy", np.array(valid_ids))
print(f"Feature Extraction Complete")






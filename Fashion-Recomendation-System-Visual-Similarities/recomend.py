import faiss
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image

class RecommendationEngine:
    def __init__(self):
        print("Loading Image features ...")
        self.features = np.load('Fashion-Recomendation-System-Visual-Similarities/image_features.npy').astype('float32')
        
        print("Loading Images....")
        self.image_ids = np.load("Fashion-Recomendation-System-Visual-Similarities/valid_ids.npy")

        df = pd.read_csv(
            'Fashion-Recomendation-System-Visual-Similarities/styles.csv',
            on_bad_lines='skip'
        )
        self.df = df.set_index('id').loc[self.image_ids].reset_index()

        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    def filtered_recommendation(self, query_vector, k=5, article=None, gender=None, excluded_id=None):
        mask = pd.Series(True, index=self.df.index)
        
        # --- FIX 1: Must re-assign mask with `mask = mask & ...` ---
        if article and article != "All":
            mask = mask & (self.df['articleType'].str.lower() == article.lower())
        if gender and gender != "All":
            mask = mask & (self.df['gender'].str.lower() == gender.lower())
        if excluded_id is not None:
            mask = mask & (self.df['id'] != excluded_id)

        filtered_df = self.df[mask]
        if filtered_df.empty:
            return pd.DataFrame()  # Return empty DataFrame instead of string for UI compatibility

        filtered_indices = filtered_df.index.values
        filtered_features = self.features[filtered_indices]
        
        # --- FIX 2: Dimension check on filtered features ---
        dimension = filtered_features.shape[1]
        filtered_index = faiss.IndexFlatIP(dimension)
        filtered_index.add(filtered_features)

        k_search = min(k, len(filtered_df))
        distances, indices = filtered_index.search(query_vector, k_search)
        finalized_indices = filtered_indices[indices[0]]

        return self.df.iloc[finalized_indices]

    def extract_feature_of_single_image(self, img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            X = image.img_to_array(img)
            X = np.expand_dims(X, axis=0)
            X = preprocess_input(X)

            feature = self.model.predict(X, verbose=0).squeeze()
            return (feature / np.linalg.norm(feature)).reshape(1, -1).astype('float32')

        except Exception as e:
            print("Error extracting features:", e)
            return None

    def recommend_using_product_id(self, target_id, k=5, same_category=True, same_gender=True):
        if target_id not in self.df['id'].values:
            return pd.DataFrame()
        
        item_row = self.df[self.df['id'] == target_id].iloc[0]
        idx = self.df[self.df['id'] == target_id].index[0]
        query_vec = self.features[idx].reshape(1, -1)

        article = item_row['articleType'] if same_category else None
        gender = item_row['gender'] if same_gender else None
        excluded_id = target_id

        return self.filtered_recommendation(
            query_vector=query_vec,
            k=k,
            article=article,
            gender=gender,
            excluded_id=excluded_id
        )

    def recommend_by_path(self, img_path, k=5, article=None, gender=None):
        query_vec = self.extract_feature_of_single_image(img_path)
        if query_vec is None:
            return pd.DataFrame()
            
        return self.filtered_recommendation(
            query_vector=query_vec,
            k=k,
            article=article,
            gender=gender
        )
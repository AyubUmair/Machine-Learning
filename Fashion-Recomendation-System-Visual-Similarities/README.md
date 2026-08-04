# 👗 Fashion Recommendation System Using Visual Similarity

A content-based fashion recommendation system that suggests visually similar clothing items using deep learning and computer vision. The application allows users to either upload a fashion image or search using an existing product ID, then recommends similar products based on visual features extracted with ResNet-50 and indexed using FAISS.

---

## 📖 Overview

Traditional recommendation systems rely on purchase history or user interactions. This project instead focuses on **visual similarity**, enabling users to discover fashion products that look alike.

The system extracts deep feature embeddings from clothing images using a pretrained **ResNet-50** model and performs fast nearest-neighbor search using **Facebook AI Similarity Search (FAISS)**.

An interactive **Streamlit** web application provides two recommendation modes:

- 📷 Search by uploading an image
- 🆔 Search using an existing product ID

---

## ✨ Features

- 📷 Upload any clothing image for recommendations
- 🆔 Search recommendations using an existing product ID
- 🧠 Deep feature extraction with pretrained ResNet-50
- ⚡ Fast similarity search using FAISS
- 👕 Filter recommendations by:
  - Gender
  - Article Type
- 🎛 Adjustable number of recommendations
- 🖥 Interactive Streamlit web interface
- 📂 Precomputed feature vectors for faster inference

---

## 🏗 System Architecture

```
                    User Input
                         │
          ┌──────────────┴──────────────┐
          │                             │
     Upload Image                 Product ID
          │                             │
          ▼                             ▼
   Feature Extraction          Load Stored Feature
      (ResNet-50)                      │
          │                            │
          └──────────────┬─────────────┘
                         ▼
                 L2 Normalization
                         ▼
                  FAISS Index Search
                         ▼
             Filter by Gender / Category
                         ▼
          Display Similar Fashion Items
```

---

# 🖼 Dataset

This project uses the **Fashion Product Images (Small)** dataset from Kaggle.

Dataset contains thousands of fashion product images together with metadata such as:

- Product Name
- Gender
- Category
- Article Type
- Base Colour
- Season
- Usage

**Dataset Source**

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

---

# 🧠 Model

### Feature Extractor

- ResNet-50
- ImageNet pretrained weights
- Classification head removed
- Global Average Pooling enabled

Each image is converted into a **2048-dimensional feature vector**.

```python
ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)
```

---

## 🔍 Similarity Search

The extracted features are:

1. L2 normalized
2. Indexed using FAISS
3. Queried using Euclidean distance on normalized vectors (equivalent to cosine similarity)

This enables real-time recommendations even for thousands of images.

---

# 📂 Project Structure

```
Fashion-Recomendation-System-Visual-Similarities/
│
├── app.py                         # Streamlit application
├── recomend.py                    # Recommendation engine
├── feature_extraction.py          # ResNet feature extractor
├── train_features.py              # Generate feature vectors
│
├── image_features.npy
├── image_ids.npy
├── fashion_small.csv
│
├── images/
│   ├── 10001.jpg
│   ├── 10002.jpg
│   └── ...
│
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── home.png
    ├── upload_search.png
    └── product_search.png
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/AyubUmair/Fashion-Recomendation-System-Visual-Similarities.git

cd Fashion-Recomendation-System-Visual-Similarities
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Generate Feature Database

Before launching the application, extract image features.

```bash
python train_features.py
```

This creates

```
image_features.npy
image_ids.npy
```

These files are loaded automatically by the application.

---

# ▶ Run the Application

```bash
streamlit run app.py
```

The application will launch at

```
http://localhost:8501
```

---

# 🖥 Application Interface

## 📷 Search by Image Upload

Users can upload any fashion image.

The application:

- extracts image features
- searches the FAISS index
- applies selected filters
- displays visually similar products

---

## 🆔 Search by Product ID

Users can select an existing product from the dataset.

The application displays:

- Target product
- Similar products
- Optional category matching
- Optional gender matching

---

# 🎛 Available Filters

The sidebar includes:

- Gender Filter
- Article Type Filter
- Number of Recommendations (3–10)

These filters help narrow down recommendation results.

---

# ⚙ Technologies Used

- Python
- TensorFlow
- Keras
- ResNet-50
- FAISS
- Streamlit
- NumPy
- Pandas
- Pillow
- tqdm

---

# 📊 Recommendation Pipeline

```
Input Image
      │
      ▼
Image Preprocessing
      │
      ▼
ResNet-50 Feature Extraction
      │
      ▼
2048-D Feature Vector
      │
      ▼
L2 Normalization
      │
      ▼
FAISS Nearest Neighbor Search
      │
      ▼
Apply User Filters
      │
      ▼
Top-K Similar Fashion Products
```

---

# 🚀 Future Improvements

- Hybrid recommendation system
- Color-based filtering
- Brand-based recommendations
- Text-to-fashion search using CLIP
- Outfit recommendation
- Mobile application
- User favorites
- Recommendation history
- Deploy using Docker
- Cloud deployment on Streamlit Community Cloud

---

# 📚 References

**ResNet**

He, K., Zhang, X., Ren, S., & Sun, J.

Deep Residual Learning for Image Recognition

https://arxiv.org/abs/1512.03385

---

**FAISS**

https://github.com/facebookresearch/faiss

---

**TensorFlow Documentation**

https://www.tensorflow.org/

---

**Kaggle Dataset**

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

---

# 👨‍💻 Author

**A. Umair**

Electronic and Telecommunication Engineering Undergraduate

University of Moratuwa

GitHub: https://github.com/AyubUmair

---

## ⭐ If you found this project useful, consider giving it a star!
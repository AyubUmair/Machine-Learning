import streamlit as st
import os
import pandas as pd
from recomend import RecommendationEngine

st.set_page_config(page_title="Fashion Visual Recommender", layout="wide")

st.title("👗 Visual Fashion Recommendation Engine")

# -------------------------------------------------------------
# LOAD ENGINE WITH CACHING
# -------------------------------------------------------------
@st.cache_resource
def get_engine():
    with st.spinner("Loading AI model and dataset features..."):
        return RecommendationEngine()

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Failed to load Recommendation Engine: {e}")
    st.stop()

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.header("Filters & Settings")
genders = ["All"] + list(engine.df['gender'].dropna().unique())
selected_gender = st.sidebar.selectbox("Gender", genders)

categories = ["All"] + sorted(list(engine.df['articleType'].dropna().unique()))
selected_category = st.sidebar.selectbox("Article Type", categories)

top_k = st.sidebar.slider("Number of Recommendations", 3, 10, 5)

# Helper function to display results in columns
def display_results(results):
    if isinstance(results, pd.DataFrame) and results.empty:
        st.error("No items matched your filter criteria.")
    elif isinstance(results, str):
        st.error(results)
    else:
        cols = st.columns(top_k)
        for idx, (_, row) in enumerate(results.iterrows()):
            img_path = os.path.join(
                "Fashion-Recomendation-System-Visual-Similarities", 
                "images", 
                f"{row['id']}.jpg"
            )
            with cols[idx % top_k]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=row['productDisplayName'], use_container_width=True)
                    st.caption(f"**Category:** {row['articleType']}\n\n**Gender:** {row['gender']}")
                else:
                    st.warning(f"Image missing for ID {row['id']}")

# -------------------------------------------------------------
# TABS FOR SEARCH MODES
# -------------------------------------------------------------
tab_upload, tab_id = st.tabs(["📷 Search by Image Upload", "🆔 Search by Product ID"])

# --- TAB 1: UPLOAD IMAGE ---
with tab_upload:
    st.write("Upload a fashion item image to discover visually similar products.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        temp_img_path = "temp_query.jpg"
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col_left, col_right = st.columns([1, 3])

        with col_left:
            st.subheader("Your Input Image")
            st.image(temp_img_path, use_container_width=True)

        with col_right:
            st.subheader("Visually Similar Items")
            with st.spinner("Analyzing visual features and searching dataset..."):
                results = engine.recommend_by_path(
                    img_path=temp_img_path,
                    k=top_k,
                    article=selected_category,
                    gender=selected_gender
                )
            display_results(results)

# --- TAB 2: PRODUCT ID ---
with tab_id:
    st.write("Select or enter an existing product ID from the dataset.")

    # Get list of valid IDs for dropdown search
    valid_ids = list(engine.df['id'].values)
    selected_id = st.selectbox("Select or Type Product ID:", options=valid_ids)

    # Checkboxes for category matching
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        match_category = st.checkbox("Strictly match category", value=True)
    with col_c2:
        match_gender = st.checkbox("Strictly match gender", value=True)

    if st.button("Get Recommendations"):
        col_left, col_right = st.columns([1, 3])

        # Show target product image
        target_img_path = os.path.join(
            "Fashion-Recomendation-System-Visual-Similarities", 
            "images", 
            f"{selected_id}.jpg"
        )
        
        with col_left:
            st.subheader("Target Product")
            if os.path.exists(target_img_path):
                st.image(target_img_path, use_container_width=True)
                target_row = engine.df[engine.df['id'] == selected_id].iloc[0]
                st.caption(f"**Name:** {target_row['productDisplayName']}")
            else:
                st.warning(f"Image missing for ID {selected_id}")

        with col_right:
            st.subheader("Recommended Similar Products")
            with st.spinner("Searching nearest vectors..."):
                results = engine.recommend_using_product_id(
                    target_id=selected_id,
                    k=top_k,
                    same_category=match_category,
                    same_gender=match_gender
                )
            display_results(results)
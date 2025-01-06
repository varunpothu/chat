import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import gdown
import os

# Set page configuration
st.set_page_config(page_title="Smart Healthcare Chatbot", layout="wide")

# Function to download files from Google Drive
def download_file(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# Preprocess user query
def preprocess_query(query):
    corrected_query = str(TextBlob(query).correct())
    return re.sub(r'[^\w\s]', '', corrected_query.lower()).strip()

# File IDs from Google Drive
file_ids = {
    "cleaned_dataset": "YOUR_FILE_ID_FOR_cleaned_dataset_with_embeddings",
    "recommender_model": "YOUR_FILE_ID_FOR_medicine_recommender",
    "symptom_embeddings": "YOUR_FILE_ID_FOR_symptom_embeddings",
    "data5": "YOUR_FILE_ID_FOR_5.csv"
}

# Paths to download files locally
file_paths = {
    "cleaned_dataset": "cleaned_dataset_with_embeddings.pkl",
    "recommender_model": "medicine_recommender.pkl",
    "symptom_embeddings": "symptom_embeddings.pkl",
    "data5": "5.csv"
}

# Download all required files
for key, file_id in file_ids.items():
    download_file(file_id, file_paths[key])

# Load resources for Disease Q&A
@st.cache_resource
def load_disease_resources():
    with open(file_paths["cleaned_dataset"], "rb") as f:
        df = pickle.load(f)

    mini_lm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    distilroberta_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    bert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    return mini_lm_model, distilroberta_model, bert_model, df

# Load resources for Medicine Recommendation
@st.cache_resource
def load_medicine_resources():
    with open(file_paths["recommender_model"], "rb") as f:
        medicine_recommender = pickle.load(f)
    with open(file_paths["symptom_embeddings"], "rb") as f:
        symptom_embeddings = pickle.load(f)
    data5 = pd.read_csv(file_paths["data5"])
    return symptom_embeddings, medicine_recommender, data5

# Load all resources
mini_lm_model, distilroberta_model, bert_model, df = load_disease_resources()
symptom_embeddings, medicine_recommender, data5 = load_medicine_resources()

# Sidebar for navigation
st.sidebar.header("Navigation")
tabs = ["Disease Q&A", "Medicine Recommendation"]
selected_tab = st.sidebar.selectbox("Choose a Tab", tabs)

if selected_tab == "Disease Q&A":
    st.title("🩺 Disease Q&A Chatbot")
    user_query = st.text_input("Ask a healthcare question:", placeholder="Type your question here...")
    if user_query:
        query_clean = preprocess_query(user_query)
        query_embedding = mini_lm_model.encode(query_clean).reshape(1, -1)
        df['similarity'] = df['mini_lm_embedding'].apply(
            lambda x: cosine_similarity(query_embedding, np.array(x).reshape(1, -1))[0][0]
        )
        top_match = df.loc[df['similarity'].idxmax()]
        st.success(f"**Answer:** {top_match['answer']}")
        st.markdown(f"**Source:** {top_match['source']}")
        st.markdown(f"**Focus Area:** {top_match['focus_area']}")

elif selected_tab == "Medicine Recommendation":
    st.title("💊 Medicine Recommendation Chatbot")
    user_input = st.text_input("Enter symptoms:")
    if user_input:
        recommendations = medicine_recommender.recommend(user_input, top_n=3)
        st.write(recommendations)

# Footer
st.info("💡 For accurate healthcare advice, consult a medical professional.")

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import pickle
import gdown
import joblib
from fuzzywuzzy import process
import os
import re

# Set page configuration
st.set_page_config(page_title="Smart Healthcare Chatbot", layout="wide")

# Function to download files from Google Drive
def download_file(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# Google Drive File IDs and Paths
file_ids = {
    "cleaned_dataset": "11tLsqLqVF3WFLvcoTz0wcs0iYYbncMrH",
    "gru_model": "1VCXEiEADyVz2NLJ8b0HGU8TftpZlFSS8",
    "tokenizer": "122m9vzR4cvsRLC7lxlE8lcIfhVro_EPk",
    "label_encoder": "12C6U60REeFWUEdxbNDOIWSA_hSE0YOGV",
    "processed_data": "12MNsyrMBEylIhC__S0LPV_7cjhUTxFck",
    "symptom_embeddings": "128-LKwh37MMOrIgO5HutE4c3PG5NX76i",
    "data5": "11xhQufvXsTwjb4iKLp4l5S0ube8af5Rs"
}

file_paths = {
    "cleaned_dataset": "cleaned_dataset_with_embeddings.pkl",
    "gru_model": "gru_model.h5",
    "tokenizer": "tokenizer.pkl",
    "label_encoder": "label_encoder.pkl",
    "processed_data": "processed_data.csv",
    "symptom_embeddings": "symptom_embeddings.pkl",
    "data5": "5.csv"
}

# Download all files
for key, file_id in file_ids.items():
    download_file(file_id, file_paths[key])

# Preprocess user query
def preprocess_query(query):
    corrected_query = str(TextBlob(query).correct())
    return re.sub(r'[^\w\s]', '', corrected_query.lower()).strip()

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
    gru_model = load_model(file_paths["gru_model"])
    tokenizer = joblib.load(file_paths["tokenizer"])
    label_encoder = joblib.load(file_paths["label_encoder"])
    processed_data = pd.read_csv(file_paths["processed_data"])
    return gru_model, tokenizer, label_encoder, processed_data

# Load all resources
mini_lm_model, distilroberta_model, bert_model, df = load_disease_resources()
gru_model, tokenizer, label_encoder, processed_data = load_medicine_resources()

# Sidebar for navigation
st.sidebar.header("Navigation")
tabs = ["Disease Q&A", "Medicine Recommendation", "Medicine Details"]
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
    user_input = st.text_input("Enter your symptoms:")
    if user_input:
        def recommend_medicine(symptoms):
            data5 = pd.read_csv(file_paths["data5"])
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(data5['uses'])
            user_vector = vectorizer.transform([symptoms])
            similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
            top_indices = similarity_scores.argsort()[0][-3:][::-1]
            return data5.iloc[top_indices][['name', 'uses']]

        recommendations = recommend_medicine(user_input)
        st.write("### Recommended Medicines:")
        st.write(recommendations)

elif selected_tab == "Medicine Details":
    st.title("🔍 Medicine Details Lookup")

    def correct_medicine_name(input_name):
        matched_name, score = process.extractOne(input_name.lower(), processed_data['medicine_name'].values)
        return matched_name if score > 80 else None

    user_input = st.text_input("Enter Medicine Name:")
    if user_input:
        corrected_name = correct_medicine_name(user_input)
        if corrected_name:
            details = processed_data[processed_data['medicine_name'] == corrected_name].iloc[0]
            st.write("### Medicine Details:")
            for key, value in details.items():
                st.write(f"**{key}:** {value}")
        else:
            st.warning("Medicine not found.")

# Footer
st.info("💡 For accurate healthcare advice, consult a medical professional.")

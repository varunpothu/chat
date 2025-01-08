# Streamlit App for Healthcare Chatbot
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import gdown
import re
import os

# Google Drive File IDs and Paths
file_ids = {
    "gru_model": "1VCXEiEADyVz2NLJ8b0HGU8TftpZlFSS8",
    "tokenizer": "122m9vzR4cvsRLC7lxlE8lcIfhVro_EPk",
    "label_encoder": "12C6U60REeFWUEdxbNDOIWSA_hSE0YOGV",
    "processed_data": "12MNsyrMBEylIhC__S0LPV_7cjhUTxFck",
    "data5": "11xhQufvXsTwjb4iKLp4l5S0ube8af5Rs",
    "cleaned_dataset": "11tLsqLqVF3WFLvcoTz0wcs0iYYbncMrH"
}

file_paths = {
    "gru_model": "gru_model.h5",
    "tokenizer": "tokenizer.pkl",
    "label_encoder": "label_encoder.pkl",
    "processed_data": "processed_data.csv",
    "data5": "5.csv",
    "cleaned_dataset": "cleaned_dataset_with_embeddings.pkl"
}

# Set page configuration
st.set_page_config(page_title="Smart Healthcare Chatbot", layout="wide")

# Function to download file from Google Drive
def download_file_from_gdrive(file_id, output_path):
    try:
        if not os.path.exists(output_path):
            url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
            gdown.download(url, output_path, quiet=False)
    except Exception as e:
        st.error(f"Failed to download file: {output_path}. Please ensure it exists.")
        raise e

# Preprocess user query
def preprocess_query(query):
    corrected_query = str(TextBlob(query).correct())
    return re.sub(r'[^\w\s]', '', corrected_query.lower()).strip()

# Load resources for Disease Q&A
@st.cache_resource
def load_disease_resources():
    download_file_from_gdrive(file_ids["cleaned_dataset"], file_paths["cleaned_dataset"])
    with open(file_paths["cleaned_dataset"], "rb") as f:
        df = pickle.load(f)
    mini_lm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    distilroberta_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    bert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return mini_lm_model, distilroberta_model, bert_model, df

# Load resources for Medicine Recommendation
@st.cache_resource
def load_medicine_resources():
    download_file_from_gdrive(file_ids["data5"], file_paths["data5"])
    data5 = pd.read_csv(file_paths["data5"])
    return data5

# Medicine Details Functionality
def get_medicine_details(medicine_name):
    try:
        processed_data = pd.read_csv(file_paths["processed_data"])
        matched_data = processed_data[processed_data['medicine_name'].str.contains(medicine_name, case=False)]
        if not matched_data.empty:
            details = matched_data.iloc[0]
            return {
                "Medicine Name": details.get('medicine_name', "Not Available"),
                "Uses": details.get('combined_uses', "Not Available"),
                "Side Effects": details.get('combined_side_effects', "Not Available"),
                "Substitutes": details.get('combined_substitutes', "Not Available"),
                "Therapeutic Class": details.get('therapeutic_class', "Not Available"),
                "Image URL": details.get('image_url', "Not Available"),
                "Manufacturer": details.get('manufacturer', "Not Available"),
            }
        else:
            return {"Error": "Medicine not found. Please check the spelling or try another name."}
    except Exception as e:
        return {"Error": f"Failed to retrieve medicine details: {e}"}

# Functions for Medicine Recommendation
def is_emergency(symptoms):
    emergency_symptoms = [
        "chest pain", "severe bleeding", "difficulty breathing", "sudden confusion",
        "loss of consciousness", "heart attack", "stroke", "difficulty speaking"
    ]
    for emergency in emergency_symptoms:
        if emergency in symptoms.lower():
            return True
    return False

def chatbot_response(symptoms, data5):
    if is_emergency(symptoms):
        return "This is an emergency. Please consult a healthcare professional immediately."
    # Placeholder for actual recommendation logic
    return {
        "Medicine Name": "Paracetamol",
        "Uses": "Fever and mild pain relief",
        "Side Effects": "Nausea, rash",
        "Substitutes": "Acetaminophen",
        "Therapeutic Class": "Analgesic",
        "Manufacturer": "Generic"
    }

# Sidebar for navigation
st.sidebar.header("Navigation")
tabs = ["Disease Q&A", "Medicine Recommendation", "Medicine Details"]
selected_tab = st.sidebar.selectbox("Choose a Tab", tabs)

if selected_tab == "Disease Q&A":
    st.info("Loading Disease Q&A resources...")
    mini_lm_model, distilroberta_model, bert_model, df = load_disease_resources()
    embedding_type = st.sidebar.selectbox(
        "Choose Embedding Type:",
        ["mini_lm_embedding", "distilroberta_embedding", "bert_embedding"]
    )
    st.title("🩺 Smart Healthcare Chatbot - Disease Q&A")
    user_query = st.text_input("Ask your healthcare question:", placeholder="Type your question here...")
    if user_query:
        query_clean = preprocess_query(user_query)
        if embedding_type == "mini_lm_embedding":
            query_embedding = mini_lm_model.encode(query_clean).reshape(1, -1)
        elif embedding_type == "distilroberta_embedding":
            query_embedding = distilroberta_model.encode(query_clean).reshape(1, -1)
        elif embedding_type == "bert_embedding":
            query_embedding = bert_model.encode(query_clean).reshape(1, -1)
        df['similarity'] = df[embedding_type].apply(
            lambda x: cosine_similarity(query_embedding, np.array(x).reshape(1, -1))[0][0]
        )
        top_match = df.loc[df['similarity'].idxmax()]
        st.success(f"**Answer:** {top_match['answer']}")
        st.markdown(f"**Source:** {top_match['source']}")
        st.markdown(f"**Focus Area:** {top_match['focus_area']}")

elif selected_tab == "Medicine Recommendation":
    st.info("Loading Medicine Recommendation resources...")
    data5 = load_medicine_resources()
    st.title("💊 Healthcare Medicine Recommendation Chatbot")
    user_input = st.text_input("Enter your symptoms:")
    if user_input:
        response = chatbot_response(user_input, data5)
        if isinstance(response, dict):
            st.write("### Medicine Details:")
            for key, value in response.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write(response)

elif selected_tab == "Medicine Details":
    st.header("🔍 Medicine Details Lookup")
    user_input = st.text_input("Enter Medicine Name:", "")
    if user_input:
        details = get_medicine_details(user_input)
        if "Error" in details:
            st.write(details["Error"])
        else:
            st.write("### Medicine Details:")
            for key, value in details.items():
                if key == "Image URL" and value != "Not Available":
                    st.image(value, caption="Medicine Image")
                else:
                    st.write(f"**{key}:** {value}")

# Footer
st.info("💡 For accurate healthcare advice, consult a medical professional.")

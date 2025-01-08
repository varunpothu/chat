# Streamlit App for Healthcare Chatbot
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import gdown
import re

# Set page configuration
st.set_page_config(page_title="Smart Healthcare Chatbot", layout="wide")

# Preprocess user query
def preprocess_query(query):
    corrected_query = str(TextBlob(query).correct())
    return re.sub(r'[^\w\s]', '', corrected_query.lower()).strip()

# Download the file from Google Drive
def download_file_from_gdrive(file_id, output_path):
    try:
        url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
        gdown.download(url, output_path, quiet=False)
    except Exception as e:
        raise RuntimeError(f"Failed to download file from Google Drive: {e}")

# Load resources for Disease Q&A
@st.cache_resource
def load_disease_resources():
    file_id = "11tLsqLqVF3WFLvcoTz0wcs0iYYbncMrH"  # Replace with your actual file ID
    output_path = "cleaned_dataset_with_embeddings.pkl"
    download_file_from_gdrive(file_id, output_path)

    with open(output_path, "rb") as f:
        df = pickle.load(f)

    mini_lm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    distilroberta_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    bert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    return mini_lm_model, distilroberta_model, bert_model, df

# Load resources for Medicine Recommendation
@st.cache_resource
def load_medicine_resources():
    with open('symptom_embeddings.pkl', 'rb') as f:
        symptom_embeddings = pickle.load(f)
    with open('medicine_recommender.pkl', 'rb') as f:
        medicine_recommender = pickle.load(f)

    data5 = pd.read_csv('5.csv')  # Ensure the file exists in your repo

    return symptom_embeddings, medicine_recommender, data5

# Functions for Medicine Recommendation
def is_emergency(symptoms):
    emergency_symptoms = [
        "chest pain", "severe bleeding", "difficulty breathing", "sudden confusion", "loss of consciousness"
    ]
    for emergency in emergency_symptoms:
        if emergency in symptoms.lower():
            return True
    return False

def chatbot_response(symptoms, medicine_recommender, data5):
    if is_emergency(symptoms):
        return "This is an emergency. Please consult a healthcare professional immediately."

    recommendations = medicine_recommender.recommend(symptoms, top_n=1)
    if recommendations.empty:
        return "No medicines found for the provided symptoms. Please consult a doctor."

    medicine = recommendations.iloc[0]
    side_effects = data5[data5['name'] == medicine['name']]['side_effects'].iloc[0]
    substitutes = data5[data5['name'] == medicine['name']]['substitutes'].iloc[0]
    therapeutic_class = data5[data5['name'] == medicine['name']]['Therapeutic Class'].iloc[0]
    manufacturer = data5[data5['name'] == medicine['name']]['Manufacturer'].iloc[0]

    return {
        "Medicine Name": medicine['name'],
        "Uses": medicine['uses'],
        "Side Effects": side_effects,
        "Substitutes": substitutes,
        "Therapeutic Class": therapeutic_class,
        "Manufacturer": manufacturer
    }

# Load all resources
mini_lm_model, distilroberta_model, bert_model, df = load_disease_resources()
symptom_embeddings, medicine_recommender, data5 = load_medicine_resources()

# Sidebar for navigation
st.sidebar.header("Navigation")
tabs = ["Disease Q&A", "Medicine Recommendation"]
selected_tab = st.sidebar.selectbox("Choose a Tab", tabs)

if selected_tab == "Disease Q&A":
    embedding_type = st.sidebar.selectbox("Choose Embedding Type:", ["mini_lm_embedding", "distilroberta_embedding", "bert_embedding"])
    st.title("🩺 Smart Healthcare Chatbot - Disease Q&A")
    user_query = st.text_input("Ask your healthcare question:")
    if user_query:
        query_clean = preprocess_query(user_query)
        if embedding_type == "mini_lm_embedding":
            query_embedding = mini_lm_model.encode(query_clean).reshape(1, -1)
        elif embedding_type == "distilroberta_embedding":
            query_embedding = distilroberta_model.encode(query_clean).reshape(1, -1)
        elif embedding_type == "bert_embedding":
            query_embedding = bert_model.encode(query_clean).reshape(1, -1)

        df['similarity'] = df[embedding_type].apply(lambda x: cosine_similarity(query_embedding, np.array(x).reshape(1, -1))[0][0])
        top_match = df.loc[df['similarity'].idxmax()]

        st.success(f"**Answer:** {top_match['answer']}")
        st.markdown(f"**Source:** {top_match['source']}")

elif selected_tab == "Medicine Recommendation":
    st.title("💊 Healthcare Medicine Recommendation Chatbot")
    user_input = st.text_input("Enter your symptoms:")
    if user_input:
        response = chatbot_response(user_input, medicine_recommender, data5)
        if isinstance(response, dict):
            st.write("### Medicine Details:")
            for key, value in response.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write(response)

# Footer
st.info("💡 For accurate healthcare advice, consult a medical professional.")

# Streamlit App for Disease Q&A Chatbot
import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import gdown

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

# Load model and embeddings
@st.cache_resource
def load_resources():
    # File ID from Google Drive
    file_id = "11tLsqLqVF3WFLvcoTz0wcs0iYYbncMrH"  # Correct file ID
    output_path = "cleaned_dataset_with_embeddings.pkl"

    # Download the file
    download_file_from_gdrive(file_id, output_path)

    # Load the dataset
    with open(output_path, "rb") as f:
        df = pickle.load(f)

    # Load models
    mini_lm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    distilroberta_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    bert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # Example BERT variant

    return mini_lm_model, distilroberta_model, bert_model, df

mini_lm_model, distilroberta_model, bert_model, df = load_resources()

# Sidebar for navigation
st.sidebar.header("Navigation")
st.sidebar.info(f"**Dataset Size:** {len(df)} questions")

# Embedding selection
embedding_type = st.sidebar.selectbox(
    "Choose Embedding Type:",
    ["mini_lm_embedding", "distilroberta_embedding", "bert_embedding", "bert_embedding_normalized"]
)

# Main Application
st.title("🩺 Smart Healthcare Chatbot")
st.markdown("""Welcome to the healthcare chatbot. Ask any question related to medical topics, and I'll provide the most relevant answer.""")

# User input
user_query = st.text_input("Ask your healthcare question:", placeholder="Type your question here...")
st.markdown("---")

if user_query:
    # Preprocess query
    query_clean = preprocess_query(user_query)

    # Select model based on embedding type
    if embedding_type == "mini_lm_embedding":
        query_embedding = mini_lm_model.encode(query_clean).reshape(1, -1)
    elif embedding_type == "distilroberta_embedding":
        query_embedding = distilroberta_model.encode(query_clean).reshape(1, -1)
    elif embedding_type == "bert_embedding" or embedding_type == "bert_embedding_normalized":
        query_embedding = bert_model.encode(query_clean).reshape(1, -1)

    # Calculate cosine similarity
    df['similarity'] = df[embedding_type].apply(
        lambda x: cosine_similarity(query_embedding, np.array(x).reshape(1, -1))[0][0]
    )
    top_match = df.loc[df['similarity'].idxmax()]

    # Display results
    st.success(f"**Answer:** {top_match['answer']}")
    st.markdown(f"**Source:** {top_match['source']}")
    st.markdown(f"**Focus Area:** {top_match['focus_area']}")
    st.markdown(f"**Similarity Score:** {top_match['similarity']:.2f}")

    # Recommendations
    st.subheader("Related Questions")
    top_related = df.sort_values(by='similarity', ascending=False).head(3)
    for _, row in top_related.iterrows():
        st.write(f"- **{row['question_clean']}**")

# Footer
st.info("💡 For accurate healthcare advice, consult a medical professional.")

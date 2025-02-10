import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set Streamlit page config
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Custom CSS for Netflix-style UI with reddish-brown theme
st.markdown("""
    <style>
    body {
        background-color: #1f0d0d;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #B22222;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px;
    }
    .stSelectbox, .stTextInput {
        border-radius: 8px;
    }
    .movie-box {
        background-color: #3e1f1f;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(255, 69, 0, 0.5);
    }
    .section {
        padding: 20px;
        background-color: #2e0d0d;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🍿 Movie Recommendation System")
st.write("Find your next favorite movie based on what you love!")

# Button to redirect to GitHub for downloading the dataset
st.markdown("""
    <a href="https://github.com/Kailas-lab/Machine_Learning/blob/main/Final%20Project/movies.csv" target="_blank">
        <button style="background-color:#B22222;color:white;padding:10px;border-radius:5px;font-size:16px;">
            📥 Download Movie Dataset
        </button>
    </a>
    """, unsafe_allow_html=True)

# File uploader for user-provided CSV file
dataset = st.file_uploader("📂 Upload your movie dataset (CSV)", type=["csv"])

if dataset:
    movies = pd.read_csv(dataset)
    st.write("### 🎥 Sample Data")
    st.dataframe(movies.head())
else:
    st.warning("⚠️ Please upload a valid movie dataset file.")
    st.stop()

# Ensure dataset has required columns
if 'original_title' not in movies.columns or 'overview' not in movies.columns:
    st.error("❌ The dataset must contain 'original_title' and 'overview' columns.")
    st.stop()

# Movie selection box
selected_movie = st.selectbox("🎬 Choose a movie", movies['original_title'].values)

# Movie recommendation function
def recommend(movie, movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    index = movies[movies['original_title'] == movie].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
    recommended_movies = [movies.iloc[i[0]].original_title for i in similarity_scores]
    return recommended_movies

# Create sections
st.markdown("<div class='section'>", unsafe_allow_html=True)
if st.button("🎥 Get Recommendations"):
    recommendations = recommend(selected_movie, movies)
    st.write("### 🍿 Recommended Movies for You")
    for rec in recommendations:
        st.markdown(f"<div class='movie-box'><strong>{rec}</strong></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Model insights section
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("🔍 Model Insights")
st.write("This recommendation system is powered by **TF-IDF Vectorization** and **Cosine Similarity**, which helps find movies similar to your selection based on their descriptions.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
---
<center>© 2025 Movie Recommendation System | Developed by Kailas M.</center>
""", unsafe_allow_html=True)

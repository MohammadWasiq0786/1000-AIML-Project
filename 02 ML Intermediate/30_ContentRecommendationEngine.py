"""
Project 70. Content recommendation engine
Description:
A content recommendation engine suggests articles, videos, or products to users based on similarity to previously viewed or liked content. In this project, we use TF-IDF to convert article descriptions into vectors, then apply cosine similarity to recommend the most relevant items.

About:
📚 What This Project Demonstrates:
Uses TF-IDF + cosine similarity to match similar text-based content

Ranks recommendations based on semantic relevance

Forms the core of article, video, or product recommender systems
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# Simulated content dataset
content = [
    "Learn Python programming with hands-on projects and challenges.",
    "10 tips for effective machine learning model tuning and evaluation.",
    "Understanding React components, props, and state management.",
    "How to build REST APIs using Flask and deploy them with Docker.",
    "Exploring deep learning with PyTorch: CNNs and RNNs explained.",
    "Getting started with Django: Build your first web application.",
    "Natural Language Processing basics with spaCy and NLTK.",
    "Master SQL for data analysis and reporting in modern systems.",
    "Data visualization using Matplotlib and Seaborn in Python.",
    "Deploying machine learning models using FastAPI and Streamlit."
]
 
# Create DataFrame
df = pd.DataFrame({'Content': content})
 
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Content'])
 
# Choose a content index as query (e.g., article 0)
query_index = 0
query_vector = tfidf_matrix[query_index]
 
# Calculate cosine similarities
similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
 
# Get top 5 recommendations (excluding the query itself)
top_indices = similarities.argsort()[::-1][1:6]
 
# Display recommendations
print(f"\nRecommendations for: '{df.iloc[query_index]['Content']}'\n")
for idx in top_indices:
    print(f"Score: {similarities[idx]:.2f} | {df.iloc[idx]['Content']}")
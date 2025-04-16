"""
Project 352. News recommendation system
Description:
A news recommendation system suggests articles to users based on:

User preferences (e.g., topics they’re interested in)

Content similarity (e.g., similar articles based on keywords, tags, or article content)

In this project, we’ll build a news recommendation system using text similarity techniques (such as TF-IDF or word embeddings) to recommend news articles based on the content.

About:
✅ What It Does:
Uses TF-IDF to convert article text into numerical features, capturing the importance of words in each article

Computes cosine similarity to measure the similarity between articles based on their content

Recommends top similar articles based on content similarity
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate news articles and their content
articles = ['Article1', 'Article2', 'Article3', 'Article4', 'Article5']
content = [
    "Breaking news: A major earthquake has struck the city, causing widespread damage.",
    "Sports update: The football team won the championship in a thrilling final match.",
    "Technology breakthrough: New AI advancements are transforming the industry.",
    "Entertainment news: A new blockbuster movie is breaking records at the box office.",
    "Health update: Scientists have discovered a new treatment for heart disease."
]
 
# 2. Use TF-IDF to convert article content into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(content)
 
# 3. Function to recommend news articles based on content similarity
def news_recommendation(query_article, articles, tfidf_matrix, top_n=3):
    # Get the index of the query article
    query_idx = articles.index(query_article)
    
    # Compute the cosine similarity between the query article and all other articles
    cosine_sim = cosine_similarity(tfidf_matrix[query_idx:query_idx+1], tfidf_matrix).flatten()
    
    # Get the indices of the most similar articles
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]  # Exclude the query article itself (index 0)
    
    recommended_articles = [articles[i] for i in similar_indices]
    return recommended_articles
 
# 4. Recommend articles based on the content similarity of 'Article1'
recommended_articles = news_recommendation('Article1', articles, tfidf_matrix)
print(f"News Recommendations for 'Article1': {recommended_articles}")
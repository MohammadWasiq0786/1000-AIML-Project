"""
Project 322. Content-based filtering
Description:
Content-based filtering recommends items based on the features of the items and a user’s past preferences. The system computes a similarity score between the features of items and the user’s profile, which is created from items they have liked in the past.

We’ll implement a simple content-based filtering system using TF-IDF for text-based features (e.g., movie descriptions, product features).

About:
✅ What It Does:
Simulates movie descriptions as features

Uses TF-IDF to convert descriptions into vectors

Computes cosine similarity between item features

Recommends movies similar to Movie1 based on textual features
"""


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate item features (movie descriptions)
items = ['Movie1', 'Movie2', 'Movie3', 'Movie4', 'Movie5']
descriptions = [
    "Action movie with thrilling sequences",
    "Romantic comedy with funny moments",
    "Action-packed superhero movie",
    "Drama about love and loss",
    "A romantic comedy with unexpected twists"
]
df = pd.DataFrame({'Item': items, 'Description': descriptions})
 
# 2. TF-IDF vectorization of item descriptions
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])
 
# 3. Compute cosine similarity between items
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
 
# 4. Recommend items for a given item (e.g., Movie1)
item_idx = 0  # Movie1
similarities = cosine_sim[item_idx]
recommendations = list(enumerate(similarities))
 
# Sort recommendations based on similarity score
recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
 
# 5. Print recommended items for Movie1
print("Recommendations for Movie1:")
for idx, score in recommendations[1:]:
    print(f"{df['Item'][idx]}: {score:.2f}")
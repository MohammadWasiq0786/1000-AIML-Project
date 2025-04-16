"""
Project 323. Hybrid recommendation system
Description:
A Hybrid Recommendation System combines multiple recommendation approaches to leverage the strengths of each. By merging techniques like:

Collaborative filtering (user-item interactions)

Content-based filtering (item features)

This system can handle:

Cold-start problems (when user/item data is sparse)

Improved accuracy by combining different data sources

In this project, we’ll build a hybrid system that blends collaborative filtering and content-based filtering.

About:
✅ What It Does:
Simulates user-item ratings and item descriptions

Applies collaborative filtering (user-based) and content-based filtering (item descriptions)

Combines both approaches with weighted averaging to generate hybrid recommendations

Recommends items to User1 based on both their ratings and item features
"""


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
 
# 1. Simulate user-item ratings (Collaborative Filtering Data)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
ratings = np.random.randint(1, 6, size=(5, 5))  # Random ratings between 1 and 5
 
# 2. Simulate item features (Content-Based Data)
descriptions = [
    "Action movie with thrilling sequences",
    "Romantic comedy with funny moments",
    "Action-packed superhero movie",
    "Drama about love and loss",
    "A romantic comedy with unexpected twists"
]
 
# DataFrames for Collaborative and Content-based data
ratings_df = pd.DataFrame(ratings, index=users, columns=items)
content_df = pd.DataFrame({'Item': items, 'Description': descriptions})
 
# 3. Collaborative filtering (user-based) - Cosine Similarity between Users
user_similarity = cosine_similarity(ratings_df)
user_similarity_df = pd.DataFrame(user_similarity, index=users, columns=users)
 
# 4. Content-based filtering (item-based) - TF-IDF Similarity between Items
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(content_df['Description'])
item_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
 
# 5. Hybrid recommendation (combine both methods)
# Example: Recommend items for User1 based on both content and collaborative methods
user_idx = 0  # User1
collab_similarities = user_similarity_df['User1'].values
content_similarities = item_similarity[0]  # Content similarity for Item1
 
# Combine the two similarity scores
alpha = 0.5  # Weight for collaborative vs content-based filtering
hybrid_scores = alpha * collab_similarities + (1 - alpha) * content_similarities
 
# 6. Recommend top items for User1
recommendations = list(enumerate(hybrid_scores))
recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
 
# 7. Display recommendations for User1
print("Hybrid Recommendations for User1:")
for idx, score in recommendations[1:]:
    print(f"{items[idx]}: {score:.2f}")
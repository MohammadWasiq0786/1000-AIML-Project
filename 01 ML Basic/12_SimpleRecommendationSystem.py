"""
Project 12. Simple recommendation system
Description:
A Recommendation System suggests items to users based on similarity between their preferences and item characteristics. In this project, we’ll build a content-based filtering system that recommends movies based on their descriptions using cosine similarity on TF-IDF vectors.

About:
What this system does:

Converts movie descriptions into TF-IDF vectors

Uses cosine similarity to find the most similar items

Returns top N recommendations for a given movie title

You can enhance this by:

Adding genres, tags, or user ratings

Switching to collaborative filtering

Building a hybrid model with content + user preferences
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
 
# Sample movie dataset: (Movie Title, Description)
movies = [
    ("The Matrix", "sci-fi action future reality virtual AI"),
    ("John Wick", "action revenge assassin dog"),
    ("Interstellar", "space science drama wormhole time travel"),
    ("Inception", "dream heist subconscious action mind-bending"),
    ("The Notebook", "romance love drama emotional"),
    ("Avengers", "superheroes marvel action battle save world"),
    ("Gravity", "space survival astronaut drama"),
    ("Titanic", "romance tragedy love ship historical"),
    ("The Martian", "space science survival alone Mars"),
    ("Edge of Tomorrow", "time loop alien war action")
]
 
# Extract movie titles and descriptions
titles = [title for title, desc in movies]
descriptions = [desc for title, desc in movies]
 
# Convert descriptions to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(descriptions)
 
# Compute cosine similarity between all movie pairs
cosine_sim = cosine_similarity(tfidf_matrix)
 
# Function to recommend movies based on a given title
def recommend(title, top_n=3):
    if title not in titles:
        print("Movie not found in database.")
        return
    
    idx = titles.index(title)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} recommendations for '{title}':\n")
    for i, score in sim_scores[1:top_n + 1]:
        print(f"{titles[i]} (Similarity: {score:.2f})")
 
# Example usage
recommend("Interstellar")
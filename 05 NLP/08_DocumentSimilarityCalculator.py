"""
Project 168. Document similarity calculator
Description:
A Document Similarity Calculator measures how similar two or more documents are using vector representations. In this project, we use TF-IDF vectorization and cosine similarity to compute the similarity score between multiple text documents, which is useful for plagiarism detection, duplicate detection, and clustering tasks.

About:
🧠 What This Project Demonstrates:
Uses TF-IDF vectorization to represent documents numerically

Computes cosine similarity to measure pairwise document closeness

Identifies the most similar document pair in a collection
"""

# Install if not already: pip install scikit-learn
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
 
# Sample documents (can be articles, paragraphs, etc.)
documents = [
    "AI is changing the world through automation and innovation.",
    "Artificial Intelligence and automation are reshaping industries.",
    "The global economy is impacted by inflation and interest rates.",
    "Machine learning techniques are a part of AI advancements."
]
 
# Convert documents into TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
 
# Compute cosine similarity between documents
similarity_matrix = cosine_similarity(tfidf_matrix)
 
# Display similarity matrix
print("📊 Document Similarity Matrix (Cosine):\n")
for i in range(len(documents)):
    for j in range(len(documents)):
        print(f"Doc {i+1} vs Doc {j+1}: {similarity_matrix[i][j]:.2f}")
    print("-" * 40)
 
# Optional: Find most similar pair
def most_similar_pair(sim_matrix):
    max_score = 0
    pair = (0, 0)
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            if sim_matrix[i][j] > max_score:
                max_score = sim_matrix[i][j]
                pair = (i, j)
    return pair, max_score
 
pair, score = most_similar_pair(similarity_matrix)
print(f"🔍 Most Similar Docs: Doc {pair[0]+1} and Doc {pair[1]+1} → Similarity Score: {score:.2f}")
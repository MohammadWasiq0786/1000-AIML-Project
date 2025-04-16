"""
Project 189. Plagiarism detector
Description:
A Plagiarism Detector checks if a given document or sentence is copied—either partially or fully—from another source by analyzing textual similarity. In this project, we build a basic plagiarism detection system using TF-IDF vectorization and cosine similarity, ideal for comparing essays, reports, or content submissions.

About:
🧠 What This Project Demonstrates:
Uses TF-IDF to represent documents numerically

Computes cosine similarity to assess textual overlap

Flags submissions that exceed a similarity threshold
"""

# Install if not already: pip install scikit-learn
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# Sample documents (could be from a database or file system)
documents = [
    "Machine learning is a field of artificial intelligence that uses statistical techniques to give computers the ability to learn from data.",
    "Artificial intelligence involves training machines to perform tasks that typically require human intelligence.",
    "Machine learning allows computers to learn from data using statistical techniques.",
    "This document contains entirely original content and is not copied."
]
 
# New submission to check for plagiarism
submission = "Machine learning uses statistical methods to enable machines to learn from data."
 
# Combine the submission with existing documents
all_docs = [submission] + documents
 
# Vectorize with TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_docs)
 
# Compute cosine similarity
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
 
# Threshold for flagging plagiarism (e.g., > 0.7 = potentially plagiarized)
threshold = 0.7
 
print("🧠 Plagiarism Detection Results:\n")
for idx, score in enumerate(similarity_scores):
    verdict = "⚠️ Possible Plagiarism" if score > threshold else "✅ Likely Original"
    print(f"Compared to Document {idx+1} → Similarity Score: {score:.2f} → {verdict}")
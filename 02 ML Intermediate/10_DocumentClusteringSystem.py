"""
Project 50. Document clustering system
Description:
Document clustering is used to automatically group similar text documents without predefined labels. This helps in organizing, summarizing, or discovering hidden topics in a corpus. In this project, we build a system that vectorizes documents using TF-IDF and clusters them using K-Means, then visualizes the results.

About:
🧠 What This Project Demonstrates:
Converts documents to numerical vectors using TF-IDF

Groups similar documents using K-Means clustering

Reduces dimensions with PCA for easy 2D plotting
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
 
# Sample documents
documents = [
    "Apple released a new iPhone today.",
    "The stock market saw major gains.",
    "Google announced a new Android update.",
    "Investors are optimistic about tech stocks.",
    "iPhones are selling fast this year.",
    "The economy is recovering from the pandemic.",
    "Samsung's Galaxy phones compete with iPhones.",
    "Financial experts predict more growth in the market.",
    "Android phones have improved camera features.",
    "Inflation rates remain a concern for investors."
]
 
# Step 1: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
 
# Step 2: K-Means Clustering
n_clusters = 3
model = KMeans(n_clusters=n_clusters, random_state=42)
model.fit(X)
labels = model.labels_
 
# Step 3: Reduce dimensions for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())
 
# Step 4: Plot the clustered documents
plt.figure(figsize=(8, 5))
for i in range(n_clusters):
    plt.scatter(X_reduced[labels == i, 0], X_reduced[labels == i, 1], label=f"Cluster {i+1}")
plt.title("Document Clustering with K-Means")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
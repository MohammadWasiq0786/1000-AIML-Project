"""
Project 176. Text clustering implementation
Description:
Text clustering groups similar documents together without labeled data. It’s useful for document categorization, topic detection, and information retrieval. In this project, we use TF-IDF vectorization and K-Means clustering to group text samples (e.g., news snippets or short articles) into similar clusters.

About:
🧠 What This Project Demonstrates:
Vectorizes text using TF-IDF

Clusters documents using K-Means based on semantic similarity

Optionally visualizes document clusters in 2D using PCA
"""

# Install if not already: pip install scikit-learn nltk matplotlib
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk
 
nltk.download('punkt')
 
# Sample documents (can be news snippets, blog posts, etc.)
documents = [
    "AI is transforming the future of healthcare and medicine.",
    "Deep learning and neural networks are driving AI innovations.",
    "The stock market saw a significant decline due to inflation.",
    "Investors are worried about the rising interest rates.",
    "NASA is planning new missions to explore Mars.",
    "SpaceX successfully launched its rocket into orbit.",
]
 
# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
 
# Apply KMeans clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
 
# Display results
print("📊 Cluster Assignments:")
for idx, doc in enumerate(documents):
    print(f"Document {idx+1} → Cluster {labels[idx]}")
 
# Optional: Visualize clusters using PCA
reduced = PCA(n_components=2).fit_transform(X.toarray())
 
plt.figure(figsize=(6, 5))
for i in range(num_clusters):
    plt.scatter(reduced[labels == i, 0], reduced[labels == i, 1], label=f"Cluster {i}")
plt.title("Document Clusters (2D PCA)")
plt.legend()
plt.show()
"""
Project 196. Content recommendation based on text
Description:
A content-based recommender suggests similar content (articles, videos, blog posts, etc.) by analyzing the text features of the content itself. This project uses TF-IDF vectorization and cosine similarity to recommend items similar to a selected piece, based purely on textual content.

About:
🧠 What This Project Demonstrates:
Recommends content using TF-IDF vectors and cosine similarity

Captures semantic closeness based on shared keywords and themes

Demonstrates a content-based filtering approach
"""

# Install if not already: pip install scikit-learn pandas
 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# Sample content data (titles + descriptions)
data = {
    "title": [
        "Introduction to Artificial Intelligence",
        "Deep Learning for Image Recognition",
        "AI in Healthcare: Opportunities and Challenges",
        "Understanding Natural Language Processing",
        "The Future of Autonomous Vehicles"
    ],
    "content": [
        "Artificial intelligence involves machines that can learn, reason, and make decisions.",
        "Deep learning uses neural networks for tasks like image classification and object detection.",
        "Healthcare applications of AI include diagnosis, treatment planning, and patient monitoring.",
        "NLP enables machines to understand and process human language in text or speech.",
        "Self-driving cars rely on AI to perceive their environment and make driving decisions."
    ]
}
 
df = pd.DataFrame(data)
 
# Vectorize content using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["content"])
 
# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)
 
# Function to recommend similar content based on a title
def recommend(title, top_n=3):
    idx = df[df["title"] == title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in sim_scores[1:top_n+1]]
    return df.iloc[top_indices][["title", "content"]]
 
# Recommend similar articles
selected_title = "AI in Healthcare: Opportunities and Challenges"
recommendations = recommend(selected_title)
 
print(f"🧠 Recommendations for: \"{selected_title}\"\n")
for i, row in recommendations.iterrows():
    print(f"🔗 {row['title']}\n   → {row['content'][:80]}...\n")
"""
Project 166. Text summarization using extractive methods
Description:
Extractive summarization selects key sentences from the original text to build a concise summary. This project implements extractive summarization using TF-IDF-based sentence scoring and cosine similarity, ranking the most relevant sentences to form a summary.

About:
🧠 What This Project Demonstrates:
Uses TF-IDF vectorization to represent sentence importance

Applies cosine similarity to measure sentence relevance

Selects top N sentences to form a readable, factual summary
"""

# Install if not already: pip install nltk scikit-learn numpy
 
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
 
nltk.download('punkt')
 
# Sample text for summarization
text = """
Artificial Intelligence (AI) is transforming industries worldwide. 
It is being used in healthcare to detect diseases early, in finance to prevent fraud, 
and in manufacturing to optimize supply chains. AI-powered assistants are now part of daily life. 
Despite its promise, AI raises ethical concerns like bias and job displacement. 
As AI evolves, regulations and responsible development become crucial.
"""
 
# Step 1: Split text into sentences
sentences = sent_tokenize(text)
 
# Step 2: Vectorize sentences using TF-IDF
vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(sentences)
 
# Step 3: Compute similarity matrix
similarity_matrix = cosine_similarity(sentence_vectors)
 
# Step 4: Rank sentences using similarity scores
scores = similarity_matrix.sum(axis=1)
 
# Step 5: Select top-N sentences as summary
N = 3
top_sentence_indices = np.argsort(scores)[-N:][::-1]
summary = [sentences[i] for i in sorted(top_sentence_indices)]
 
# Final output
print("🧠 Extractive Summary:\n")
for sent in summary:
    print("- " + sent)
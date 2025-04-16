"""
Project 167. Topic modeling with LDA
Description:
Topic modeling automatically discovers abstract topics within a collection of documents. In this project, we implement LDA (Latent Dirichlet Allocation) using Gensim, which groups words into topics based on co-occurrence patterns—ideal for summarizing large text corpora like news, research papers, or reviews.

About:
🧠 What This Project Demonstrates:
Preprocesses text for tokenization and stopword removal

Builds a bag-of-words corpus and dictionary

Applies LDA to discover latent topics and top words per topic
"""

# Install if not already: pip install gensim nltk
 
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
 
nltk.download('punkt')
nltk.download('stopwords')
 
# Sample corpus (can be replaced with real documents)
documents = [
    "Artificial Intelligence is transforming business and healthcare.",
    "Machine learning and deep learning are subfields of AI.",
    "Hospitals use AI to predict patient conditions.",
    "Stock market prediction is a hot topic in finance using ML.",
    "AI ethics and bias are growing concerns in technology."
]
 
# Preprocessing
stop_words = set(stopwords.words('english'))
processed_docs = []
 
for doc in documents:
    tokens = word_tokenize(doc.lower())
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words]
    processed_docs.append(filtered)
 
# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
 
# Build LDA model
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=dictionary,
                                   num_topics=3,
                                   random_state=42,
                                   passes=15,
                                   alpha='auto',
                                   per_word_topics=True)
 
# Print topics
print("🧠 Top Topics Discovered:\n")
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic #{idx+1}: {topic}")
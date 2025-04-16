"""
Project 183. Text normalization toolkit
Description:
Text normalization transforms raw text into a consistent and standard format for NLP tasks. It typically includes lowercasing, punctuation removal, stopword removal, stemming or lemmatization, and more. In this project, we create a customizable Python toolkit to normalize text—an essential preprocessing step for all language models and pipelines.

About:
🧠 What This Project Demonstrates:
Converts raw text into a clean, consistent format

Removes stopwords, punctuation, numbers, and applies lemmatization or stemming

Forms a reusable preprocessing function for any NLP pipeline
"""

# Install if not already: pip install nltk
 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
 
# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
 
# Normalization function
def normalize_text(text, use_stemming=False, use_lemmatization=True):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Apply stemming or lemmatization
    if use_stemming:
        processed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    elif use_lemmatization:
        processed_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    else:
        processed_tokens = filtered_tokens
 
    return " ".join(processed_tokens)
 
# Example input
sample_text = "Running, jumping, and swimming are fun activities! But are they productive?"
 
# Normalize
print("🧠 Original Text:\n", sample_text)
normalized = normalize_text(sample_text)
print("\n✅ Normalized Text:\n", normalized)
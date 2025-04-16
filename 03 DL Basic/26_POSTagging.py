"""
Project 106. Part-of-speech tagging
Description:
Part-of-Speech (POS) tagging is the process of labeling each word in a sentence with its grammatical role, such as noun, verb, adjective, etc. This is foundational in NLP for understanding syntax, sentence structure, and downstream tasks like parsing or entity recognition. In this project, we use spaCy to perform POS tagging on a given sentence.

About:
🧠 What This Project Demonstrates:
Tags each word in a sentence with its part of speech

Helps in understanding sentence structure and grammar

Uses spaCy's built-in linguistic pipeline with excellent performance
"""

# Install if not already: pip install spacy
# Download the English model: python -m spacy download en_core_web_sm
 
import spacy
 
# Load English NLP model
nlp = spacy.load("en_core_web_sm")
 
# Sample sentence
text = "The quick brown fox jumps over the lazy dog near the riverbank."
 
# Process text
doc = nlp(text)
 
# Print each word with its POS tag and explanation
print("Word\t\tPOS\t\tExplanation")
print("-" * 40)
for token in doc:
    print(f"{token.text:<10}\t{token.pos_:<10}\t{spacy.explain(token.pos_)}")
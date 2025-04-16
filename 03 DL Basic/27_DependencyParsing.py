"""
Project 107. Dependency parsing implementation
Description:
Dependency parsing analyzes the grammatical structure of a sentence by establishing relationships between "head" words and words that modify those heads. In this project, we use spaCy to perform dependency parsing and visualize how different words in a sentence relate to one another.

About:
🧠 What This Project Demonstrates:
Identifies grammatical dependencies (e.g., subject, object, modifiers)

Reveals the syntactic structure of complex sentences

Uses spaCy’s dependency parser for fast and accurate results
"""

# Install if not already: pip install spacy
# Download the English model: python -m spacy download en_core_web_sm
 
import spacy
from spacy import displacy
 
# Load English language model
nlp = spacy.load("en_core_web_sm")
 
# Input sentence
text = "The little girl saw a dog chasing a squirrel in the park."
 
# Process the sentence
doc = nlp(text)
 
# Print tokens with their dependency relationships
print(f"{'Token':<15} {'Dep':<15} {'Head':<15} {'POS':<10}")
print("-" * 60)
for token in doc:
    print(f"{token.text:<15} {token.dep_:<15} {token.head.text:<15} {token.pos_:<10}")
 
# Visualize dependency tree in the browser
displacy.render(doc, style="dep", jupyter=False)
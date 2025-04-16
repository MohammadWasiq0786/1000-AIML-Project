"""
Project 105. Named entity recognition system
Description:
Named Entity Recognition (NER) is the task of identifying and classifying entities in text into predefined categories such as person names, organizations, locations, dates, and more. In this project, we build a simple NER system using the spaCy library, which provides pre-trained models out of the box.

About:
🧠 What This Project Demonstrates:
Extracts real-world entities from unstructured text

Uses pre-trained spaCy models for accurate recognition

Identifies entities like PERSON, ORG, GPE, DATE, etc.
"""

# Install if not already: pip install spacy
# Download the English model: python -m spacy download en_core_web_sm
 
import spacy
from spacy import displacy
 
# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
 
# Sample text
text = "Barack Obama was born in Hawaii and served as the 44th President of the United States. He studied at Harvard University."
 
# Process text
doc = nlp(text)
 
# Print named entities
print("Named Entities:\n")
for ent in doc.ents:
    print(f"{ent.text:<30} ➤ {ent.label_}")
 
# Optional: Visualize entities in a browser
displacy.render(doc, style="ent", jupyter=False)
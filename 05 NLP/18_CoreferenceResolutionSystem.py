"""
Project 178. Coreference resolution system
Description:
Coreference resolution is the task of identifying when two or more expressions in a text refer to the same entity (e.g., resolving “he” to “John”). It’s essential for improving context understanding in dialogue systems, summarizers, and information extraction tools. In this project, we use Hugging Face’s coreferee or SpaCy with neuralcoref to resolve pronouns and noun phrases.

About:
🧠 What This Project Demonstrates:
Identifies co-referring expressions (e.g., "John", "He", "his")

Resolves pronouns to their antecedents for better semantic understanding

Uses transformer-based NLP pipelines for high accuracy
"""

# Install dependencies:
# pip install spacy coreferee
# python -m spacy download en_core_web_trf
# python -m coreferee install en
 
import spacy
import coreferee
 
# Load SpaCy transformer-based English model with coreferee
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe('coreferee')
 
# Sample text with pronouns and repeated entities
text = """
John is a software engineer. He works at Google. 
Every day, John rides his bike to the office. His colleagues admire him for his dedication.
"""
 
# Process the text
doc = nlp(text)
 
# Display coreference chains
print("🧠 Coreference Resolution Results:\n")
 
if doc._.has_coref:
    for chain in doc._.coref_chains:
        print("🔗 Coreference Chain:")
        for mention in chain:
            print(f" - {mention}")
        print()
else:
    print("⚠️ No coreferences found.")
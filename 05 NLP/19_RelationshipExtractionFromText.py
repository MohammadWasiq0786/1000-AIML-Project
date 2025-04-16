"""
Project 179. Relationship extraction from text
Description:
Relationship extraction identifies and extracts meaningful semantic relationships (e.g., works_for, located_in, founded_by) between entities in unstructured text. This is a crucial step in building knowledge graphs, search engines, and question answering systems. In this project, we extract subject-relation-object triples using SpaCy's dependency parsing.

About:
🧠 What This Project Demonstrates:
Uses SpaCy's dependency parser to identify grammatical roles

Extracts subject-verb-object (SVO) triplets from natural text

Builds foundational relational knowledge from unstructured data
"""

# Install if not already: pip install spacy
# And download model: python -m spacy download en_core_web_sm
 
import spacy
 
# Load English SpaCy model
nlp = spacy.load("en_core_web_sm")
 
# Sample text for relationship extraction
text = """
Elon Musk founded SpaceX. He is also the CEO of Tesla. 
Microsoft was founded by Bill Gates. Google is headquartered in California.
"""
 
# Process the document
doc = nlp(text)
 
# Function to extract subject-verb-object triples
def extract_relations(doc):
    relations = []
    for sent in doc.sents:
        subject = ''
        object_ = ''
        verb = ''
        for token in sent:
            if "subj" in token.dep_:
                subject = token.text
            if "obj" in token.dep_:
                object_ = token.text
            if token.pos_ == "VERB":
                verb = token.lemma_
        if subject and verb and object_:
            relations.append((subject, verb, object_))
    return relations
 
# Extract and display relations
triples = extract_relations(doc)
 
print("🧠 Extracted Relationship Triples:\n")
for subj, rel, obj in triples:
    print(f"🔗 ({subj}, {rel}, {obj})")
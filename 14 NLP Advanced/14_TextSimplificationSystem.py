"""
Project 534: Text Simplification System
Description:
Text simplification aims to rewrite complex text in simpler language while preserving its original meaning. This task is especially useful for making content more accessible, particularly for people with language barriers or learning disabilities. In this project, we will use a pre-trained transformer model for simplifying a given text.
"""

from transformers import pipeline
 
# 1. Load pre-trained text simplification model
simplifier = pipeline("text2text-generation", model="t5-small")
 
# 2. Provide a complex sentence to simplify
complex_text = "The implementation of the new system was designed to optimize the overall operational efficiency of the organization."
 
# 3. Use the model to simplify the text
simplified_text = simplifier(complex_text, max_length=100, num_return_sequences=1)
 
# 4. Display the simplified text
print(f"Simplified Text: {simplified_text[0]['generated_text']}")
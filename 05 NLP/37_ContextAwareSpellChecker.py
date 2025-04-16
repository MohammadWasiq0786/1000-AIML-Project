"""
Project 197. Context-aware spell checker
Description:
A Context-Aware Spell Checker not only corrects misspelled words but also considers the context to suggest the right correction (e.g., "there" vs. "their"). Unlike basic spell checkers, it uses language models or contextual embeddings to understand sentence structure. In this project, we use transformers with a fill-mask pipeline to suggest contextually appropriate corrections.

About:
🧠 What This Project Demonstrates:
Identifies and replaces misused but correctly spelled words using context

Uses a pretrained BERT model to generate likely word candidates

Suggests top alternatives ranked by confidence
"""

# Install if not already: pip install transformers
 
from transformers import pipeline
 
# Load a BERT-based fill-mask model
corrector = pipeline("fill-mask", model="bert-base-uncased")
 
# Function to simulate context-aware correction
def context_correct(sentence, incorrect_word):
    masked_sentence = sentence.replace(incorrect_word, "[MASK]")
    predictions = corrector(masked_sentence)
 
    print(f"🔍 Original Sentence: {sentence}")
    print(f"🧠 Correcting: {incorrect_word}\n")
    print("✅ Suggestions based on context:")
    for p in predictions[:5]:
        print(f"- {p['token_str']} (confidence: {p['score']:.2f})")
 
# Example usage
sentence = "She went to the sea to meat her friend."
incorrect_word = "meat"  # Misused homophone
context_correct(sentence, incorrect_word)
"""
Project 536: Grammatical Error Correction
Description:
Grammatical error correction involves automatically identifying and fixing grammatical errors in text. This can be applied to written text in various domains, including essays, emails, and social media posts. In this project, we will use a transformer model for correcting grammatical mistakes in a given sentence.
"""

from transformers import pipeline
 
# 1. Load pre-trained model for grammatical error correction
grammar_corrector = pipeline("text2text-generation", model="t5-base")
 
# 2. Provide a sentence with grammatical errors
incorrect_text = "She don't like going to the park on weekends."
 
# 3. Use the model to correct grammatical errors
corrected_text = grammar_corrector(f"correct grammar: {incorrect_text}", max_length=100, num_return_sequences=1)
 
# 4. Display the corrected text
print(f"Corrected Text: {corrected_text[0]['generated_text']}")
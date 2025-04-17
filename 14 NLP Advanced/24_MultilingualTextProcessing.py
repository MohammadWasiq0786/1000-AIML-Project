"""
Project 544: Multilingual Text Processing
Description:
Multilingual text processing involves handling text data in multiple languages, including tasks like translation, text classification, and summarization. In this project, we will demonstrate multilingual text processing using a pre-trained transformer model such as mBERT or XLM-R, capable of processing text in various languages.
"""

from transformers import pipeline
 
# 1. Load pre-trained multilingual model for text classification (using XLM-R)
classifier = pipeline("zero-shot-classification", model="xlm-roberta-base")
 
# 2. Provide a multilingual text (e.g., English and Spanish)
text_en = "The economy is growing rapidly."
text_es = "La economía está creciendo rápidamente."
 
# 3. Define candidate labels for classification
candidate_labels = ["economy", "sports", "technology", "politics"]
 
# 4. Classify the English text
result_en = classifier(text_en, candidate_labels)
# Classify the Spanish text
result_es = classifier(text_es, candidate_labels)
 
# 5. Display the results
print(f"English Text Classification: {result_en['labels'][0]} with score {result_en['scores'][0]:.2f}")
print(f"Spanish Text Classification: {result_es['labels'][0]} with score {result_es['scores'][0]:.2f}")
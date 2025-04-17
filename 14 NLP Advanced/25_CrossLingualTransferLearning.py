"""
Project 545: Cross-lingual Transfer Learning
Description:
Cross-lingual transfer learning involves leveraging a model trained in one language and applying it to other languages. This enables NLP models to work across languages, even with limited data in the target language. In this project, we will demonstrate cross-lingual transfer learning using a multilingual transformer model (e.g., XLM-R or mBERT) to classify text in a language different from the one it was trained on.
"""

from transformers import pipeline
 
# 1. Load pre-trained multilingual model for text classification (using XLM-R)
classifier = pipeline("zero-shot-classification", model="xlm-roberta-base")
 
# 2. Provide a French text (target language) for classification
text_fr = "L'économie mondiale continue de croître malgré les défis économiques."
 
# 3. Define candidate labels for classification
candidate_labels = ["economy", "sports", "technology", "politics"]
 
# 4. Classify the French text using cross-lingual transfer
result_fr = classifier(text_fr, candidate_labels)
 
# 5. Display the result
print(f"French Text Classification: {result_fr['labels'][0]} with score {result_fr['scores'][0]:.2f}")
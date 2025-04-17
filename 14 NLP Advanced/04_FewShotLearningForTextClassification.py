"""
Project 524: Few-Shot Learning for Text Classification
Description:
Few-shot learning enables a model to learn and generalize from a small number of training examples. In this project, we will apply few-shot learning for text classification using transformers like GPT-3 or T5, allowing the model to classify text into categories based on only a few labeled examples.
"""


from transformers import pipeline
 
# 1. Load the text classification pipeline (few-shot learning)
classifier = pipeline("zero-shot-classification")
 
# 2. Provide the text to be classified
text = "The company just launched a new AI product for healthcare."
 
# 3. Define candidate labels for classification
candidate_labels = ["technology", "healthcare", "finance", "politics"]
 
# 4. Perform the classification with few-shot learning (zero-shot)
result = classifier(text, candidate_labels)
 
# 5. Display the result
print(f"Predicted Category: {result['labels'][0]} with score {result['scores'][0]:.2f}")
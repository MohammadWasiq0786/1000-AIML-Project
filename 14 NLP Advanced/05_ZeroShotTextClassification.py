"""
Project 525: Zero-Shot Text Classification
Description:
Zero-shot text classification allows the model to classify text into categories without requiring any training data for those categories. Instead, the model uses pre-trained knowledge and is able to classify text into user-defined labels even if it has never seen examples of those categories. We will use a zero-shot classification pipeline from Hugging Face's transformers for this task.
"""

from transformers import pipeline
 
# 1. Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")
 
# 2. Provide the text to be classified
text = "The stock market continues to show strong growth despite inflation concerns."
 
# 3. Define candidate labels for classification
candidate_labels = ["economy", "finance", "politics", "sports", "technology"]
 
# 4. Perform zero-shot classification
result = classifier(text, candidate_labels)
 
# 5. Display the result
print(f"Predicted Category: {result['labels'][0]} with score {result['scores'][0]:.2f}")
"""
Project 558: Explainable NLP Models
Description:
Explainability in NLP models refers to the ability to understand and interpret the decisions made by a machine learning model. This is particularly important for high-stakes applications such as healthcare, finance, and legal systems, where model decisions need to be transparent. In this project, we will use a pre-trained model and techniques like attention visualization or LIME to provide explanations for the model's predictions.
"""

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import matplotlib.pyplot as plt
 
# 1. Load pre-trained model and tokenizer for sequence classification
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
 
# 2. Tokenize a sample text for explanation
text = "The quick brown fox jumped over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")
 
# 3. Get model's attention weights for explanation
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # Attention from each layer
 
# 4. Visualize attention weights (e.g., for the first layer and head)
attention_weights = attentions[0][0]  # First layer, first head
attention_matrix = attention_weights.sum(dim=0).cpu().numpy()
 
# 5. Plot the attention weights
plt.figure(figsize=(10, 8))
plt.imshow(attention_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Attention Visualization for Token Pairing")
plt.xticks(range(len(inputs.tokens())), inputs.tokens(), rotation=90)
plt.yticks(range(len(inputs.tokens())), inputs.tokens())
plt.show()
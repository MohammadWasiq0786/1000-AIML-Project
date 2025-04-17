"""
Project 742: Interpretable NLP Models
Description:
Interpretable NLP models are designed to make the decision-making process of natural language processing models more transparent and understandable to humans. These models aim to explain how different parts of the input (e.g., words, phrases, sentences) contribute to the final prediction, allowing for better trust and accountability. In this project, we will create an interpretable NLP model using BERT for a classification task. We will visualize and interpret the model's predictions using techniques like attention visualization, SHAP, and LIME.

Explanation:
Dataset Loading: We load the 20 Newsgroups dataset, which is a text classification dataset. It consists of 20 different categories of news articles. This dataset is useful for testing NLP models on real-world tasks.

Preprocessing: The text data is preprocessed using the BERT tokenizer to convert the text into tokenized input sequences that BERT can understand.

Model and Tokenizer: We load the pre-trained BERT model for sequence classification and the corresponding tokenizer from Hugging Face. The BERT model is fine-tuned for text classification tasks.

Prediction Pipeline: A text classification pipeline is created using the BERT model and tokenizer. This pipeline allows us to make predictions on text inputs.

Interpreting with SHAP: We use SHAP to explain the model's predictions. The SHAP values show the contribution of each token to the model’s prediction. We visualize the SHAP summary plot to understand which parts of the text are most important for the model's decision.

Interpreting with LIME: We use LIME (Local Interpretable Model-agnostic Explanations) to explain a single prediction. LIME perturbs the input and observes how the model's output changes, helping us understand the model's local decision boundaries. We visualize the explanation for the first text in the test set.

By using these interpretability techniques, we gain insights into the behavior of the model and can understand which words or phrases the model relies on when making predictions. This is crucial for making NLP models more transparent and understandable.
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import torch
 
# 1. Load the dataset (20 Newsgroups for text classification)
def load_dataset():
    """
    Load the 20 Newsgroups dataset for text classification.
    """
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    return newsgroups.data, newsgroups.target, newsgroups.target_names
 
# 2. Preprocess the dataset and tokenize
def preprocess_data(texts, tokenizer):
    """
    Tokenize the input texts using BERT's tokenizer.
    """
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    return encodings
 
# 3. Load BERT model and tokenizer
def load_model_and_tokenizer():
    """
    Load pre-trained BERT model for sequence classification and its tokenizer.
    """
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=20)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer
 
# 4. Create the BERT pipeline for prediction
def create_prediction_pipeline(model, tokenizer):
    """
    Create a prediction pipeline using the BERT model for sequence classification.
    """
    nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return nlp_pipeline
 
# 5. Explain predictions using SHAP
def explain_with_shap(model, tokenizer, texts, labels):
    """
    Use SHAP to explain the predictions made by the BERT model.
    """
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer(texts)
    shap.summary_plot(shap_values, texts)
 
# 6. Explain predictions using LIME
def explain_with_lime(model, tokenizer, texts):
    """
    Use LIME to explain the predictions made by the BERT model.
    """
    explainer = LimeTextExplainer(class_names=[str(i) for i in range(20)])
    
    # Wrap the model's prediction function for LIME
    def predict_proba(texts):
        encodings = preprocess_data(texts, tokenizer)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
        return torch.softmax(logits, dim=1).numpy()
 
    # Explain one prediction
    explanation = explainer.explain_instance(texts[0], predict_proba)
    
    # Visualize the explanation
    explanation.show_in_notebook()
 
# 7. Example usage
texts, labels, target_names = load_dataset()
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
 
# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer()
 
# Create the prediction pipeline
nlp_pipeline = create_prediction_pipeline(model, tokenizer)
 
# Example prediction for one text
example_text = X_test[0]
prediction = nlp_pipeline(example_text)
print(f"Predicted class: {target_names[prediction[0]['label']]}")
 
# Explain predictions with SHAP
explain_with_shap(model, tokenizer, X_test[:100], y_test[:100])
 
# Explain predictions with LIME
explain_with_lime(model, tokenizer, X_test[:10])
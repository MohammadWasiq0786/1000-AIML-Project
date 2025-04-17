"""
Project 750: Human-AI Collaboration Frameworks
Description:
Human-AI collaboration frameworks are systems designed to leverage the strengths of both human intelligence and artificial intelligence. The goal is to create workflows where humans and AI work together, complementing each other's capabilities. For example, AI can handle repetitive tasks, while humans provide creativity, judgment, and domain expertise. In this project, we will design a framework for human-AI collaboration, where AI models assist humans by providing decision support, and humans can intervene to refine or adjust the AI’s output.

Explanation:
Dataset Loading: We load the 20 Newsgroups dataset for text classification, which contains 20 different categories of newsgroup posts. We use this dataset for the AI-human collaboration task.

Model Training (AI Component): We use a pre-trained BART model from Hugging Face’s transformers library, specifically the zero-shot classification pipeline, to classify the text into one of the predefined labels (20 newsgroups categories).

Human-AI Collaboration Loop: The human_ai_collaboration() function simulates the process where the AI suggests a label for each text input, and the human user can either accept the suggestion, modify it, or provide their own label. In practice, this would involve more interaction with the human user, but here we simulate it with text inputs.

Accuracy Evaluation: The evaluate_model_accuracy() function calculates the accuracy of the model after the human feedback. The accuracy is determined by comparing the model's suggestions (with human modifications) to the actual labels.

Interactive Feedback Simulation: In the human_ai_collaboration() function, the AI first provides a label suggestion. The user can accept it by typing 'y', or they can reject it and provide their own label.

This project demonstrates how a Human-AI collaboration system can improve model performance by incorporating human feedback into the learning loop. The AI assists with tasks like text classification, but the human user ensures that the final output is correct and relevant.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
 
# 1. Load the dataset (20 Newsgroups for text classification)
def load_dataset():
    """
    Load the 20 Newsgroups dataset for text classification.
    """
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    return newsgroups.data, newsgroups.target, newsgroups.target_names
 
# 2. Train the model (Pre-trained Hugging Face transformer for text classification)
def train_ai_model():
    """
    Load a pre-trained transformer model for text classification (e.g., BERT).
    """
    classifier = pipeline('zero-shot-classification', model="facebook/bart-large-mnli")
    return classifier
 
# 3. Human-AI collaboration loop: AI suggests labels, human reviews
def human_ai_collaboration(classifier, texts, target_labels, labels):
    """
    The AI suggests labels for the texts, and the human can accept, modify, or reject suggestions.
    """
    suggestions = []
    for text in texts:
        # AI makes a suggestion
        ai_suggestion = classifier(text, candidate_labels=labels)
        suggested_label = ai_suggestion['labels'][0]  # Most probable label
 
        # Human feedback (In practice, this would be an interactive step)
        print(f"AI Suggestion: {suggested_label}")
        human_feedback = input("Accept suggestion (y/n)? (Type 'n' to provide your own label): ").strip().lower()
 
        if human_feedback == 'y':
            suggestions.append(suggested_label)
        else:
            # Human provides their own label (simulating)
            human_label = input(f"Provide label from {labels}: ").strip()
            suggestions.append(human_label)
    
    return suggestions
 
# 4. Evaluate model accuracy after human feedback
def evaluate_model_accuracy(suggestions, target_labels):
    """
    Evaluate the accuracy of the AI model with human feedback.
    """
    accuracy = accuracy_score(target_labels, suggestions)
    print(f"Accuracy after human feedback: {accuracy:.4f}")
 
    return accuracy
 
# 5. Example usage
texts, target_labels, target_names = load_dataset()
 
# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(target_labels)
 
# Train the AI model (using a pre-trained transformer model)
classifier = train_ai_model()
 
# Set the candidate labels (from the target names)
labels = target_names
 
# Run the human-AI collaboration loop
suggestions = human_ai_collaboration(classifier, texts[:10], y_encoded[:10], labels)
 
# Evaluate the performance of the human-AI collaboration system
evaluate_model_accuracy(suggestions, y_encoded[:10])
"""
Project 185. Emotion detection in text
Description:
Emotion detection goes beyond basic sentiment analysis by identifying specific emotions expressed in text, such as joy, anger, sadness, fear, surprise, or love. This project implements emotion classification using a pretrained deep learning model like j-hartmann/emotion-english-distilroberta-base from Hugging Face.

About:
🧠 What This Project Demonstrates:
Uses a distilRoBERTa-based model fine-tuned for emotion recognition

Identifies emotions like joy, sadness, anger, love, fear, surprise

Provides probabilistic confidence scores for each emotion label
"""

# Install if not already: pip install transformers torch
 
from transformers import pipeline
 
# Load a pretrained emotion classifier
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
 
# Sample texts
texts = [
    "I can't believe I got the job! This is amazing!",
    "I'm so sad and tired of everything lately.",
    "That movie scared the life out of me.",
    "I'm furious about how they treated me.",
    "You always make me feel loved and appreciated."
]

print("🧠 Emotion Detection Results:\n")
 
for text in texts:
    result = emotion_classifier(text)[0]  # Get top emotion scores
    top_emotion = max(result, key=lambda x: x['score'])
    print(f"💬 Text: {text}")
    print(f"🔍 Detected Emotion: {top_emotion['label']} (Confidence: {top_emotion['score']:.2f})\n")
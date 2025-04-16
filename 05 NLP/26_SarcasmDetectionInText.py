"""
Project 186. Sarcasm detection in text
Description:
Sarcasm detection identifies when text expresses irony—often saying the opposite of what is meant. This is a challenging NLP task because sarcastic sentences may appear positive on the surface but are negative in context. In this project, we use a pretrained transformer model fine-tuned for sarcasm detection.

About:
🧠 What This Project Demonstrates:
Uses a fine-tuned transformer model for sarcasm detection

Handles subtle tone shifts common in tweets, reviews, and dialogue

Classifies text as sarcastic or not sarcastic, with confidence scores
"""

# Install if not already: pip install transformers torch
 
from transformers import pipeline
 
# Load a sarcasm detection model (fine-tuned RoBERTa or similar)
sarcasm_detector = pipeline(
    "text-classification",
    model="mrm8488/t5-base-finetuned-sarcasm-twitter",
    truncation=True,
    max_length=128
)
 
# Example sentences (some sarcastic, some not)
texts = [
    "Oh great, another Monday morning. Just what I needed!",
    "I love waiting in line for 3 hours. Best day ever.",
    "Thank you so much for your help, really appreciate it!",
    "Wow, what a genius idea. Who would’ve thought?",
    "I’m honestly happy you showed up today. Thanks!"
]
 
print("🧠 Sarcasm Detection Results:\n")
 
for text in texts:
    result = sarcasm_detector(text)[0]
    label = result['label']
    confidence = result['score']
    print(f"💬 Text: {text}")
    print(f"🔎 Prediction: {label} (Confidence: {confidence:.2f})\n")
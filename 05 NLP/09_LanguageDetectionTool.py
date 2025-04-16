"""
Project 169. Language detection tool
Description:
A language detection tool automatically identifies the language of a given piece of text. This project uses a pretrained language identification model from langdetect (or langid), which supports dozens of languages and can be extended to route text to language-specific NLP pipelines.

About:
🧠 What This Project Demonstrates:
Uses langdetect to identify language codes (e.g., 'en', 'fr', 'de')

Detects language from short or long text inputs

Provides confidence scores for language predictions
"""

# Install if not already: pip install langdetect
 
from langdetect import detect, detect_langs
 
# Sample texts in different languages
texts = [
    "This is an English sentence.",
    "Ceci est une phrase française.",
    "Dies ist ein deutscher Satz.",
    "これは日本語の文章です。",
    "Esto es una oración en español.",
    "Это предложение на русском языке."
]
 
print("🧠 Detected Languages:\n")
for i, text in enumerate(texts, start=1):
    language = detect(text)
    confidence = detect_langs(text)[0]
    print(f"Text {i}: {text}")
    print(f"🔍 Detected Language: {language} (Confidence: {confidence.prob:.2f})\n")
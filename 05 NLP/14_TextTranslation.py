"""
Project 174. Text translation system
Description:
A text translation system converts text from one language to another using pretrained models. In this project, we build a simple translator using Hugging Face Transformers with the MarianMT or M2M100 models, allowing translations between multiple language pairs.

About:
🧠 What This Project Demonstrates:
Loads a pretrained translation model (MarianMT)

Performs English-to-French translation using Hugging Face Transformers

Uses tokenizer + model pipeline to produce the translation
"""

# Install if not already: pip install transformers sentencepiece
 
from transformers import MarianMTModel, MarianTokenizer
 
# Choose a language pair model: English to French (you can change this)
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
 
# Input sentence in English
text = "Artificial Intelligence is transforming the world."
 
# Tokenize and translate
tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
translated = model.generate(**tokens)
output = tokenizer.decode(translated[0], skip_special_tokens=True)
 
print("🗣️ Original (English):", text)
print("🌍 Translated (French):", output)
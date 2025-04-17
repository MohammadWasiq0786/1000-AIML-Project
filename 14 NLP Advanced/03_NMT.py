"""
Project 523: Neural Machine Translation
Description:
Neural machine translation (NMT) involves using deep learning models to translate text from one language to another. In this project, we will use a pre-trained transformer model (e.g., MarianMT from Hugging Face) to perform neural machine translation for translating text from one language to another.
"""

from transformers import MarianMTModel, MarianTokenizer
 
# 1. Load the pre-trained MarianMT model and tokenizer for English to French translation
model_name = 'Helsinki-NLP/opus-mt-en-fr'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
 
# 2. Provide the text to be translated
text = "Hello, how are you today?"
 
# 3. Tokenize the input text
tokens = tokenizer(text, return_tensors="pt", padding=True)
 
# 4. Translate the text
translated = model.generate(**tokens)
 
# 5. Decode and display the translated text
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
print(f"Translated Text: {translated_text}")
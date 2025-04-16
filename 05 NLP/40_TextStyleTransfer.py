"""
Project 200. Text style transfer implementation
Description:
Text Style Transfer rewrites a sentence to preserve its meaning while changing its style — for example, converting formal language to informal, positive to negative, or modern to Shakespearean. In this project, we implement style transfer using a pretrained transformer model like T5 or GPT-2 fine-tuned for stylistic transformations.

About:
🧠 What This Project Demonstrates:
Generates alternative phrasings of the same content with different tones/styles

Uses sequence-to-sequence models like T5 for controlled text generation

Applies in chatbots, writing assistants, or persona-based storytelling
"""

# Install if not already: pip install transformers torch
 
from transformers import pipeline
 
# Load a style transfer model
# Model options: some are fine-tuned T5s, GPT-2, or paraphrasers
# For demo, we'll use a general text2text generation model (T5 paraphrasing variant)
style_transfer = pipeline("text2text-generation", model="prithivida/parrot_paraphraser_on_T5")
 
# Sample sentences and style instructions
input_texts = [
    "The presentation was highly professional and informative.",
    "I am extremely disappointed with the product I received."
]
 
print("🧠 Text Style Transfer Examples:\n")
 
for text in input_texts:
    print(f"🔤 Original: {text}")
    output = style_transfer(f"paraphrase: {text}", max_length=50, num_return_sequences=2)
    for i, o in enumerate(output):
        print(f"  ✍️ Style Variant {i+1}: {o['generated_text']}")
    print()
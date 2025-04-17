"""
Project 532: Controllable Text Generation
Description:
Controllable text generation allows you to control specific attributes of generated text, such as sentiment, style, or length. In this project, we will use transformers to generate text with control over certain attributes, such as generating positive or negative sentiment text.
"""

from transformers import pipeline
 
# 1. Load pre-trained text generation model
generator = pipeline("text-generation", model="gpt2")
 
# 2. Define a prompt for generating positive or negative sentiment text
prompt = "I feel very excited about the future of technology. The possibilities are endless and full of potential."
 
# 3. Generate text based on the prompt with positive sentiment
positive_text = generator(prompt, max_length=100, num_return_sequences=1)
 
# Display the generated text with positive sentiment
print(f"Generated Text (Positive Sentiment): {positive_text[0]['generated_text']}")
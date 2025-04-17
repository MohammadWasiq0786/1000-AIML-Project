"""
Project 533: Sentiment-Controlled Text Generation
Description:
Sentiment-controlled text generation allows you to control the sentiment (positive or negative) of the generated text. In this project, we will use a transformer model to generate text with a specific sentiment based on a given input prompt, enabling us to control whether the output text expresses positive or negative sentiment.
"""

from transformers import pipeline
 
# 1. Load pre-trained text generation model
generator = pipeline("text-generation", model="gpt2")
 
# 2. Define a prompt for generating text with positive sentiment
positive_prompt = "The weather is great today and I am feeling optimistic about the future."
 
# 3. Generate positive sentiment text
positive_sentiment_text = generator(positive_prompt, max_length=100, num_return_sequences=1)
 
# 4. Define a prompt for generating text with negative sentiment
negative_prompt = "I feel like everything is falling apart and there seems to be no hope."
 
# 5. Generate negative sentiment text
negative_sentiment_text = generator(negative_prompt, max_length=100, num_return_sequences=1)
 
# Display the results
print(f"Generated Positive Sentiment Text: {positive_sentiment_text[0]['generated_text']}")
print(f"Generated Negative Sentiment Text: {negative_sentiment_text[0]['generated_text']}")
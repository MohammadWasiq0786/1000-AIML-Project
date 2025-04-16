"""
Project 381. Story generation system
Description:
A story generation system uses generative models to create coherent narratives or stories based on an initial input, prompt, or theme. These models typically use language models such as transformers or RNNs to learn patterns in text and generate creative and engaging stories. This project will focus on building a simple story generation system using a pre-trained language model such as GPT or LSTM to generate stories.

About:
✅ What It Does:
Uses a pre-trained GPT-2 model to generate coherent and creative stories.

Tokenizer is used to convert the input text (prompt) into tokens, which the model processes to generate text.

The generated story continues the input prompt in a coherent manner, producing a narrative that fits the context.

Key features:
GPT-2 model is fine-tuned for general language generation tasks and is capable of generating meaningful text.

Temperature controls the randomness of predictions. A higher value (like 1.0) introduces more randomness, making the output more creative.

No-repeat n-gram size helps in generating more coherent text by avoiding repetitive n-grams.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
 
# 1. Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # GPT-2 model (can be replaced with larger models like gpt2-medium)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 
model.eval()  # Set model to evaluation mode
 
# 2. Generate a story from a prompt
def generate_story(prompt, max_length=200):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text with the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=1.0)
    
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story
 
# 3. Example story prompt and generation
prompt = "Once upon a time in a faraway kingdom, there was a princess who"
generated_story = generate_story(prompt)
 
print("Generated Story:")
print(generated_story)
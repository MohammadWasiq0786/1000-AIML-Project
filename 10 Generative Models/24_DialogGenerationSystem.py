"""
Project 384. Dialog generation system
Description:
A dialogue generation system is a model that generates responses to user inputs, typically used in chatbots or virtual assistants. These models learn to generate meaningful, context-aware, and fluent responses by training on large conversational datasets. Models like GPT-3, BERT, and DialoGPT have been widely used for dialogue generation. In this project, we’ll implement a simple dialogue generation system using GPT-2.

About:
✅ What It Does:
Generates a dialogue response using the GPT-2 model.

Given a user prompt, the model generates a meaningful and context-aware response.

The model uses temperature and top-p sampling to control the randomness and creativity of the generated response.

Key features:
Temperature and top-p control the diversity of the generated dialogue.

The GPT-2 model can generate fluent and contextually relevant dialogue responses for chatbots or virtual assistants.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
# 1. Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # GPT-2 model, can use larger models like gpt2-medium or gpt3 for more advanced results
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 
model.eval()  # Set model to evaluation mode
 
# 2. Generate dialogue responses based on user input
def generate_dialogue_response(prompt, max_length=100, temperature=0.7, top_p=0.9):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate dialogue response with the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1,
                                 temperature=temperature, top_p=top_p, no_repeat_ngram_size=2)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
 
# 3. Example dialogue prompt and generation
prompt = "Hello, how are you?"
generated_response = generate_dialogue_response(prompt)
 
print("Generated Response:")
print(generated_response)
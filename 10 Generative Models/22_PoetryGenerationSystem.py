"""
Project 382. Poetry generation system
Description:
A poetry generation system is a creative application of generative models that produces poetry based on a given prompt or theme. Similar to story generation, poetry generation requires the model to capture not just the meaning of the text but also the structure, rhythm, and often the rhyme. In this project, we will implement a poetry generation system using a language model (e.g., GPT-2) to create poetic text that adheres to stylistic and structural elements.

About:
✅ What It Does:
Uses a pre-trained GPT-2 model to generate poetry based on a given prompt.

The tokenizer encodes the input prompt, which is then passed through the model to generate poetic text.

The model generates a continuation of the input prompt in the style of poetry, considering creativity, rhyme, and rhythm.

Key features:
Temperature controls the randomness of predictions, with lower values making the output more deterministic.

Top-p sampling (nucleus sampling) ensures the model selects tokens from a subset of the probability distribution that covers the cumulative probability of p.

The model generates short poetic texts with creative expressions, often fitting the input prompt's theme and style.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
# 1. Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # GPT-2 model (can be replaced with a larger model like gpt2-medium)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 
model.eval()  # Set model to evaluation mode
 
# 2. Generate poetry from a prompt
def generate_poetry(prompt, max_length=100, temperature=0.7, top_p=0.9):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate poetry with the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1,
                                 temperature=temperature, top_p=top_p, no_repeat_ngram_size=2)
    
    poetry = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return poetry
 
# 3. Example poetry prompt and generation
prompt = "The moonlit sky is full of dreams, where"
generated_poetry = generate_poetry(prompt)
 
print("Generated Poetry:")
print(generated_poetry)
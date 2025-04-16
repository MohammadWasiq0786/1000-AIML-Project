"""
Project 386. Conditional text generation
Description:
Conditional text generation refers to generating text that is conditioned on some input, such as a prompt, a keyword, sentiment, or other context-specific information. This task is widely used in applications such as chatbots, story generation, and text summarization. By conditioning the model on a particular attribute, the generated text can be tailored to meet specific requirements. In this project, we'll implement conditional text generation using a pre-trained language model (e.g., GPT-2) that generates text based on a specified condition.

About:
✅ What It Does:
Generates text that is conditioned on a specified condition, such as a sentiment, topic, or task.

The GPT-2 model is used for conditional text generation, where the input prompt is combined with the condition to guide the text generation process.

Text generation can be tailored to different contexts, such as generating reviews, stories, or descriptions based on the given input.

Key features:
Conditioning the model with a specific task (e.g., writing a positive review) influences the output to align with the desired context.

Temperature and top-p sampling control the creativity and diversity of the generated text.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
# 1. Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # GPT-2 model (can be replaced with a larger model like gpt2-medium or gpt2-large)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 
model.eval()  # Set model to evaluation mode
 
# 2. Generate conditional text based on a given prompt or condition
def generate_conditional_text(prompt, condition, max_length=100, temperature=0.7, top_p=0.9):
    # Combine prompt and condition to create a conditional input
    conditional_prompt = f"{condition}: {prompt}"
    inputs = tokenizer.encode(conditional_prompt, return_tensors='pt')
    
    # Generate text with the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1,
                                 temperature=temperature, top_p=top_p, no_repeat_ngram_size=2)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
 
# 3. Example conditional text generation
condition = "Write a positive review for a product"
prompt = "I really love this product because"
generated_text = generate_conditional_text(prompt, condition)
 
print("Generated Conditional Text:")
print(generated_text)
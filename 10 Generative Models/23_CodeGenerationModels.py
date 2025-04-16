"""
Project 383. Code generation models
Description:
Code generation models are a type of generative model trained to generate code in a specific programming language based on natural language descriptions or code snippets. These models can assist developers by automating code writing, auto-completing code, or generating entire functions based on a description. We can leverage pre-trained models like GPT-2 or specialized models like Codex (a variant of GPT-3) to generate code from prompts.

In this project, we will use GPT-2 to generate code based on simple natural language prompts.

About:
✅ What It Does:
Code generation using a pre-trained GPT-2 model.

Given a natural language prompt, the model generates a Python code snippet that matches the description.

This can be extended for various programming languages and more complex code generation tasks.

Key features:
Temperature and top-p sampling help control the creativity of the generated code. Lower values make the code more predictable and deterministic, while higher values lead to more diverse generation.

The model can generate short functions or code snippets based on a simple description.
"""


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
# 1. Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # GPT-2 model (can be replaced with Codex or larger models for better performance)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 
model.eval()  # Set model to evaluation mode
 
# 2. Generate code from a prompt (natural language)
def generate_code(prompt, max_length=150, temperature=0.7, top_p=0.9):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate code with the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1,
                                 temperature=temperature, top_p=top_p, no_repeat_ngram_size=2)
    
    code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return code
 
# 3. Example code prompt and generation
prompt = "Write a Python function that calculates the factorial of a number"
generated_code = generate_code(prompt)
 
print("Generated Code:")
print(generated_code)
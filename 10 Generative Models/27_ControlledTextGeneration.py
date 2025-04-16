"""
Project 387. Controlled text generation
Description:
Controlled text generation refers to generating text while exerting specific control over certain aspects of the generated output. For example, you might want to control the tone (e.g., formal or informal), style (e.g., academic or conversational), or length of the text. This kind of generation is useful in scenarios like content writing, social media post generation, and creative writing, where a particular style or tone is important.

In this project, we'll implement controlled text generation using GPT-2, where we will control the tone of the generated text by conditioning it with a tone-specific input.

About:
✅ What It Does:
Generates controlled text based on a specified tone (e.g., formal, informal).

The GPT-2 model is used for conditional generation where the prompt is modified to include the control information (tone in this case).

This enables the generation of text that adheres to a desired tone or style, such as formal business communication or casual conversation.

Key features:
Tone control allows you to dictate the style of the generated text (formal or informal).

Temperature and top-p sampling provide flexibility in controlling the randomness and creativity of the output.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
# 1. Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # GPT-2 model (can be replaced with a larger model like gpt2-medium)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 
model.eval()  # Set model to evaluation mode
 
# 2. Generate controlled text based on a given tone or style
def generate_controlled_text(prompt, tone="formal", max_length=150, temperature=0.7, top_p=0.9):
    # Condition the input based on tone (e.g., formal or informal)
    conditioned_prompt = f"Write a {tone} paragraph: {prompt}"
    inputs = tokenizer.encode(conditioned_prompt, return_tensors='pt')
    
    # Generate text with the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1,
                                 temperature=temperature, top_p=top_p, no_repeat_ngram_size=2)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
 
# 3. Example controlled text generation
prompt = "The company's recent product launch"
generated_text_formal = generate_controlled_text(prompt, tone="formal")
generated_text_informal = generate_controlled_text(prompt, tone="informal")
 
print("Generated Formal Text:")
print(generated_text_formal)
 
print("\nGenerated Informal Text:")
print(generated_text_informal)
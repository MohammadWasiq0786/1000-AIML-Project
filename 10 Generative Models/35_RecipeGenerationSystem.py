"""
Project 395. Recipe generation system
Description:
A recipe generation system takes a set of ingredients and generates a recipe with instructions for preparing a dish. This system can be used for applications such as meal planning, cooking assistants, or even automated recipe suggestion based on available ingredients. Generative models like GPT or Seq2Seq can be used to produce recipes from a list of ingredients or a cooking style description.

In this project, we will use a pre-trained language model (e.g., GPT-2) to generate recipes based on a list of ingredients or a cooking style prompt.

About:
✅ What It Does:
Generates a recipe based on a list of ingredients and a cooking style using a pre-trained GPT-2 model.

The model processes the input ingredients and cooking style to create a coherent recipe with instructions.

The generated recipe includes ingredient lists, cooking instructions, and possibly even serving suggestions.

Key features:
Conditional text generation: The model is conditioned on the input ingredients and cooking style to generate a relevant recipe.

Creative recipe generation: The model can generate novel recipes, offering new ideas based on the given inputs.

Flexibility: The approach can be customized to generate recipes for different cuisines, dietary preferences, or meal types.
"""


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
# 1. Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # GPT-2 model (can use larger models like gpt2-medium for better quality)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 
model.eval()  # Set model to evaluation mode
 
# 2. Generate recipe based on ingredients and cooking style
def generate_recipe(ingredients, cooking_style, max_length=150, temperature=0.7, top_p=0.9):
    # Combine ingredients and cooking style to form a prompt for recipe generation
    prompt = f"Generate a {cooking_style} recipe using the following ingredients: {ingredients}"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate recipe with the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1,
                                 temperature=temperature, top_p=top_p, no_repeat_ngram_size=2)
    
    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return recipe
 
# 3. Example ingredients and cooking style
ingredients = "chicken, garlic, lemon, olive oil, parsley"
cooking_style = "Italian"
 
# 4. Generate recipe
generated_recipe = generate_recipe(ingredients, cooking_style)
 
print("Generated Recipe:")
print(generated_recipe)
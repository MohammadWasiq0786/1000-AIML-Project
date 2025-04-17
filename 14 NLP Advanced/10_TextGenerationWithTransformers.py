"""
Project 530: Text Generation with Transformers
Description:
Text generation is a task in NLP where a model generates human-like text based on a given prompt. In this project, we will use transformer models like GPT-2 or T5 to generate coherent text based on an initial seed or prompt.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
# 1. Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 
# 2. Provide a prompt for text generation
prompt = "Once upon a time, in a land far, far away"
 
# 3. Tokenize the input prompt
inputs = tokenizer.encode(prompt, return_tensors="pt")
 
# 4. Generate text using GPT-2
generated_text = model.generate(inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
 
# 5. Decode and print the generated text
generated_text_decoded = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text_decoded}")
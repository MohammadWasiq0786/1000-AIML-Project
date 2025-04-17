"""
Project 529: GPT Model Implementation
Description:
GPT (Generative Pre-trained Transformer) is a transformer-based model designed for text generation. It is trained to predict the next word in a sequence, enabling it to generate coherent and contextually relevant text. In this project, we will implement a GPT model for text generation using Hugging Face’s GPT-2 pre-trained model.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
# 1. Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 
# 2. Provide a prompt to generate text
prompt = "The future of artificial intelligence is"
 
# 3. Tokenize the input prompt
inputs = tokenizer.encode(prompt, return_tensors="pt")
 
# 4. Generate text using GPT-2
generated_text = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
 
# 5. Decode and print the generated text
generated_text_decoded = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text_decoded}")
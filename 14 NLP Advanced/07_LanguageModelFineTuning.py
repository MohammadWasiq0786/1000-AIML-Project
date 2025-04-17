"""
Project 527: Language Model Fine-Tuning
Description:
Fine-tuning a language model involves training a pre-trained model like GPT-2 or BERT on a specific task such as text generation, question answering, or text classification. In this project, we will fine-tune a pre-trained language model on a custom dataset to improve its performance on a specific NLP task.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
 
# 1. Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 
# 2. Load custom dataset for fine-tuning (e.g., a text corpus for text generation)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
 
# 3. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)
 
tokenized_datasets = dataset.map(tokenize_function, batched=True)
 
# 4. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    evaluation_strategy="epoch",     # evaluation strategy to adopt during training
    learning_rate=5e-5,              # learning rate
    per_device_train_batch_size=4,   # batch size for training
    per_device_eval_batch_size=4,    # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
)
 
# 5. Initialize Trainer and start fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)
 
# 6. Train the model
trainer.train()
 
# 7. Generate text using the fine-tuned model
input_text = "The future of artificial intelligence is"
inputs = tokenizer.encode(input_text, return_tensors="pt")
generated_text = model.generate(inputs, max_length=100, num_return_sequences=1)
 
# 8. Decode and print the generated text
generated_text_decoded = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text_decoded}")
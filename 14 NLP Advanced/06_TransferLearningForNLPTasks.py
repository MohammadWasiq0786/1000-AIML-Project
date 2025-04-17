"""
Project 526: Transfer Learning for NLP Tasks
Description:
Transfer learning involves using a pre-trained model on one task and fine-tuning it for a different but related task. In this project, we will demonstrate transfer learning for NLP tasks by fine-tuning a pre-trained transformer model (like BERT or RoBERTa) for a specific task such as sentiment analysis or text classification.
"""

from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
 
# 1. Load a pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)
 
# 2. Load a dataset for text classification (e.g., IMDb dataset for sentiment analysis)
dataset = load_dataset("imdb")
 
# 3. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)
 
tokenized_datasets = dataset.map(tokenize_function, batched=True)
 
# 4. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    evaluation_strategy="epoch",     # evaluation strategy to adopt during training
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
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
 
# 7. Evaluate the model
results = trainer.evaluate()
print(f"Evaluation Results: {results}")
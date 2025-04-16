"""
Project 121. BERT fine-tuning for text classification
Description:
BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art transformer model for various NLP tasks. In this project, we fine-tune a pre-trained BERT model from Hugging Face Transformers on a custom text classification task, such as spam detection or sentiment analysis.

About:
🧠 What This Project Demonstrates:
Fine-tunes a pre-trained BERT model for binary text classification

Uses Hugging Face Transformers for rapid development

Includes tokenization, training, and evaluation pipeline
"""

# Install if not already: pip install transformers datasets torch scikit-learn
 
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np
import torch
 
# Load a small binary classification dataset (e.g., IMDb)
dataset = load_dataset("imdb", split="train[:5%]").train_test_split(test_size=0.2)
 
# Load pre-trained tokenizer and tokenize the dataset
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
 
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
 
encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
 
# Load BERT for sequence classification (binary)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
 
# Define metric
metric = load_metric("accuracy")
 
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)
 
# Define Trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics,
)
 
# Fine-tune the model
trainer.train()
 
# Evaluate
trainer.evaluate()
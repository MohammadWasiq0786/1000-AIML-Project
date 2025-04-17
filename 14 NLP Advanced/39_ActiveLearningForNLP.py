"""
Project 559: Active Learning for NLP
Description:
Active learning is a machine learning approach where the model queries the user (or an oracle) for labels on the most informative data points, instead of using random samples. This is particularly useful in NLP tasks where labeling data can be expensive. In this project, we will implement an active learning loop to iteratively improve the performance of an NLP model by labeling the most uncertain examples.
"""

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import random
 
# 1. Load dataset and pre-trained BERT model
dataset = load_dataset("imdb", split="train[:10%]")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
 
# 2. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)
 
dataset = dataset.map(tokenize_function, batched=True)
 
# 3. Define the active learning loop
def active_learning(model, dataset, num_iterations=5):
    for i in range(num_iterations):
        # Randomly select a small sample from the unlabeled data
        unlabeled_data = dataset.shuffle(seed=42).select(range(10))  # Select the first 10 samples for simplicity
        inputs = tokenizer(unlabeled_data['text'], padding=True, truncation=True, return_tensors="pt")
 
        # Get predictions from the model
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
        
        # Select the most uncertain samples (lowest confidence)
        uncertainties = -torch.max(predictions, dim=1)[0]  # Negative log-likelihood
        most_uncertain_indices = uncertainties.argsort()[:3]  # Select 3 most uncertain examples
        
        # Simulate labeling the uncertain examples (in real case, get human labels)
        labeled_data = unlabeled_data[most_uncertain_indices]
 
        # Add the labeled data to the training set
        dataset = dataset.concatenate(labeled_data)
 
        print(f"Iteration {i+1}: Added {len(labeled_data)} labeled samples.")
        
    return model
 
# 4. Run active learning loop
trained_model = active_learning(model, dataset)
 
# 5. Evaluate the model performance (for simplicity, not using a validation set here)
trainer = Trainer(model=trained_model, args=TrainingArguments(output_dir='./results'))
trainer.train()
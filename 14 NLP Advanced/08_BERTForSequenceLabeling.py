"""
Project 528: BERT for Sequence Labeling
Description:
Sequence labeling involves assigning a label to each element in a sequence, such as part-of-speech tagging, named entity recognition (NER), or chunking. In this project, we will fine-tune BERT for sequence labeling tasks, such as tagging named entities (e.g., people, organizations, locations) in text.
"""

from transformers import BertForTokenClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
 
# 1. Load pre-trained BERT model and tokenizer for token classification (NER task)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
model = BertForTokenClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
 
# 2. Load dataset for named entity recognition (e.g., CONLL-03 dataset)
dataset = load_dataset("conll2003")
 
# 3. Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], padding='max_length', truncation=True)
    labels = examples['ner_tags']
    tokenized_inputs['labels'] = labels
    return tokenized_inputs
 
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
 
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
 
# 8. Predict labels for a sample sentence
sentence = "Hawking was a renowned theoretical physicist"
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
inputs = tokenizer(sentence, return_tensors="pt")
 
outputs = model(**inputs).logits
predictions = outputs.argmax(dim=-1)
 
# Display the predicted labels for each token
predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]
for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {label}")
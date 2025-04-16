"""
Project 108. Semantic role labeling
Description:
Semantic Role Labeling (SRL) is the process of assigning roles to words or phrases in a sentence to determine who did what to whom, when, and how. It helps in understanding sentence meaning beyond grammar. In this project, we use the AllenNLP library to perform SRL on English text.

About:
Example Output:
▶ Verb: gave
   Tags: B-ARG0 B-V B-ARG1 I-ARG1 B-ARG2 B-ARGM-TMP I-ARGM-TMP
This tells us:

ARG0: Who? → John

V: Verb → gave

ARG1: What? → a book

ARG2: To whom? → to Mary

ARGM-TMP: When? → on her birthday

🧠 What This Project Demonstrates:
Identifies semantic roles in sentences (like subject, object, time, etc.)

Helps in understanding the meaning and intent behind statements

Uses BERT-based pretrained model from AllenNLP
"""

# Install if not already:
# pip install allennlp allennlp-models
 
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
 
# Load pretrained SRL model from AllenNLP
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
 
# Input sentence
sentence = "John gave a book to Mary on her birthday."
 
# Perform Semantic Role Labeling
result = predictor.predict(sentence=sentence)
 
# Display predicates and their arguments
print("\nSentence:", sentence)
print("\nSemantic Roles:\n")
for verb in result['verbs']:
    print(f"▶ Verb: {verb['verb']}")
    print(f"   Tags: {' '.join(verb['tags'])}")
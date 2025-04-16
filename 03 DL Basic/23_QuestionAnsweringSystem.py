"""
Project 103. Question answering system
Description:
A Question Answering (QA) system automatically provides precise answers to user questions from a given context. In this project, we use a pre-trained BERT model from Hugging Face Transformers to build a contextual QA system, where the model reads a paragraph and answers questions based on it.

About:
🤖 What This Project Demonstrates:
Loads a pre-trained BERT-based model for question answering

Extracts answers from unstructured text using attention mechanisms

Works out-of-the-box with DistilBERT fine-tuned on SQuAD
"""

# Install if not already: pip install transformers torch
 
from transformers import pipeline
 
# Load QA pipeline using pre-trained BERT
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
 
# Provide a context paragraph
context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, 
it was initially criticized by some of France's leading artists and intellectuals for its design, 
but it has become a global cultural icon of France and one of the most recognizable structures in the world.
"""
 
# Ask a question based on the context
question = "Who designed the Eiffel Tower?"
 
# Run question-answering
result = qa_pipeline(question=question, context=context)
 
# Show answer
print(f"Q: {question}")
print(f"A: {result['answer']}")
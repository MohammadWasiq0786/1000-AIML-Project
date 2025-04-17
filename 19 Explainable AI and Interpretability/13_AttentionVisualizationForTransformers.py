"""
Project 733: Attention Visualization for Transformers
Description:
Attention visualization is a technique used to understand how transformer models (such as BERT or GPT) allocate attention to different parts of the input during the model's processing. Transformers use self-attention mechanisms to focus on relevant words or tokens when making predictions, and attention visualization helps to highlight which tokens the model is attending to for a specific prediction. This is useful for model interpretability, especially for NLP tasks like text classification, machine translation, and question answering. In this project, we will implement attention visualization for a transformer model (such as BERT) and visualize how the model attends to various tokens in a sentence.

Explanation:
Load BERT Model and Tokenizer: We use the Hugging Face Transformers library to load the pre-trained BERT model (bert-base-uncased) and its corresponding tokenizer. We set output_attentions=True to extract the attention weights from the model.

Encode the Sentence: The input sentence is tokenized using the BERT tokenizer, and the tokenized input is passed through the model to obtain the attention weights.

Extract Attention Weights: The attention weights are a tensor of shape [num_layers, num_heads, seq_length, seq_length], where num_layers refers to the number of layers in BERT, num_heads is the number of attention heads in each layer, and seq_length is the length of the tokenized input sequence. We extract the attention weights for the specific layer and head we want to visualize.

Visualize Attention Weights: The seaborn heatmap is used to visualize the attention weights for a given layer and attention head. The attention matrix shows which tokens the model attends to when processing each token in the sequence. Higher values indicate stronger attention.

This method provides an insightful visualization of how the BERT model attends to different parts of the sentence, which is useful for interpreting the model's decision-making process.
"""

import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
 
# 1. Load the pre-trained BERT model and tokenizer
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)  # Output attentions
    return tokenizer, model
 
# 2. Encode a sentence and get the attention weights
def get_attention_weights(model, tokenizer, sentence):
    """
    Tokenize the input sentence and get the attention weights from the model.
    """
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get attention weights (shape: [num_layers, num_heads, seq_length, seq_length])
    attentions = outputs.attentions
    return attentions
 
# 3. Visualize the attention weights for the first layer and first attention head
def visualize_attention(attentions, sentence, layer=0, head=0):
    """
    Visualize the attention weights for a specific layer and attention head.
    """
    # Select the attention weights for the chosen layer and head
    attention = attentions[layer][0, head].cpu().numpy()
 
    # Tokenize the sentence and get tokenized words
    words = sentence.split()
 
    # Create a heatmap of the attention weights
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=words, yticklabels=words, cmap='viridis', cbar=True)
    plt.title(f'Attention Weights for Layer {layer+1}, Head {head+1}')
    plt.xlabel('Tokens Attended To')
    plt.ylabel('Tokens Attending')
    plt.show()
 
# 4. Example usage
sentence = "The quick brown fox jumps over the lazy dog"
tokenizer, model = load_bert_model()
 
# Get attention weights
attentions = get_attention_weights(model, tokenizer, sentence)
 
# Visualize attention for the first layer and first head
visualize_attention(attentions, sentence, layer=0, head=0)
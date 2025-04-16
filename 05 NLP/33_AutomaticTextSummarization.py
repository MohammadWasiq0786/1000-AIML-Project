"""
Project 193. Automatic text summarization
Description:
Automatic text summarization generates a shorter version of a document that retains its key information. There are two main types: extractive summarization, which pulls important sentences, and abstractive summarization, which generates new sentences. In this project, we use a pretrained transformer model to create abstractive summaries of longer texts.

About:
🧠 What This Project Demonstrates:
Generates human-like summaries using transformer models (e.g., BART, T5)

Compresses long input into coherent and concise summaries

Can be applied to news, articles, reports, or reviews
"""

# Install if not already: pip install transformers torch
 
from transformers import pipeline
 
# Load the summarization pipeline with a pretrained model (e.g., BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
 
# Long sample text
text = """
Artificial Intelligence (AI) is rapidly transforming various sectors including healthcare, education, finance, and transportation. 
With the ability to process massive datasets, identify patterns, and automate decision-making, AI is streamlining operations 
and improving efficiency. In healthcare, AI assists in diagnosing diseases and personalizing treatments. In finance, it's used 
to detect fraud and make investment decisions. As AI continues to evolve, it’s essential to address ethical concerns, including 
bias, privacy, and job displacement.
"""
 
# Generate summary
summary = summarizer(text, max_length=60, min_length=25, do_sample=False)
 
# Output the result
print("🧠 Original Text:\n")
print(text)
print("\n✅ Abstractive Summary:\n")
print(summary[0]['summary_text'])
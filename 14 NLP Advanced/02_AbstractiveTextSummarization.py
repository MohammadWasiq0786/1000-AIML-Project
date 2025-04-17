"""
Project 522: Abstractive Text Summarization
Description:
Abstractive text summarization involves generating a summary of a text by understanding its meaning and then rephrasing it in a concise form. Unlike extractive summarization, which selects portions of the text, abstractive summarization generates new sentences. In this project, we will use a pre-trained transformer model like BART or T5 to perform abstractive summarization.
"""

from transformers import pipeline
 
# 1. Load the pre-trained summarization model (e.g., BART or T5)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
 
# 2. Provide the text to be summarized
text = """
Abstractive text summarization is an NLP task where the goal is to generate a summary that is a paraphrased version of the original text. 
This is different from extractive summarization, where the summary is made up of direct excerpts from the original text. 
Abstractive summarization models, like BART and T5, rely on transformer architectures to understand the context and generate coherent summaries. 
These models are trained on large datasets and can summarize long pieces of text into concise, coherent summaries that capture the key ideas.
"""
 
# 3. Use the pipeline to generate the summary
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
 
# 4. Display the result
print(f"Summary: {summary[0]['summary_text']}")
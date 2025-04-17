"""
Project 543: Multi-Document Summarization
Description:
Multi-document summarization involves generating a concise summary by combining information from multiple documents. The goal is to create a summary that captures the most important points from all documents, without redundant information. In this project, we will use a pre-trained transformer model to generate a summary based on multiple input texts.
"""

from transformers import pipeline
 
# 1. Load pre-trained model for summarization
summarizer = pipeline("summarization", model="t5-small")
 
# 2. Provide multiple documents to summarize
documents = [
    "Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
    "Machine learning (ML) is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns, and make decisions with minimal human intervention.",
    "Natural language processing (NLP) is a field of computer science, artificial intelligence, and computational linguistics concerned with the interactions between computers and human (natural) languages, and, in particular, concerned with programming computers to fruitfully process large natural language data."
]
 
# 3. Concatenate the documents into one text
text = " ".join(documents)
 
# 4. Use the model to summarize the concatenated text
summary = summarizer(text, max_length=100, min_length=50, do_sample=False)
 
# 5. Display the summary
print(f"Generated Summary: {summary[0]['summary_text']}")
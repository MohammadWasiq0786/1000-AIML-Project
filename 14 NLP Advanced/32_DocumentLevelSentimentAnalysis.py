"""
Project 552: Document-Level Sentiment Analysis
Description:
Document-level sentiment analysis involves classifying the overall sentiment of a document as positive, negative, or neutral, rather than analyzing sentiment at the sentence or aspect level. In this project, we will use a pre-trained transformer model to classify the sentiment of an entire document.
"""

from transformers import pipeline
 
# 1. Load pre-trained model for document-level sentiment analysis
classifier = pipeline("sentiment-analysis", model="bert-base-uncased")
 
# 2. Provide a sample document for sentiment analysis
document = """
The market has seen significant growth this quarter, with many industries reporting increased earnings. 
Despite this, some sectors are facing challenges due to rising inflation and supply chain disruptions. 
Overall, however, the economic outlook remains positive, and experts are optimistic about the future.
"""
 
# 3. Analyze the sentiment of the document
sentiment = classifier(document)
 
# 4. Display the result
print(f"Sentiment: {sentiment[0]['label']} with a score of {sentiment[0]['score']:.2f}")
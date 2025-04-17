"""
Project 553: Aspect-Based Sentiment Analysis
Description:
Aspect-based sentiment analysis (ABSA) involves determining the sentiment towards specific aspects or features of a product or service, such as the quality of service, price, or delivery time. In this project, we will use a pre-trained transformer model to perform aspect-based sentiment analysis on product reviews.
"""

from transformers import pipeline
 
# 1. Load pre-trained model for sentiment analysis
classifier = pipeline("sentiment-analysis", model="bert-base-uncased")
 
# 2. Provide a sample text with multiple aspects to analyze sentiment for
text = """
The food at the restaurant was delicious, but the service was slow. The ambiance was great, but the price was a bit high for the portion size.
"""
 
# 3. Define aspects to analyze sentiment for
aspects = ["food", "service", "ambiance", "price"]
 
# 4. Analyze sentiment for each aspect
aspect_sentiments = {}
for aspect in aspects:
    sentiment = classifier(f"The {aspect} is {text}")
    aspect_sentiments[aspect] = sentiment[0]['label']
 
# 5. Display the aspect-based sentiment analysis results
for aspect, sentiment in aspect_sentiments.items():
    print(f"Sentiment for {aspect}: {sentiment}")
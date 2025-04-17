"""
Project 499: Financial News Analysis
Description:
Financial news analysis involves extracting insights from financial news articles to gauge the market sentiment or understand market trends. In this project, we will implement a simple sentiment analysis model to analyze news headlines related to a specific stock or market, classifying the sentiment as positive, negative, or neutral.

For real-world systems:

Integrate with financial news sources like Reuters, Bloomberg, or Yahoo Finance.

You can enhance this project with advanced NLP models like BERT or FinBERT for more accurate sentiment analysis.

✅ What It Does:
Simulates financial news headlines related to Apple (or any stock) and performs sentiment analysis using TextBlob.

Classifies sentiment as positive, negative, or neutral based on the polarity of the text.

Visualizes the sentiment distribution with a bar chart to show how many headlines are classified as positive, negative, or neutral.

Key Extensions and Customizations:
Real-time news collection: Integrate with news APIs like NewsAPI or Google News API to collect live news headlines.

Advanced sentiment models: Use FinBERT or VADER for more domain-specific sentiment analysis in the financial industry.

Market trend prediction: Use sentiment analysis results as features in a model to predict stock price movement or market sentiment.
"""

from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Simulate a dataset of financial news headlines
data = {
    "headline": [
        "Apple hits record high in stock price",
        "Apple faces regulatory challenges in Europe",
        "Apple launches new iPhone model",
        "Apple's revenue growth slows in Q4",
        "Apple announces plans for environmental sustainability"
    ]
}
 
df = pd.DataFrame(data)
 
# 2. Define a function to calculate sentiment polarity
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
 
# 3. Apply sentiment analysis to the headlines
df['sentiment'] = df['headline'].apply(get_sentiment)
 
# 4. Classify sentiment into categories
def classify_sentiment(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"
 
df['sentiment_class'] = df['sentiment'].apply(classify_sentiment)
 
# 5. Display the results
print("Financial News Sentiment Analysis Results:\n")
print(df)
 
# 6. Plot the sentiment distribution
sentiment_counts = df['sentiment_class'].value_counts()
 
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title("Sentiment Distribution of Financial News Headlines")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()
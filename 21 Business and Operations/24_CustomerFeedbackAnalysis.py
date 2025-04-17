"""
Project 824. Customer Feedback Analysis

Customer feedback analysis extracts insights from open-text feedback (e.g., surveys, reviews, support tickets) to understand satisfaction, identify pain points, and guide improvements. In this project, we’ll use sentiment analysis on textual feedback to classify responses as positive, neutral, or negative.

This basic sentiment analysis helps categorize open-text feedback at scale. You can enhance it with keyword extraction, topic modeling (e.g., LDA), or fine-tuned transformers for nuanced insights.
"""

import pandas as pd
from textblob import TextBlob
 
# Sample customer feedback data
feedback = [
    "Great service and fast delivery!",
    "The product quality was terrible.",
    "Average experience, nothing special.",
    "Very helpful customer support.",
    "The item arrived late and was damaged.",
    "I'm satisfied with the purchase.",
    "Terrible app. Crashes every time!",
    "Absolutely love it!",
    "Not bad, but could be improved.",
    "Worst experience ever."
]
 
# Create a DataFrame
df = pd.DataFrame({'Feedback': feedback})
 
# Function to classify sentiment using TextBlob polarity
def classify_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return 'Positive'
    elif polarity < -0.2:
        return 'Negative'
    else:
        return 'Neutral'
 
# Apply sentiment classification
df['Sentiment'] = df['Feedback'].apply(classify_sentiment)
 
# Display results
print("Customer Feedback Sentiment Analysis:")
print(df)
 
# Plot sentiment distribution
import matplotlib.pyplot as plt
 
plt.figure(figsize=(6, 4))
df['Sentiment'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Feedback Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.tight_layout()
plt.grid(axis='y')
plt.show()
"""
Project 184. Sentiment analysis for social media
Description:
Sentiment analysis on social media helps businesses and analysts understand public opinions in real time. In this project, we analyze sentiment in short, informal text like tweets or posts using a pretrained sentiment model. We use VADER (Valence Aware Dictionary for Sentiment Reasoning), which is tailored for social media language.

About:
🧠 What This Project Demonstrates:
Analyzes short, informal social media text for sentiment

Uses VADER, which accounts for emojis, slang, capitalization, and punctuation

Classifies sentiment as positive, negative, or neutral
"""

# Install if not already: pip install nltk
 
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
 
nltk.download('vader_lexicon')
 
# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()
 
# Example social media posts
tweets = [
    "I'm so happy with the new update! 🚀🔥",
    "This is the worst experience I've ever had. #fail",
    "Meh, it's okay I guess. Nothing special.",
    "Loving the new features! Great job @devteam!",
    "Why is this so broken again? I'm tired of this..."
]
 
print("🧠 Sentiment Analysis on Social Media Posts:\n")
 
for tweet in tweets:
    scores = sia.polarity_scores(tweet)
    sentiment = "positive" if scores['compound'] > 0.05 else "negative" if scores['compound'] < -0.05 else "neutral"
    
    print(f"💬 Tweet: {tweet}")
    print(f"🔎 Sentiment: {sentiment} (Score: {scores['compound']})\n")
"""
Project 907. Content Moderation System

A content moderation system automatically flags or removes inappropriate, offensive, or policy-violating content (e.g., hate speech, profanity, violence). In this project, we simulate user comments and use keyword-based filtering with optional sentiment scoring.

What It Detects:
Use of explicit banned words

Negative sentiment, indicating possible toxicity

Easily extendable to use HateBERT, Perspective API, or custom toxicity classifiers

🧠 In real platforms:

Use multi-stage pipelines (regex → ML → human review)

Support multilingual content moderation

Auto-hide, warn, or escalate based on severity
"""

import pandas as pd
import re
from textblob import TextBlob
 
# Simulated user comments
comments = [
    "You are so dumb and annoying!",
    "I love this community. Everyone is kind.",
    "Get lost, you idiot.",
    "This post is informative and helpful.",
    "You're such a loser!",
    "Great job on the project!"
]
 
# List of banned words (can be expanded or replaced with NLP models)
banned_words = ['dumb', 'idiot', 'loser', 'stupid', 'hate']
 
# Check for moderation flags
def moderate_comment(text):
    text_lower = text.lower()
    flags = []
    
    # Check for offensive words
    if any(bad_word in text_lower for bad_word in banned_words):
        flags.append("⚠️ Offensive Language")
    
    # Optional: Check for negative sentiment
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.4:
        flags.append("😡 Negative Sentiment")
    
    return ", ".join(flags) if flags else "✅ Clean"
 
# Evaluate all comments
df = pd.DataFrame({'Comment': comments})
df['ModerationResult'] = df['Comment'].apply(moderate_comment)
 
# Show results
print("Content Moderation Report:")
print(df)
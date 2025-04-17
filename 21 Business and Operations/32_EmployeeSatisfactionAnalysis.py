"""
Project 832. Employee Satisfaction Analysis

Employee satisfaction analysis helps organizations understand workforce morale and engagement. In this project, we simulate employee survey responses and use sentiment analysis to classify satisfaction levels from open-ended feedback.

This model uses simple NLP to detect whether employees express satisfaction, neutrality, or dissatisfaction. It can be expanded with topic modeling (e.g., complaints vs. praise), time-series trends, or AI-based emotion detection.
"""

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
 
# Simulated open-text employee feedback
feedback = [
    "I love working here. The culture is amazing!",
    "Management doesn't listen to employee concerns.",
    "It's okay, but there’s a lot of room for improvement.",
    "Great team and work-life balance.",
    "Too much pressure and no recognition.",
    "I'm happy with the flexibility and support from my manager.",
    "The pay is not competitive and growth is slow.",
    "Fantastic experience overall!"
]
 
# Create DataFrame
df = pd.DataFrame({'Feedback': feedback})
 
# Function to classify sentiment using TextBlob polarity
def classify_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return 'Satisfied'
    elif polarity < -0.2:
        return 'Dissatisfied'
    else:
        return 'Neutral'
 
# Apply sentiment classification
df['SatisfactionLevel'] = df['Feedback'].apply(classify_sentiment)
 
# Show results
print("Employee Satisfaction Analysis:")
print(df)
 
# Visualize sentiment distribution
plt.figure(figsize=(6, 4))
df['SatisfactionLevel'].value_counts().plot(kind='bar', color='mediumseagreen')
plt.title('Employee Satisfaction Sentiment')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Responses')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
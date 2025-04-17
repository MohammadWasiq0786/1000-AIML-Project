"""
Project 518: Financial Sentiment Analysis
Description:
Financial sentiment analysis involves analyzing financial news, reports, or social media to gauge market sentiment and predict stock price movements or economic trends. In this project, we will use NLP techniques to perform sentiment analysis on financial news articles and classify them as positive, negative, or neutral.
"""

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
 
# 1. Sample financial news articles
data = {
    'headline': [
        "Stock market hits record high amidst economic recovery",
        "Inflation concerns lead to market volatility",
        "Tech stocks soar after strong earnings reports",
        "Economic slowdown predicted due to rising interest rates",
        "Global markets recover as stimulus packages are announced"
    ]
}
 
df = pd.DataFrame(data)
 
# 2. Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()
 
# 3. Apply sentiment analysis to each headline
df['sentiment_score'] = df['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])
 
# 4. Classify sentiment into categories
df['sentiment_class'] = df['sentiment_score'].apply(
    lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
)
 
# 5. Display the results
print("Sentiment Analysis of Financial News Headlines:")
print(df)
 
# 6. Plot the distribution of sentiments
sentiment_counts = df['sentiment_class'].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution of Financial News Headlines')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()
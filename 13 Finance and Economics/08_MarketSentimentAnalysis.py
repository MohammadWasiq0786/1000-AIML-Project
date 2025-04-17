"""
Project 488: Market Sentiment Analysis
Description:
Market sentiment analysis involves analyzing social media, news, or financial reports to gauge the overall sentiment (positive, negative, or neutral) about a particular stock, market, or economic condition. In this project, we'll implement a simple sentiment analysis model using text data from news articles or social media (e.g., Twitter) to predict market sentiment related to a specific stock or market.

For real-world use:

Use Twitter API, Reddit posts, or financial news articles to collect data.

You can enhance this project by using BERT-based models or VADER for more robust sentiment classification.

✅ What It Does:
Simulates sentiment analysis on a series of headlines related to a stock (Apple in this case).

Uses TextBlob to calculate sentiment polarity, which ranges from -1 (negative) to +1 (positive).

Plots the stock price and market sentiment over time to show the relationship between sentiment and stock movement.

Key Extensions and Customizations:
Real-time data collection: Integrate with the Twitter API or news APIs (e.g., News API or Reddit API) to gather up-to-date news articles or tweets.

Advanced sentiment models: Use VADER, BERT, or RoBERTa for more accurate and context-aware sentiment analysis.

Sentiment-based trading: Implement a simple trading strategy where sentiment scores trigger buy or sell actions.
"""

from textblob import TextBlob
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
 
# 1. Download stock data (e.g., Apple stock) for visualization
stock_data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
 
# 2. Simulated news headlines or tweets related to the stock
# In a real application, you would scrape this data from Twitter or news sources
data = {
    "date": pd.date_range(start="2020-01-01", periods=5, freq="D"),
    "headline": [
        "Apple hits record high in stock price", 
        "Apple faces regulatory challenges in Europe", 
        "Apple launches new iPhone model", 
        "Apple's revenue growth slows in Q4", 
        "Apple announces plans for environmental sustainability"
    ]
}
 
# 3. Create a DataFrame
df = pd.DataFrame(data)
 
# 4. Define a function to calculate sentiment polarity
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
 
# 5. Apply sentiment analysis to the headlines
df['sentiment'] = df['headline'].apply(get_sentiment)
 
# 6. Plot the sentiment over time
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['sentiment'], marker='o', linestyle='-', color='b', label="Sentiment")
plt.title("Market Sentiment Analysis for Apple Stock")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()
 
# 7. Combine sentiment with stock price (optional)
stock_data['sentiment'] = np.interp(np.linspace(0, len(stock_data)-1, len(df)), np.linspace(0, len(df)-1, len(df)), df['sentiment'].values)
 
# 8. Plot stock price and sentiment together
fig, ax1 = plt.subplots(figsize=(10, 6))
 
ax1.plot(stock_data['Close'], color='g', label="Stock Price")
ax1.set_xlabel("Date")
ax1.set_ylabel("Stock Price (USD)", color='g')
ax1.tick_params(axis='y', labelcolor='g')
 
ax2 = ax1.twinx()
ax2.plot(stock_data['sentiment'], color='b', label="Sentiment")
ax2.set_ylabel("Sentiment", color='b')
ax2.tick_params(axis='y', labelcolor='b')
 
plt.title("Apple Stock Price and Market Sentiment Over Time")
plt.show()
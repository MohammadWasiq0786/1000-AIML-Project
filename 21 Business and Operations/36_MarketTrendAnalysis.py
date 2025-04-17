"""
Project 836. Market Trend Analysis

Market trend analysis tracks shifts in consumer behavior, industry focus, or keyword popularity over time. In this project, we simulate weekly search or mention volumes of market-related keywords and use trend plotting and smoothing to visualize changes.

This model visualizes how market interests evolve over time. You can extend this with:

Google Trends API for real data,

smoothing techniques (e.g., rolling mean), and

topic modeling from news articles or social media.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# Simulated weekly keyword search/mention data for 3 market trends
weeks = pd.date_range(start='2024-01-01', periods=12, freq='W')
trend_data = {
    'AI Tools': [120, 135, 150, 160, 180, 200, 220, 240, 270, 290, 310, 330],
    'Remote Work': [300, 280, 270, 260, 250, 245, 240, 235, 230, 225, 220, 215],
    'Sustainability': [100, 105, 110, 115, 130, 145, 160, 180, 200, 210, 220, 230]
}
 
df = pd.DataFrame(trend_data, index=weeks)
 
# Plot raw trends
plt.figure(figsize=(10, 6))
for column in df.columns:
    plt.plot(df.index, df[column], marker='o', label=column)
 
plt.title('Market Trend Analysis (Weekly)')
plt.xlabel('Week')
plt.ylabel('Search/Mention Volume')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
 
# Identify top growing trend
growth_rates = df.iloc[-1] - df.iloc[0]
top_trend = growth_rates.idxmax()
print(f"Top growing market trend: {top_trend} (+{growth_rates[top_trend]} units)")
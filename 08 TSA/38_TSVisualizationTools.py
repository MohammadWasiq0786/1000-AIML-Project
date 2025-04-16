"""
Project 318. Time series visualization tools
Description:
Effective visualization is key to understanding time series data. It helps uncover:

Trends, seasonality, and outliers

Comparisons across series

Temporal correlations and anomalies

In this project, we’ll explore several tools and plots for visualizing time series data using Matplotlib, Seaborn, and Plotly for interactivity.

About:
✅ What It Does:
Creates a multivariate time series (e.g. weather data)

Visualizes series together on a line chart

Shows inter-series relationships using a correlation heatmap

Adds an interactive Plotly plot for exploration

These tools help in EDA, presentation, and identifying key patterns before modeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
 
# 1. Generate multivariate time series
np.random.seed(42)
n = 200
dates = pd.date_range("2022-01-01", periods=n)
ts = pd.DataFrame({
    'Date': dates,
    'Temperature': 20 + 5 * np.sin(np.linspace(0, 3 * np.pi, n)) + np.random.randn(n),
    'Humidity': 60 + 10 * np.cos(np.linspace(0, 2 * np.pi, n)) + np.random.randn(n),
    'Pressure': 1013 + 2 * np.random.randn(n)
})
 
# 2. Line plots using Matplotlib
plt.figure(figsize=(12, 4))
plt.plot(ts['Date'], ts['Temperature'], label="Temperature")
plt.plot(ts['Date'], ts['Humidity'], label="Humidity")
plt.plot(ts['Date'], ts['Pressure'], label="Pressure")
plt.title("Time Series – Multiple Lines")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
 
# 3. Heatmap of correlations
plt.figure(figsize=(5, 4))
sns.heatmap(ts.drop('Date', axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Time Series Variables")
plt.show()
 
# 4. Interactive Plotly line chart
fig = px.line(ts, x='Date', y=['Temperature', 'Humidity', 'Pressure'],
              title="Interactive Multivariate Time Series")
fig.show()
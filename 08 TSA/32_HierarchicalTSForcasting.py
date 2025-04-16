"""
Project 312. Hierarchical time series forecasting
Description:
Hierarchical Time Series (HTS) forecasting deals with data organized in nested groups, like:

Total → Region → Store

Company → Department → Product

The goal is to make consistent forecasts across all levels of aggregation — so that forecasts at the bottom sum up to forecasts at the top.

In this project, we’ll simulate a hierarchy (e.g., total sales broken into groups), forecast each level using ARIMA, and reconcile them using simple bottom-up aggregation.

About:
✅ What It Does:
Simulates a 3-region sales hierarchy with an aggregated total

Forecasts each region individually using ARIMA

Combines them to produce a top-level forecast using bottom-up reconciliation

Ensures that forecasts respect the hierarchy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
 
# 1. Simulate data: 3 regions whose sales sum to total
np.random.seed(42)
periods = 60
time = pd.date_range(start="2020-01", periods=periods, freq='M')
 
region1 = np.random.normal(loc=100, scale=5, size=periods).cumsum()
region2 = np.random.normal(loc=120, scale=6, size=periods).cumsum()
region3 = np.random.normal(loc=80, scale=4, size=periods).cumsum()
 
total = region1 + region2 + region3
 
# Create DataFrame
df = pd.DataFrame({
    'Date': time,
    'Region1': region1,
    'Region2': region2,
    'Region3': region3,
    'Total': total
}).set_index('Date')
 
# 2. Forecast bottom-level (regions) using ARIMA
forecast_horizon = 12
region_forecasts = {}
 
for region in ['Region1', 'Region2', 'Region3']:
    model = ARIMA(df[region], order=(1, 1, 0)).fit()
    forecast = model.forecast(forecast_horizon)
    region_forecasts[region] = forecast
 
# 3. Aggregate to get top-level forecast (bottom-up)
df_future = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='M')
bottom_up_total = region_forecasts['Region1'] + region_forecasts['Region2'] + region_forecasts['Region3']
 
# 4. Plot results
plt.figure(figsize=(12, 5))
plt.plot(df['Total'], label='Historical Total', linewidth=2)
plt.plot(df_future, bottom_up_total, label='Forecast Total (Bottom-Up)', linestyle='--')
plt.title("Hierarchical Time Series Forecasting – Bottom-Up Aggregation")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
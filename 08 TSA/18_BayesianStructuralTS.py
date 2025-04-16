"""
Project 298. Bayesian structural time series
Description:
Bayesian Structural Time Series (BSTS) models allow for flexible decomposition of time series into components like:

Trend

Seasonality

Regression effects

And most importantly: Bayesian uncertainty estimation

BSTS is widely used in forecasting, causal inference (e.g., impact of an intervention), and nowcasting. We'll implement a basic BSTS model using the orbit library.

About:
✅ What It Does:
Uses a built-in dataset for weekly jobless claims

Fits a BSTS model with built-in seasonality + uncertainty

Outputs confidence intervals around predictions

Ready for forecasting, causal impact analysis, and more
"""

# Install the required package:
# pip install orbit
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from orbit.models import BSTS
from orbit.diagnostics.backtest import backtest
from orbit.diagnostics.plot import plot_predicted_data
from orbit.utils.dataset import load_iclaims
 
# 1. Load sample dataset: weekly initial jobless claims
data = load_iclaims()
data = data.rename(columns={'claims': 'y', 'week': 'ds'})  # Orbit expects 'ds' and 'y'
 
# 2. Define and fit BSTS model
model = BSTS(
    response_col='y',
    date_col='ds',
    seasonality=52,  # weekly data
    seed=2024
)
model.fit(data=data)
 
# 3. Predict future 13 weeks
pred_df = model.predict(df=data)
 
# 4. Plot results
plot_predicted_data(training_actual_df=data, predicted_df=pred_df, date_col='ds', value_col='y')
plt.title("Bayesian Structural Time Series – Orbit")
plt.grid(True)
plt.show()
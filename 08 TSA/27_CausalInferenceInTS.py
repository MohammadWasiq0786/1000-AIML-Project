"""
Project 307. Causal inference in time series
Description:
Causal inference in time series aims to determine whether changes in one variable cause changes in another — beyond simple correlation. Key methods include:

Granger causality

Lag-based regression

Conditional independence tests

In this project, we’ll apply Granger causality tests to check if one time series Granger-causes another — meaning past values of one improve predictions of the other.

About:
✅ What It Does:
Simulates two time series: one influencing the other with a time lag

Applies Granger causality test:

H₀: X does not Granger-cause Y

Rejecting H₀ suggests causality

Uses F-statistics and p-values to assess lagged predictive power
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
 
# 1. Generate synthetic time series
np.random.seed(42)
n = 200
 
# Series X: white noise
x = np.random.normal(size=n)
 
# Series Y: influenced by lagged X
y = np.zeros_like(x)
for t in range(2, n):
    y[t] = 0.5 * y[t-1] + 0.3 * x[t-2] + np.random.normal()
 
# Combine into DataFrame
df = pd.DataFrame({'x': x, 'y': y})
 
# 2. Plot the time series
plt.figure(figsize=(10, 4))
plt.plot(df['x'], label='X (Cause)')
plt.plot(df['y'], label='Y (Effect)')
plt.title("Synthetic Time Series for Granger Causality")
plt.legend()
plt.grid(True)
plt.show()
 
# 3. Perform Granger causality test
# Null Hypothesis: X does NOT Granger-cause Y
print("\nGranger Causality Test: Does X → Y?")
grangercausalitytests(df[['y', 'x']], maxlag=5, verbose=True)
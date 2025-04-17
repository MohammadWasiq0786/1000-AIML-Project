"""
Project 512: Bond Yield Curve Analysis
Description:
The bond yield curve is a graphical representation of the relationship between bond yields (interest rates) and the time to maturity of the bonds. Analyzing the yield curve is crucial for understanding market expectations of interest rates, economic growth, and inflation. In this project, we will analyze the yield curve by plotting the yield vs. maturity for government bonds and interpreting its shape to derive insights.

For real-world applications:

Use data from government bonds (e.g., U.S. Treasury bonds) with different maturities.

Extend this project by analyzing yield curve inversions and their implications for economic forecasting.

About:
✅ What It Does:
Simulates bond yields for different maturities (from 1 year to 30 years) using random noise added to realistic yield data.

Plots the bond yield curve (yield vs. maturity) and visualizes it using matplotlib.

Analyzes the shape of the yield curve and classifies it as:

Upward sloping: Typical market condition where long-term yields are higher than short-term yields.

Downward sloping (inverted): Often seen as an indicator of a potential recession.

Flat or mixed slopes: Suggests uncertainty or a transitioning market.

Key Extensions and Customizations:
Real-world data: Use U.S. Treasury yields or Eurozone bond data from sources like FRED (Federal Reserve Economic Data) or Bloomberg.

Yield curve inversions: Extend this project to track yield curve inversions and analyze their implications on the economy.

Advanced analysis: Use econometric models to analyze the relationship between the yield curve and macroeconomic indicators like GDP growth or inflation.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# 1. Simulate bond yield data for different maturities (short-term to long-term)
maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30])  # Bond maturities in years
# Simulating yields (in %) for each maturity. Normally, long-term bonds have higher yields than short-term ones.
yields = np.array([1.5, 1.7, 1.8, 2.0, 2.3, 2.5, 3.0, 3.5]) + np.random.normal(0, 0.1, len(maturities))  # Adding some randomness
 
# 2. Plot the bond yield curve
plt.figure(figsize=(10, 6))
plt.plot(maturities, yields, marker='o', color='b', label='Bond Yield Curve')
plt.title('Bond Yield Curve Analysis')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.grid(True)
plt.xticks(maturities)  # Ensures that all maturity years are shown on the x-axis
plt.yticks(np.arange(1, 4, 0.25))  # Adjust y-axis ticks for better visualization
plt.legend(loc="best")
plt.show()
 
# 3. Analyze the shape of the yield curve
if np.all(yields[1:] > yields[:-1]):
    print("The yield curve is upward sloping, indicating normal market conditions (long-term rates > short-term rates).")
elif np.all(yields[1:] < yields[:-1]):
    print("The yield curve is downward sloping (inverted), indicating potential recessionary fears.")
else:
    print("The yield curve is flat or has mixed slopes, suggesting uncertain or transitioning market conditions.")
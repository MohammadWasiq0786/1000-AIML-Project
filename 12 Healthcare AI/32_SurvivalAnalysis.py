"""
Project 472. Survival analysis implementation
Description:
Survival analysis models the time until an event occurs—commonly used in healthcare to estimate time to death, disease recurrence, or hospital discharge. In this project, we implement a basic Kaplan-Meier estimator and Cox Proportional Hazards model to study survival time with risk factors.

About:
✅ What It Does:
Uses Kaplan-Meier to visualize survival over time.

Fits a Cox regression model to identify which features (age, income, etc.) affect hazard rates.

Can be extended to:

Include medical covariates (e.g., tumor stage, treatment type)

Stratify by groups (e.g., cancer type, gender)

Deploy with dashboards for clinical interpretation

For real-world data:

Use TCGA clinical datasets, SEER, or hospital EHR discharge summaries
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.datasets import load_rossi
import matplotlib.pyplot as plt
 
# 1. Use a sample dataset from lifelines (can simulate or replace with real medical data)
df = load_rossi()
 
# Columns:
#  - week: duration until event or censoring
#  - arrest: 1 = event occurred (re-arrest), 0 = censored
#  - age, fin (financial aid), etc.
 
# 2. Kaplan-Meier Estimator (univariate survival curve)
kmf = KaplanMeierFitter()
kmf.fit(durations=df["week"], event_observed=df["arrest"])
 
# Plot survival curve
kmf.plot_survival_function()
plt.title("Kaplan-Meier Survival Curve")
plt.xlabel("Weeks")
plt.ylabel("Survival Probability")
plt.show()
 
# 3. Cox Proportional Hazards Model (multivariate regression)
cph = CoxPHFitter()
cph.fit(df, duration_col="week", event_col="arrest")
print("\nCox Model Summary:\n")
cph.print_summary()
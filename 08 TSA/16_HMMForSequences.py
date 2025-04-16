"""
Project 296. Hidden Markov models for sequences
Description:
A Hidden Markov Model (HMM) is a statistical model that assumes an underlying system follows a Markov process with hidden states. You only observe emissions (outputs), and the goal is to infer:

The most likely sequence of hidden states

The transition and emission probabilities

We’ll simulate a simple HMM with two hidden states (e.g., "Bull" and "Bear" market), then use the Viterbi algorithm to decode the hidden sequence.

About:
✅ What It Does:
Simulates a sequence with hidden regimes (bull vs bear)

Learns transition probabilities and emission distributions

Applies the Viterbi algorithm to decode the most probable hidden states

Visualizes the regime switch alongside actual observations
"""



import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
 
# 1. Simulated time series with hidden regimes (2-state HMM)
np.random.seed(42)
 
# Hidden states: 0 = Bull market (positive returns), 1 = Bear market (negative returns)
n_samples = 300
hidden_states = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
 
# Define emission means and stds
means = [0.02, -0.02]     # Bull (upward trend), Bear (downward trend)
stds = [0.01, 0.02]
 
# Generate observations (daily returns)
observations = np.array([
    np.random.normal(means[state], stds[state]) for state in hidden_states
]).reshape(-1, 1)
 
# 2. Fit Gaussian HMM (with 2 hidden states)
model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
model.fit(observations)
 
# Decode hidden states with Viterbi
predicted_states = model.predict(observations)
 
# 3. Plot results
plt.figure(figsize=(12, 4))
plt.plot(observations, label="Observed Returns")
plt.plot(predicted_states * 0.05, label="Inferred Hidden State (scaled)", linewidth=2)
plt.title("Hidden Markov Model – Regime Detection")
plt.xlabel("Time")
plt.ylabel("Returns")
plt.legend()
plt.grid(True)
plt.show()
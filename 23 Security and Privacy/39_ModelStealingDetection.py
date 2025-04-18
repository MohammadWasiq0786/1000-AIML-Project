"""
Project 919. Model Stealing Detection

Model stealing detection aims to identify when a user or attacker is querying a model in a way that suggests they're trying to replicate or reverse-engineer it. In this project, we simulate detection by analyzing query patterns—such as frequency, diversity, or entropy of inputs.

What This Does:
Measures input entropy to detect if someone is exploring the model’s full input space (a sign of stealing)

Flags abnormally diverse or systematic queries

🚨 Real-world detection signals:

Unusually high query rates

Rare combinations of input values

Model responses used to train a surrogate model

🔒 In practice:

Use rate limiting, randomized responses, or API fingerprinting

Deploy honey-triggers (trap queries) to track suspicious users
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
 
# Simulated user query dataset (each row is a flattened input vector)
np.random.seed(42)
legit_queries = np.random.normal(loc=0, scale=1, size=(100, 10))     # normal distribution
stealing_queries = np.random.uniform(low=-5, high=5, size=(100, 10))  # high diversity, exhaustive probing
 
# Combine into a DataFrame with user labels
queries = np.vstack((legit_queries, stealing_queries))
users = ['user1'] * 100 + ['userX'] * 100
df = pd.DataFrame(queries)
df['User'] = users
 
# Function to estimate query diversity (entropy across feature bins)
def compute_query_entropy(user_df):
    hist = []
    for col in user_df.columns[:-1]:  # exclude 'User'
        counts, _ = np.histogram(user_df[col], bins=10, range=(-5, 5))
        hist.append(counts + 1e-9)  # avoid log(0)
    stacked = np.stack(hist)
    return entropy(stacked.mean(axis=0))  # average feature entropy
 
# Evaluate entropy per user
results = df.groupby('User').apply(compute_query_entropy).reset_index()
results.columns = ['User', 'EntropyScore']
 
# Flag suspicious activity (high entropy = suspicious)
threshold = 2.0
results['Suspicious'] = results['EntropyScore'] > threshold
 
print("🛡️ Model Stealing Detection Report:")
print(results)
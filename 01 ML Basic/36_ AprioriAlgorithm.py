"""
Project 36. Apriori algorithm implementation
Description:
The Apriori algorithm is a classic algorithm for frequent itemset mining and association rule learning. It identifies items that frequently co-occur in transactional datasets, such as products bought together. This project uses the Apriori algorithm (via mlxtend) to mine frequent itemsets and generate interpretable association rules from sample transactions.

About:
🧠 What This Project Shows:
How to convert transactional data into a format suitable for rule mining

Use of Apriori to extract frequent itemsets

Generates interpretable association rules with confidence and lift metrics
"""

# Install mlxtend if not already installed:
# pip install mlxtend
 
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
 
# Sample transaction data
transactions = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter'],
    ['bread', 'jam'],
    ['milk', 'jam'],
    ['bread', 'butter', 'jam'],
    ['milk', 'bread', 'jam'],
    ['butter']
]
 
# Step 1: Convert to one-hot encoded dataframe
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
 
# Step 2: Apply Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print("🔍 Frequent Itemsets:\n")
print(frequent_itemsets)
 
# Step 3: Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
 
# Display rules
print("\n📊 Association Rules:\n")
for _, row in rules.iterrows():
    ant = ', '.join(list(row['antecedents']))
    con = ', '.join(list(row['consequents']))
    print(f"If a user buys [{ant}], they are likely to buy [{con}] — "
          f"support: {row['support']:.2f}, confidence: {row['confidence']:.2f}, lift: {row['lift']:.2f}")
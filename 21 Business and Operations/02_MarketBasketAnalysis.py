"""
Project 802. Market Basket Analysis

This project aims to identify associations between products frequently bought together using association rule mining. It's widely used in retail to create recommendation systems, optimize store layout, or design cross-selling strategies. We'll use the Apriori algorithm to extract frequent itemsets and generate association rules from transaction data.

This code performs classic market basket analysis, revealing patterns like:
“If a customer buys milk and bread, they are likely to also buy eggs.”
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
 
# Sample transaction dataset (each sublist is a customer's basket)
transactions = [
    ['milk', 'bread', 'eggs'],
    ['beer', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter', 'eggs'],
    ['beer', 'bread']
]
 
# Convert transaction data to a format suitable for analysis
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
 
# Generate frequent itemsets using Apriori algorithm
# Set minimum support (frequency) threshold
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
 
# Generate association rules from frequent itemsets
# Set minimum confidence threshold to filter strong rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
 
# Display the resulting rules
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
"""
Project 35. Association rule mining
Description:
Association Rule Mining is used to uncover interesting relationships between variables in large datasets. It’s commonly applied in market basket analysis to find product combinations that frequently co-occur. In this project, we’ll use the Apriori algorithm to extract frequent itemsets and generate association rules using the mlxtend library.

About:

🛒 What This Project Demonstrates:
Applies the Apriori algorithm to find frequent itemsets

Generates association rules with support, confidence, and lift

Uses a transactional-style dataset and one-hot encoding
"""

# Install if not already installed:
# pip install mlxtend
 
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
 
# Sample transactional dataset
dataset = [
    ['milk', 'bread', 'butter'],
    ['beer', 'bread'],
    ['milk', 'bread', 'butter'],
    ['beer', 'bread', 'milk'],
    ['bread', 'butter']
]
 
# Convert dataset to one-hot encoded DataFrame
from mlxtend.preprocessing import TransactionEncoder
 
te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_array, columns=te.columns_)
 
# Generate frequent itemsets with min support of 0.5
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print("Frequent Itemsets:\n", frequent_itemsets)
 
# Generate association rules with min confidence of 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
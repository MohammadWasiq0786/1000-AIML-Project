"""
Project 49. Market basket analysis tool
Description:
Market Basket Analysis is a data mining technique used to identify purchase patterns by analyzing customer transactions. This project builds a rule-based tool using the Apriori algorithm to discover frequent itemsets and generate association rules (like "If X is bought, Y is likely to be bought too").

About:
🧠 What This Project Demonstrates:
Converts transactional data into one-hot format

Applies Apriori algorithm to find frequent itemsets

Generates association rules with metrics like support, confidence, and lift

Helps uncover actionable insights for cross-selling
"""

# Install mlxtend if not installed:
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
 
# Encode transactions into one-hot matrix
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
 
# Generate frequent itemsets with minimum support of 0.3
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
 
# Generate association rules with minimum confidence of 0.6
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
 
# Display meaningful results
rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print("\n📊 Association Rules (Market Basket):\n")
for _, row in rules_display.iterrows():
    ant = ', '.join(list(row['antecedents']))
    con = ', '.join(list(row['consequents']))
    print(f"If a customer buys [{ant}], they are likely to buy [{con}] "
          f"(support: {row['support']:.2f}, confidence: {row['confidence']:.2f}, lift: {row['lift']:.2f})")
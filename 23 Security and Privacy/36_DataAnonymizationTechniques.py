"""
Project 916. Data Anonymization Techniques

Data anonymization removes or masks personally identifiable information (PII) while retaining data utility. In this project, we demonstrate key techniques including masking, generalization, and k-anonymity using a toy dataset.

Techniques Illustrated:
Masking: Hides direct identifiers (e.g., name)

Generalization: Reduces precision of quasi-identifiers (e.g., ZIP, age)

K-anonymity prep: Ensures individuals can't be re-identified from combinations

🔐 For advanced anonymization:

Use sdcMicro, ARX, or smartnoise

Apply k-anonymity, l-diversity, or t-closeness

Combine with differential privacy for stronger guarantees
"""

import pandas as pd
 
# Sample dataset with sensitive info
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [29, 34, 32, 33],
    'ZIP': ['12345', '12346', '12347', '12348'],
    'Diagnosis': ['Flu', 'Cold', 'Asthma', 'Flu']
}
 
df = pd.DataFrame(data)
print("🔍 Original Data:")
print(df)
 
# Technique 1: Masking names (replace with pseudonyms or drop)
df['Name'] = ['P1', 'P2', 'P3', 'P4']  # or df.drop('Name', axis=1)
 
# Technique 2: Generalizing ZIP code (e.g., keep only first 3 digits)
df['ZIP'] = df['ZIP'].apply(lambda z: z[:3] + 'XX')
 
# Technique 3: Age generalization (into bins)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40], labels=['20-30', '30-40'])
df.drop('Age', axis=1, inplace=True)
 
print("\n🛡️ Anonymized Data:")
print(df)
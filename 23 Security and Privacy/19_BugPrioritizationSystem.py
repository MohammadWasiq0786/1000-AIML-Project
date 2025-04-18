"""
Project 899. Bug Prioritization System

A bug prioritization system classifies reported bugs by their urgency or importance so that development teams can resolve the most critical issues first. In this project, we simulate a dataset of bug reports and use a text classification model to predict priority levels (e.g., Low, Medium, High).

Why It’s Useful:
Prioritizes bugs automatically based on semantic analysis of their descriptions.

Can be integrated with bug tracking systems (e.g., Jira, GitHub Issues).

📈 Advanced ideas:

Include features like frequency of reports, affected users, or crash logs.

Use BERT or other transformers for more context-aware classification.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated bug report descriptions and priority levels
data = {
    'Description': [
        "App crashes on startup",
        "Minor visual glitch on settings screen",
        "Unable to complete checkout process",
        "Typos in help section text",
        "Security vulnerability in login module",
        "Button misaligned on mobile view",
        "Database timeout under load",
        "Spelling mistake in error message"
    ],
    'Priority': [
        'High', 'Low', 'High', 'Low', 'High', 'Medium', 'High', 'Low'
    ]
}
 
df = pd.DataFrame(data)
 
# Convert text descriptions to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Description'])
y = df['Priority']
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train the classifier
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Evaluate model
y_pred = model.predict(X_test)
print("Bug Prioritization Report:")
print(classification_report(y_test, y_pred))
 
# Predict priority for a new bug description
new_bug = ["The app freezes when switching tabs rapidly"]
new_vec = vectorizer.transform(new_bug)
predicted_priority = model.predict(new_vec)[0]
print(f"\nPredicted Priority: {predicted_priority}")
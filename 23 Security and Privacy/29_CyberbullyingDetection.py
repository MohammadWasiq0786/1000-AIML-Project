"""
Project 909. Cyberbullying Detection

Cyberbullying detection identifies harmful, harassing, or abusive messages directed at individuals—often on social media or messaging platforms. In this project, we simulate chat data and use a text classification model to detect cyberbullying using supervised learning.

What It Does:
Flags harmful messages using keyword patterns

Uses TF-IDF + Logistic Regression for fast and interpretable results

💡 To make it production-grade:

Add contextual features (replies, targets)

Use deep learning (LSTM, BERT) for nuanced language

Include real-time filters in chat systems or comment sections
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated chat messages dataset
data = {
    'Message': [
        "You're so ugly and useless.",
        "Let's team up and win this game!",
        "Nobody likes you. Just quit already.",
        "Great job today, keep it up!",
        "You're the worst player I've ever seen.",
        "That was an awesome play!",
        "Why are you even in this group? You're trash.",
        "Thanks for the help earlier!"
    ],
    'Label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = cyberbullying, 0 = normal
}
 
df = pd.DataFrame(data)
 
# Text vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Message'])
y = df['Label']
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train classification model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Evaluate performance
y_pred = model.predict(X_test)
print("Cyberbullying Detection Report:")
print(classification_report(y_test, y_pred))
 
# Test on a new message
new_msg = ["You're such a loser. Go cry to someone else."]
new_vec = vectorizer.transform(new_msg)
prediction = model.predict(new_vec)[0]
print(f"\nPrediction: {'Cyberbullying ⚠️' if prediction == 1 else 'Normal ✅'}")
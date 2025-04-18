"""
Project 906. Spam Detection System

A spam detection system classifies incoming messages (e.g., emails, comments) as spam or ham (not spam). In this project, we simulate message content and train a simple text classification model using TfidfVectorizer + Naive Bayes.

Why This Works:
TF-IDF captures word importance and spam-specific language.

Naive Bayes works well with high-dimensional text features.

📩 In real-world applications:

Use preprocessing (stopwords, lemmatization)

Detect obfuscation (e.g., “fr33 m0ney”)

Upgrade to transformer models for contextual spam detection
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated message dataset
data = {
    'Message': [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim.",
        "Meeting scheduled for 3 PM today. Please be on time.",
        "Lowest price Viagra! Buy now!",
        "Can you review the attached report before tomorrow?",
        "URGENT! Your account has been suspended. Update now.",
        "Let’s catch up over lunch next week.",
        "Get rich quick with this crypto investment trick!",
        "Please find the invoice attached for last month’s services."
    ],
    'Label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}
 
df = pd.DataFrame(data)
 
# Convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Message'])
y = df['Label']
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train spam classifier
model = MultinomialNB()
model.fit(X_train, y_train)
 
# Evaluate model
y_pred = model.predict(X_test)
print("Spam Detection Report:")
print(classification_report(y_test, y_pred))
 
# Predict on new message
new_msg = ["You have been selected for a $500 gift card! Act now!"]
new_vec = vectorizer.transform(new_msg)
prediction = model.predict(new_vec)[0]
print(f"\nPredicted Label: {prediction}")
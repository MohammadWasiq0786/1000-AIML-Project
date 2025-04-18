"""
Project 908. Hate Speech Detection

Hate speech detection systems identify content that promotes violence, discrimination, or hostility toward individuals or groups based on race, religion, gender, or identity. In this project, we simulate social media posts and use a text classification model to label whether a post contains hate speech.

Key Features:
TF-IDF converts text into a vector of word importance

Logistic Regression classifies based on weighted patterns

Basic model for binary hate speech detection

🧠 Advanced Approaches:

Use transformer models like BERT, HateBERT, or DistilBERT

Incorporate context, user history, and target detection

Integrate with moderation pipelines and flagging thresholds
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated dataset of online posts
data = {
    'Post': [
        "I hate those people. They are ruining everything.",
        "Let's spread love and respect for everyone.",
        "Those idiots shouldn't be allowed to speak.",
        "We all deserve equal rights and kindness.",
        "Kick them out! They don't belong here.",
        "Had a great day volunteering at the shelter.",
        "They are disgusting and should be banned.",
        "Unity and peace are what we need."
    ],
    'Label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = hate speech, 0 = acceptable
}
 
df = pd.DataFrame(data)
 
# Text preprocessing and feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Post'])
y = df['Label']
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train logistic regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Evaluate model
y_pred = model.predict(X_test)
print("Hate Speech Detection Report:")
print(classification_report(y_test, y_pred))
 
# Predict on a new post
new_post = ["These people are a threat and should be silenced."]
new_vec = vectorizer.transform(new_post)
prediction = model.predict(new_vec)[0]
print(f"\nPredicted Label: {'Hate Speech' if prediction == 1 else 'Acceptable'}")
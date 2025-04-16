"""
Project 187. Fake news detector
Description:
A Fake News Detector classifies news articles or headlines as real or fake using machine learning. It helps identify misinformation and improve content credibility. In this project, we train a simple Logistic Regression classifier using TF-IDF features on a labeled dataset of news articles.

About:
🧠 What This Project Demonstrates:
Vectorizes text using TF-IDF features

Trains a binary classifier (real vs. fake) using Logistic Regression

Evaluates with a classification report and tests on custom input
"""

# Install if not already: pip install pandas scikit-learn
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
# Example dataset (replace with real dataset for production use)
data = {
    'text': [
        "The president announces a new education policy today.",
        "Aliens have landed in Ohio, officials confirm.",
        "NASA reveals details about upcoming Mars mission.",
        "COVID-19 vaccines cause people to grow antennas.",
        "Stock market hits all-time high amid tech surge."
    ],
    'label': [1, 0, 1, 0, 1]  # 1 = Real, 0 = Fake
}
 
df = pd.DataFrame(data)
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
 
# TF-IDF vectorizer + Logistic Regression model
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
 
model = LogisticRegression()
model.fit(X_train_vec, y_train)
 
# Evaluate model
y_pred = model.predict(X_test_vec)
print("🧠 Fake News Detection Report:\n")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
 
# Try on new headline
new_text = ["Government confirms alien spacecrafts in Area 51."]
new_vec = vectorizer.transform(new_text)
prediction = model.predict(new_vec)[0]
print("\n🔍 New Headline Prediction:")
print(f"'{new_text[0]}' → {'Real' if prediction == 1 else 'Fake'}")
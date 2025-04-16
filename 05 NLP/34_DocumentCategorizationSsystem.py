"""
Project 194. Document categorization system
Description:
A Document Categorization System classifies documents into predefined categories (e.g., sports, politics, technology, finance) based on their content. It's commonly used in content management systems, news portals, and email sorting. In this project, we build a multi-class text classifier using TF-IDF and Logistic Regression.

About:
🧠 What This Project Demonstrates:
Performs multi-class document classification

Uses TF-IDF features and a Logistic Regression model

Categorizes documents into predefined topical classes
"""

# Install if not already: pip install scikit-learn pandas
 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Sample labeled dataset
data = {
    "text": [
        "The government has passed a new tax reform bill.",
        "The soccer team won the championship after a thrilling match.",
        "Apple released its latest iPhone with new AI features.",
        "The stock market closed higher today due to strong tech earnings.",
        "The basketball game was intense with overtime action."
    ],
    "category": [
        "politics",
        "sports",
        "technology",
        "finance",
        "sports"
    ]
}
 
df = pd.DataFrame(data)
 
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["category"], test_size=0.3, random_state=42)
 
# Define TF-IDF + Logistic Regression pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])
 
# Train the model
model.fit(X_train, y_train)
 
# Predict on test set
y_pred = model.predict(X_test)
 
# Print evaluation results
print("🧠 Document Categorization Report:\n")
print(classification_report(y_test, y_pred))
 
# Predict new document category
new_doc = ["Tesla plans to launch a new electric truck next year."]
predicted_category = model.predict(new_doc)[0]
print(f"\n🔍 Predicted Category for New Document:\n'{new_doc[0]}' → {predicted_category}")
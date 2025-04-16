"""
Project 180. Intent recognition for chatbots
Description:
Intent recognition is a core component of chatbots that classifies a user’s input into a predefined intent (e.g., greeting, order_status, cancel_reservation). In this project, we implement a basic intent classifier using TF-IDF vectorization and Logistic Regression, ideal for command-based or FAQ-style chatbots.

About:
🧠 What This Project Demonstrates:
Classifies user input into predefined intents

Uses TF-IDF features and a Logistic Regression classifier

Forms the foundation for chatbot response logic
"""

# Install if not already: pip install scikit-learn
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Sample training data (intent-labeled)
training_data = [
    ("hi there", "greeting"),
    ("hello", "greeting"),
    ("goodbye", "farewell"),
    ("see you later", "farewell"),
    ("track my order", "order_status"),
    ("where is my package", "order_status"),
    ("cancel my order", "cancel_order"),
    ("i want to cancel the reservation", "cancel_order"),
    ("what's the weather like", "weather"),
    ("is it going to rain today", "weather"),
]
 
# Split into features and labels
texts, labels = zip(*training_data)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
 
# Create a pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])
 
# Train the model
model.fit(X_train, y_train)
 
# Evaluate the model
y_pred = model.predict(X_test)
print("🧠 Intent Recognition Results:\n")
print(classification_report(y_test, y_pred))
 
# Try with new inputs
test_inputs = [
    "hey there",
    "i want to cancel my order",
    "will it rain tomorrow?",
    "goodbye for now"
]
 
print("🔍 Predicted Intents:")
for inp in test_inputs:
    intent = model.predict([inp])[0]
    print(f"💬 \"{inp}\" → Intent: {intent}")
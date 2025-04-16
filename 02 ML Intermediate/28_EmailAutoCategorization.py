"""
Project 68. Email auto-categorization
Description:
Email auto-categorization automatically classifies emails into predefined folders like "Work", "Promotions", "Personal", or "Spam". This project builds a multi-class classifier using TF-IDF features and a Logistic Regression model trained on the subject and body of simulated emails.

About:
📬 What This Project Demonstrates:
Transforms raw email content into TF-IDF vectors

Trains a multi-class classifier to categorize messages

Uses Logistic Regression with text data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
 
# Simulated email dataset
data = {
    'EmailText': [
        "Meeting rescheduled to Monday at 9am.",
        "50% off your next online order! Limited time offer.",
        "Don't forget Mom's birthday dinner tomorrow night.",
        "Important: Your account security alert.",
        "Team sync-up call today at 4pm.",
        "Win a free trip to Paris – click now!",
        "Dinner with John next weekend?",
        "Quarterly financial results are attached.",
        "Exclusive deal just for you. Shop now!",
        "Your subscription has been renewed successfully."
    ],
    'Category': [
        'Work', 'Promotions', 'Personal', 'Work', 'Work',
        'Spam', 'Personal', 'Work', 'Promotions', 'Work'
    ]
}
 
df = pd.DataFrame(data)
 
# Encode labels
df['Category_Label'] = df['Category'].astype('category').cat.codes
label_map = dict(enumerate(df['Category'].astype('category').cat.categories))
 
# Vectorize email text
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['EmailText'])
y = df['Category_Label']
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_map.values()))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title("Confusion Matrix - Email Categorization")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
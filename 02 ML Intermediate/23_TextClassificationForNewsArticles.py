"""
Project 63. Text classification for news articles
Description:
Text classification for news articles involves assigning categories (e.g., politics, sports, tech) to news content automatically. In this project, we build a multi-class text classifier using TF-IDF vectorization and a Multinomial Naive Bayes model to classify simulated news headlines into categories.

About:
🗞️ What This Project Demonstrates:
Uses TF-IDF to convert text into numerical features

Trains a Multinomial Naive Bayes model for multi-class classification

Evaluates performance with detailed metrics and confusion matrix
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
# Simulated news dataset
data = {
    'Headline': [
        'Elections bring new political changes in the country',
        'Champions League final ends in dramatic penalty shootout',
        'Tech companies unveil latest AI breakthroughs',
        'Stock markets rally after economic recovery signs',
        'Scientists discover new exoplanets in distant galaxy',
        'Government announces new tax reforms',
        'Local football club wins national championship',
        'Startups boom in AI-driven healthcare sector',
        'Finance ministers debate inflation strategies',
        'NASA prepares for lunar exploration mission'
    ],
    'Category': [
        'Politics',
        'Sports',
        'Technology',
        'Finance',
        'Science',
        'Politics',
        'Sports',
        'Technology',
        'Finance',
        'Science'
    ]
}
 
df = pd.DataFrame(data)
 
# Encode categories into numeric labels
df['Category_Label'] = df['Category'].astype('category').cat.codes
label_map = dict(enumerate(df['Category'].astype('category').cat.categories))
 
# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Headline'])
y = df['Category_Label']
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_map.values()))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title("Confusion Matrix - News Article Classification")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()
plt.show()
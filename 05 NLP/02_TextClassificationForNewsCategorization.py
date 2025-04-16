"""
Project 162. Text classification for news categorization
Description:
News categorization is a classic NLP task where articles are classified into topics like sports, politics, tech, or business. In this project, we implement a multiclass text classifier using the 20 Newsgroups dataset with a TF-IDF vectorizer and Multinomial Naive Bayes classifier.

About:
🧠 What This Project Demonstrates:
Uses TF-IDF features for news text vectorization

Trains a Multinomial Naive Bayes model for multiclass classification

Evaluates with classification report and confusion matrix
"""

# Install if not already: pip install scikit-learn matplotlib
 
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
 
# Load dataset (subset of 4 categories for simplicity)
categories = ['rec.sport.baseball', 'sci.med', 'comp.graphics', 'talk.politics.mideast']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)
 
# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
 
# Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test_vec)
 
print("📰 Classification Report:")
print(classification_report(y_test, y_pred, target_names=categories))
 
# Visualize confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(model, X_test_vec, y_test, display_labels=categories, cmap='Blues')
plt.xticks(rotation=45)
plt.title("Confusion Matrix for News Categorization")
plt.tight_layout()
plt.show()